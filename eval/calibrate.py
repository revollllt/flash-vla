"""Calibrate the correctness tolerances from the natural dispersion of two promoted routes.

    python -m eval.calibrate --reports artifacts/.../correctness_pi05_1x1.json ...
    python -m eval.calibrate --dumps before/pi05_tilelang.pt before/pi05_pdl.pt

A tolerance is not a guess. Two implementations that both shipped differ only
by bf16 reduction order, and the size of that difference at the gate's depth
(one step, one layer) is what a wrong kernel must exceed to be caught. This
script reads either in-engine correctness reports (`eval.correctness`, which
carry every metric per stage output) or a pair of forward dumps, prints the
per-output `rel_rms` and `cosine_similarity`, and proposes tolerances at ten
times the worst dispersion (`rel_rms_max = 10 x worst rel_rms`,
`cosine_min = 1 - 10 x (1 - worst cosine)`). The registry records which run
produced its numbers. No GPU: reports and dumps are read on a CPU-only host.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.metrics import error_metrics

#: Pre-PR1 dump names -> the current stage-output names.
DUMP_NAMES = {
    "vision/vision_x": "vision_encoder/vision_encoder_x",
    "prefix/prefix_K": "llm_backbone/prefix_k",
    "prefix/prefix_V": "llm_backbone/prefix_v",
    "decoder/diffusion_noise": "action_expert/actions",
    "decoder/suffix_K": "action_expert/suffix_k",
    "decoder/suffix_V": "action_expert/suffix_v",
    "forward/diffusion_noise": "action_expert/actions",
    "forward/encoder_K": "kv_k",
    "forward/encoder_V": "kv_v",
}
FACTOR = 10.0


def from_report(path: str) -> dict[str, dict[str, float]]:
    """Per-output metrics of one `eval.correctness` report."""
    text = Path(path).read_text()
    report = json.loads(text[text.index("{"):])
    rows = {"output": report["output"]["metrics"]}
    for stage, outputs in report["stages"].items():
        for name, entry in outputs.items():
            rows[f"{stage}/{name}"] = entry["metrics"]
    return rows


def from_dumps(a: str, b: str) -> dict[str, dict[str, float]]:
    """Per-output metrics between two forward dumps of the same workload."""
    import torch
    x, y = torch.load(a, map_location="cpu"), torch.load(b, map_location="cpu")
    rows = {"output": error_metrics(x["second"], y["second"])}
    for key, tensor in x["stage"].items():
        name = DUMP_NAMES.get(key, key)
        if key in y["stage"]:
            rows[name] = error_metrics(tensor, y["stage"][key])
    return rows


def propose(rows: dict[str, dict[str, float]]) -> dict[str, Any]:
    worst_rel = max(m["rel_rms"] for m in rows.values())
    worst_cos = min(m["cosine_similarity"] for m in rows.values())
    return {"worst_rel_rms": worst_rel, "worst_cosine": worst_cos,
            "rel_rms_max": FACTOR * worst_rel,
            "cosine_min": 1.0 - FACTOR * (1.0 - worst_cos), "factor": FACTOR}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reports", nargs="*", default=[], help="eval.correctness report JSONs")
    parser.add_argument("--dumps", nargs=2, action="append", default=[],
                        metavar=("A", "B"), help="a pair of forward dumps")
    args = parser.parse_args(argv)
    sources: dict[str, dict[str, dict[str, float]]] = {}
    for path in args.reports:
        sources[path] = from_report(path)
    for a, b in args.dumps:
        sources[f"{a} vs {b}"] = from_dumps(a, b)
    if not sources:
        parser.error("give --reports and/or --dumps")
    out = {}
    for label, rows in sources.items():
        print(f"== {label}")
        print(f"{'output':44s} {'rel_rms':>10s} {'cosine':>12s} {'max_abs':>10s} {'p99_abs':>10s}")
        for name, m in rows.items():
            print(f"{name:44s} {m['rel_rms']:10.3e} {m['cosine_similarity']:12.9f} "
                  f"{m['max_abs']:10.3e} {m['p99_abs']:10.3e}")
        out[label] = {"rows": rows, "proposal": propose(rows)}
        print("   proposal:", json.dumps(out[label]["proposal"]))
    every = {name: m for v in out.values() for name, m in v["rows"].items()}
    overall = propose(every)
    print(json.dumps({"per_source": {k: v["proposal"] for k, v in out.items()},
                      "overall": overall}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
