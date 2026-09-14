"""Compare Pi0.5 prefix QKV fusion on captured actual backbone inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from benchmarks.kernels import _graph_samples
from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_prefix_qkv, torch_ops
from flash_vla.inference import build, parse_options, resolve
from flash_vla.runtime.runner import Scratch


def _record(engine):
    calls = []

    def instrument(name, fn):
        if name != fused_prefix_qkv.NAMES[0]:
            return fn

        def wrapped(x, weight, rope, Q, K, V, x_norm):
            calls.append((x.clone(), weight, rope, torch.empty_like(Q),
                          torch.empty_like(K), torch.empty_like(V), torch.empty_like(x_norm)))
            return fn(x, weight, rope, Q, K, V, x_norm)

        return wrapped

    with engine.instrument(instrument):
        engine.run_eager("llm_backbone")
    torch.cuda.synchronize()
    return calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    engine = build(resolve("rtx5090/pi05"), "reference", seed=args.seed,
                   **parse_options(args.option))
    engine.forward(**engine.sample_inputs(args.seed))
    calls = _record(engine)
    scratch = Scratch(torch.device("cuda"))
    candidate = fused_prefix_qkv.make_wrappers(scratch)[fused_prefix_qkv.NAMES[0]]
    reference = torch_ops.llm_backbone_norm_qkv_rope
    thresholds = tolerances()["shallow"]
    correctness = []
    for index in (0, len(calls) // 2, len(calls) - 1):
        call = calls[index]
        reference(*call)
        expected = [tensor.clone() for tensor in call[3:]]
        candidate(*call)
        outputs = {}
        for name, ref, actual in zip(("Q", "K", "V", "x_norm"), expected, call[3:]):
            metrics = error_metrics(ref, actual)
            outputs[name] = {"metrics": metrics, "exact": torch.equal(ref, actual)}
            assert metrics["rel_rms"] <= thresholds["rel_rms_max"], outputs[name]
            assert metrics["cosine_similarity"] >= thresholds["cosine_min"], outputs[name]
        correctness.append({"layer": index, "outputs": outputs})
    scratch.freeze()
    timings = {}
    for route, fn in (("torch", reference), ("fused", candidate)):
        samples = _graph_samples(lambda i: fn(*calls[i % len(calls)]),
                                 n_inner=len(calls), reps=args.reps)
        timings[route] = {"median_ms": statistics.median(samples), "samples_ms": samples}
    report = {"seed": args.seed, "dtype": "bf16", "rows": calls[0][0].shape[0],
              "calls": len(calls), "correctness": correctness, "timings": timings,
              "timer": "amortized CUDA graph, all18 layer weights",
              "scratch_bytes": scratch.nbytes, "identity": engine.identity.as_dict()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
