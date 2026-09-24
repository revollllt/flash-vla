"""Compare the RTX 5090 Pi0.5 FFN fusion at real recorded call-site inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from flash_vla.runtime.cuda.timing import graph_samples
from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_ffn, packed_ffn, torch_ops
from flash_vla.inference import build, resolve
from flash_vla.runtime.runner import Scratch
from measurement.cli import parse_options


def _record(engine):
    """Retain each invocation's activation, since the engine reuses that buffer."""
    calls = []

    def instrument(name, fn):
        if name != fused_ffn.NAMES[0]:
            return fn

        def wrapped(*args, **kwargs):
            x, scale, gate_w, up_w, gate_b, up_b, out, factor = args
            calls.append((x.clone(), scale, gate_w, up_w, gate_b, up_b,
                          torch.empty_like(out), torch.empty_like(factor)))
            return fn(*args, **kwargs)

        return wrapped

    with engine.instrument(instrument):
        engine.run_eager("action_expert")
    torch.cuda.synchronize()
    return calls


def _correctness(calls, candidate):
    thresholds = tolerances()["shallow"]
    reports = []
    for index in (0, 17, len(calls) // 2, len(calls) - 1):
        args = calls[index]
        torch_ops.action_expert_norm_gated_ffn(*args)
        expected = args[-2].clone()
        expected_factor = args[-1].clone()
        candidate(*args)
        metrics = error_metrics(expected, args[-2])
        factor_metrics = error_metrics(expected_factor[:args[0].shape[0]],
                                       args[-1][:args[0].shape[0]])
        reports.append({"invocation": index, "output": metrics,
                        "factor": factor_metrics})
        assert metrics["rel_rms"] <= thresholds["rel_rms_max"], reports[-1]
        assert metrics["cosine_similarity"] >= thresholds["cosine_min"], reports[-1]
        assert torch.equal(expected_factor[:args[0].shape[0]],
                           args[-1][:args[0].shape[0]]), reports[-1]
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate", choices=("fused", "packed"), default="fused")
    args = parser.parse_args()
    engine = build(resolve("rtx5090/pi05"), "reference", seed=args.seed,
                   **parse_options(args.option))
    engine.forward(**engine.sample_inputs(args.seed))
    calls = _record(engine)
    scratch = Scratch(torch.device("cuda"))
    backend = fused_ffn if args.candidate == "fused" else packed_ffn
    candidate = backend.make_wrappers(scratch)[fused_ffn.NAMES[0]]
    # Visit all layer pairs before freezing; packing must never occur in capture.
    for call in calls:
        candidate(*call)
    correctness = _correctness(calls, candidate)
    scratch.freeze()
    timings = {}
    comparison = torch_ops.action_expert_norm_gated_ffn
    comparison_name = "torch"
    if args.candidate == "packed":
        comparison_name = "fused"
        comparison = fused_ffn.make_wrappers(Scratch(torch.device("cuda")))[fused_ffn.NAMES[0]]
    for name, fn in ((comparison_name, comparison), (args.candidate, candidate)):
        samples = graph_samples(lambda i: fn(*calls[i % len(calls)]),
                                n_inner=len(calls), reps=args.reps)
        timings[name] = {"median_ms": statistics.median(samples), "samples_ms": samples}
    report = {"seed": args.seed, "shape": [50, 1024, 4096], "dtype": "bf16",
              "invocations": len(calls), "timer": "amortized CUDA graph",
              "cache": "recorded layer weights, cycling all 180 call-site invocations",
              "correctness": correctness, "timings": timings,
              "candidate": args.candidate, "scratch_bytes": scratch.nbytes,
              "identity": engine.identity.as_dict()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
