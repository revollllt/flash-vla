"""Actual-call expert residual parity and reset-inclusive ABBA timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from benchmarks.kernels import _graph_samples
from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import cutlass_expert_residual, fused_ffn
from flash_vla.inference import build, parse_options, resolve
from flash_vla.runtime.runner import Scratch


def _record(engine, site):
    calls = []

    def instrument(name, fn):
        if name != site:
            return fn

        def wrapped(x, weight, gate, out):
            calls.append((x.clone(), weight, gate, out.clone(), torch.empty_like(out)))
            return fn(x, weight, gate, out)

        return wrapped

    with engine.instrument(instrument):
        engine.run_eager("action_expert")
    torch.cuda.synchronize()
    return calls


def _invoke(fn, call):
    x, weight, gate, residual, out = call
    out.copy_(residual)
    return fn(x, weight, gate, out)


def _check(calls, reference, candidate):
    thresholds = tolerances()["shallow"]
    correctness = []
    for index, call in enumerate(calls):
        expected = _invoke(reference, call).clone()
        actual = _invoke(candidate, call)
        metrics = error_metrics(expected, actual)
        correctness.append({"call": index, "output": metrics,
                            "exact": torch.equal(expected, actual)})
        assert metrics["rel_rms"] <= thresholds["rel_rms_max"], correctness[-1]
        assert metrics["cosine_similarity"] >= thresholds["cosine_min"], correctness[-1]

    # Gate=1 and C=0 extract this very same mainloop's BF16 projection. Apply
    # the existing separate native residual stage to isolate epilogue rounding.
    projection = torch.empty_like(calls[0][-1])
    ones = torch.ones_like(calls[0][2])
    native = fused_ffn._library()
    decomposition = []
    for index in (0, len(calls) // 2, len(calls) - 1):
        x, weight, gate, residual, out = calls[index]
        projection.zero_()
        candidate(x, weight, ones, projection)
        out.copy_(residual)
        fused_ffn._check(native.gated_residual_launch(
            projection.data_ptr(), gate.data_ptr(), out.data_ptr(), 50,
            torch.cuda.current_stream().cuda_stream), "gated_residual", 50)
        expected = out.clone()
        _invoke(candidate, calls[index])
        exact = torch.equal(expected, out)
        decomposition.append({"call": index, "exact": exact,
                              "output": error_metrics(expected, out)})
        assert exact, decomposition[-1]
    return correctness, decomposition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site", choices=cutlass_expert_residual.NAMES,
                        default=cutlass_expert_residual.NAMES[0])
    args = parser.parse_args()
    engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed,
                   **parse_options(args.option))
    engine.forward(**engine.sample_inputs(args.seed))
    calls = _record(engine, args.site)
    reference = getattr(engine.ops, args.site)
    scratch = Scratch(torch.device("cuda"))
    candidate = cutlass_expert_residual.make_wrappers(scratch, [args.site])[args.site]
    correctness, decomposition = _check(calls, reference, candidate)
    scratch.freeze()
    timings = []
    for route, fn in (("current", reference), ("candidate", candidate),
                      ("candidate", candidate), ("current", reference)):
        samples = _graph_samples(lambda i: _invoke(fn, calls[i % len(calls)]),
                                 n_inner=len(calls), reps=args.reps)
        timings.append({"route": route, "median_ms": statistics.median(samples),
                        "samples_ms": samples})
    report = {"seed": args.seed, "site": args.site,
              "shape": [*calls[0][0].shape, 1024], "dtype": "bf16",
              "calls": len(calls), "correctness": correctness, "decomposition": decomposition,
              "timings": timings, "timer": "reset-inclusive CUDA graph, ABBA",
              "cache": "180 recorded calls, 18 distinct layer weights",
              "scratch_bytes": scratch.nbytes, "identity": engine.identity.as_dict()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"timings": timings, "correctness_calls": len(correctness),
                      "max_rel_rms": max(c["output"]["rel_rms"] for c in correctness),
                      "min_cosine": min(c["output"]["cosine_similarity"] for c in correctness),
                      "decomposition": decomposition, "scratch_bytes": scratch.nbytes}, indent=2))


if __name__ == "__main__":
    main()
