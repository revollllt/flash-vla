"""Compare expert residual projection fusion with controlled residual snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from benchmarks.kernels import _graph_samples
from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_residual, torch_ops
from flash_vla.inference import build, parse_options, resolve
from flash_vla.runtime.runner import Scratch


def _record(engine):
    calls = {name: [] for name in fused_residual.NAMES}

    def instrument(name, fn):
        if name not in calls:
            return fn

        def wrapped(x, weight, gate, out):
            calls[name].append((x.clone(), weight, gate, out.clone(),
                                torch.empty_like(out)))
            return fn(x, weight, gate, out)

        return wrapped

    with engine.instrument(instrument):
        engine.run_eager("action_expert")
    torch.cuda.synchronize()
    return calls


def _invoke(fn, call):
    x, weight, gate, residual, out = call
    # The identical reset is timed on both routes, avoiding cross-replay drift.
    out.copy_(residual)
    return fn(x, weight, gate, out)


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
    candidates = fused_residual.make_wrappers(scratch)
    thresholds = tolerances()["shallow"]
    sites = {}
    for name, invocations in calls.items():
        reference = getattr(torch_ops, name)
        candidate = candidates[name]
        correctness = []
        for index in (0, 17, len(invocations) // 2, len(invocations) - 1):
            call = invocations[index]
            expected = _invoke(reference, call).clone()
            actual = _invoke(candidate, call)
            metrics = error_metrics(expected, actual)
            correctness.append({"invocation": index, "output": metrics,
                                "exact": torch.equal(expected, actual)})
            assert metrics["rel_rms"] <= thresholds["rel_rms_max"], correctness[-1]
            assert metrics["cosine_similarity"] >= thresholds["cosine_min"], correctness[-1]
        scratch.freeze()
        timings = {}
        for route, fn in (("torch", reference), ("fused", candidate)):
            samples = _graph_samples(lambda i: _invoke(fn, invocations[i % len(invocations)]),
                                     n_inner=len(invocations), reps=args.reps)
            timings[route] = {"median_ms": statistics.median(samples), "samples_ms": samples}
        sites[name] = {"invocations": len(invocations), "correctness": correctness,
                       "timings": timings}
    report = {"seed": args.seed, "dtype": "bf16", "rows": 50, "sites": sites,
              "timer": "amortized CUDA graph, same residual reset copy on both routes",
              "cache": "all 180 recorded calls per site, 18 distinct layer weights",
              "scratch_bytes": scratch.nbytes, "identity": engine.identity.as_dict()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
