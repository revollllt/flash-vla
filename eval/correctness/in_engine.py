"""In-engine correctness of any Target: the reference plan against a candidate.

    python -m eval.correctness.in_engine --target h100/pi05 --steps 1 --layers 1
    python -m eval.correctness.in_engine --target h100/pi05 --steps 1 --layers 18
    python -m eval.correctness.in_engine --target h100/pi05 --plan lab/plans/pi05-attn-cuda.json --isolate

Two runners of one Target are built on the same seeded weights and fed the
same seeded inputs; one runs the reference plan, the other the candidate
(the shipped plan unless `--plan` names another). The
program runs in lockstep, one step at a time, and after every segment the
outputs the Target declares for it (`engine.stage_outputs`) are compared with
the shared five metrics, per layer where the Target marks a layer axis.

`--isolate` injects the reference's outputs into the candidate after each
comparison, so a segment is judged on its own inputs and upstream drift does
not mix into its number. Without it the comparison is cumulative, which is the
deployment reading and is read after the isolated one passes.

Gates follow the acceptance registry: the single-step, single-layer run gates
on the shallow cosine threshold; deeper runs are reports, because a
denoising loop on random weights is a chaotic map and any two
implementations that are not bit-identical separate. Replay determinism and
finiteness of every padded allocation behind a declared output gate at every
depth.

The runner contains no model or stage names.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from benchmarks.targets import PLAN_NAMES, build, resolve
from eval.acceptance import DEFAULTS, tolerances
from eval.correctness.metrics import error_metrics


def _compare(name: str, reference: torch.Tensor, candidate: torch.Tensor,
             layer_axis: int | None, active_layers: int | None) -> dict[str, Any]:
    """The shared metrics, plus a per-layer cosine profile over the active depth.

    A layer-major buffer keeps slots for every layer of the full model; at a
    bisected depth the slots beyond `active_layers` are never written on
    either side, so the profile stops there rather than comparing zeros.
    """
    entry: dict[str, Any] = {"metrics": error_metrics(reference, candidate)}
    if layer_axis is not None:
        depth = reference.shape[layer_axis]
        if active_layers is not None:
            depth = min(depth, active_layers)
        per_layer = []
        for index in range(depth):
            ref_i = reference.select(layer_axis, index)
            got_i = candidate.select(layer_axis, index)
            per_layer.append(error_metrics(ref_i, got_i)["cosine_similarity"])
        entry["per_layer_cosine"] = per_layer
        steps = [per_layer[i] - per_layer[i + 1] for i in range(len(per_layer) - 1)]
        entry["max_cosine_step"] = max(steps) if steps else 0.0
    return entry


def run(target: str, plan: str | None = "shipped", steps: int | None = 1,
        layers: int | None = 1, seed: int = 0, isolate: bool = False,
        **overrides) -> dict[str, Any]:
    """Compare `plan` against the Target's reference plan, stage by stage.

    `steps` / `layers` of `None` mean the Target's full default depth.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this on a GPU node")
    target = resolve(target)
    tol = tolerances("bf16")
    depth = {k: v for k, v in (("steps", steps), ("layers", layers)) if v is not None}
    reference = build(target, "reference", seed=seed, **depth, **overrides)
    candidate = build(target, plan or "shipped", seed=seed, **depth, **overrides)
    steps = reference.identity.shape.get("steps", steps)
    layers = reference.identity.shape.get("layers", layers)
    if not reference.identity.same_workload(candidate.identity):
        raise ValueError("reference and candidate are not the same workload")
    inputs = reference.sample_inputs(seed)
    active_layers = reference.identity.shape.get("layers")

    # One full pass on both settles host-side state (a tokenized prompt sets
    # the valid prefix length) before the lockstep pass that is compared.
    reference.forward(**inputs)
    candidate.forward(**inputs)
    torch.cuda.synchronize()

    stages: dict[str, Any] = {}
    finite = True
    reference.stage(**inputs)
    candidate.stage(**inputs)
    for step in reference.program:
        if step.kind == "host":
            reference.host(step.name, **inputs)
            candidate.host(step.name, **inputs)
            continue
        reference.replay(step.name)
        candidate.replay(step.name)
        torch.cuda.synchronize()
        outputs = reference.stage_outputs.get(step.name, ())
        stages[step.name] = {}
        for name, layer_axis in outputs:
            ref, got = reference.buffers[name], candidate.buffers[name]
            stages[step.name][name] = _compare(name, ref, got, layer_axis, active_layers)
            allocation = candidate.allocation(name)
            if layer_axis is not None and active_layers is not None:
                allocation = allocation.narrow(layer_axis, 0,
                                               min(active_layers, allocation.shape[layer_axis]))
            is_finite = bool(torch.isfinite(allocation).all().item())
            stages[step.name][name]["finite_allocation"] = is_finite
            finite &= is_finite
            if isolate:
                got.copy_(ref)
        torch.cuda.synchronize()

    # Replay determinism: the candidate's whole forward twice, bit-identical.
    first = candidate.forward(**inputs).clone()
    torch.cuda.synchronize()
    second = candidate.forward(**inputs).clone()
    torch.cuda.synchronize()
    replay_identical = bool(torch.equal(first, second))
    ref_out = reference.forward(**inputs).clone()
    torch.cuda.synchronize()
    output = _compare("output", ref_out, second, None, active_layers)

    cosines = [output["metrics"]["cosine_similarity"]] + [
        entry["metrics"]["cosine_similarity"]
        for stage in stages.values() for entry in stage.values()]
    shallow = steps == 1 and layers == 1
    report = {
        "identity": {"reference": reference.identity.as_dict(),
                     "candidate": candidate.identity.as_dict()},
        "config": {"steps": steps, "layers": layers, "seed": seed, "isolate": isolate,
                   "oracle": "in_engine_reference"},
        "stages": stages,
        "output": output,
        "replay_identical": replay_identical,
        "finite": finite,
        "min_cosine": min(cosines),
        "threshold": tol["shallow_cosine"],
        "mode": "gate" if shallow else "report",
        "checks": [c["check"] for c in DEFAULTS["correctness"]["checks"]
                   if c.get("oracle") in (None, "in_engine_reference")],
    }
    report["passed"] = bool(replay_identical and finite
                            and (not shallow or min(cosines) > tol["shallow_cosine"]))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--plan", default="shipped",
                        help=f"candidate: one of {PLAN_NAMES}, a JSON object or a "
                             "lab/plans/*.json path (default: shipped)")
    parser.add_argument("--steps", type=int, default=1, help="0 = the Target's full depth")
    parser.add_argument("--layers", type=int, default=1, help="0 = the Target's full depth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--isolate", action="store_true",
                        help="inject the reference's stage outputs into the candidate")
    args = parser.parse_args(argv)
    report = run(args.target, args.plan, steps=args.steps or None, layers=args.layers or None,
                 seed=args.seed, isolate=args.isolate)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
