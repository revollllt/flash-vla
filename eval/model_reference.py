"""Any Target against its model's end-to-end reference: python -m eval.model_reference.

    python -m eval.model_reference --target rtx5090/pi05 --steps 1 --layers 1
    python -m eval.model_reference --target rtx5090/pi05 --steps 0 --layers 0 \\
        --option converted_checkpoint=/path/to/pi05_libero_pytorch --option checkpoint_id=... ...

The engine is built as usual (`flash_vla.inference.build`) and runs one full
forward on its seeded inputs. The model's reference (`models/<model>/reference.py`,
plain torch on the official checkpoint layout) then runs the same observation on
the same model: the official-layout weights that construction ran
(`sources.official_weights`: the seeded random ones, or the checkpoint's), which
the engine only ever sees converted and folded. The model's `reference_view`
feeds the reference the engine's observation and pairs every declared stage
output with the reference's, in the engine's layout. The comparison therefore
covers the weight conversion, the fold, the graph and every kernel the plan
routes, against the model as upstream defines it.

The reference runs in float32 by default: the model's math without either
side's bfloat16 roundings, which is what the engine approximates.
`--reference-precision bfloat16` runs upstream's own inference dtypes instead
and reports how far the engine is from upstream's numerics, which sit about as
far from the float32 math as the engine does. At one step and one layer the
shared tolerances gate; deeper runs report, because two implementations of a
deep denoising loop that are not bit-identical drift apart. The runner
contains no model or stage names.
"""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
from typing import Literal, Mapping, Protocol

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.inference import PLAN_NAMES, TARGETS, build, resolve
from flash_vla.runtime.vla import ConfigValue
from measurement.cli import parse_options

ReferencePrecision = Literal["bfloat16", "float32"]


class WeightsSource(Protocol):
    """A model's `sources` module, as this runner reads it."""

    def official_weights(self, *, device: str, seed: int,
                         **construction: ConfigValue) -> dict[str, torch.Tensor]: ...


class ReferenceView(Protocol):
    """A model's `reference_view` module: the engine's observation fed to the
    reference, and each declared stage output paired as (reference, engine)."""

    def reference_outputs(self, weights: Mapping[str, torch.Tensor],
                          inputs: Mapping[str, torch.Tensor], buffers: Mapping[str, torch.Tensor], *,
                          steps: int, depth: int, precision: ReferencePrecision) -> object: ...

    def comparable(self, outputs: object, buffers: Mapping[str, torch.Tensor]
                   ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]: ...


def run(target: str, plan: str = "shipped", *, steps: int | None = 1, layers: int | None = 1,
        seed: int = 0, reference_precision: ReferencePrecision = "float32",
        **options: ConfigValue) -> dict[str, object]:
    """Compare Target `target` on `plan` with its model's reference, stage by stage."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this on a GPU node")
    entry = TARGETS[resolve(target)]
    if entry.reference_view_module is None:
        raise ValueError(f"{target} has no model reference yet")
    shape_options = {name: value for name, value in (("steps", steps), ("layers", layers))
                     if value is not None}
    engine = build(target, plan, seed=seed, **shape_options, **options)
    inputs = engine.sample_inputs(seed)
    engine.forward(**inputs)
    torch.cuda.synchronize()
    sources: WeightsSource = import_module(entry.sources_module)
    weights = sources.official_weights(device=str(engine.device), seed=seed, **options)
    view: ReferenceView = import_module(entry.reference_view_module)
    with torch.inference_mode():
        outputs = view.reference_outputs(weights, inputs, engine.buffers,
                                         steps=engine.identity.shape["steps"],
                                         depth=engine.identity.shape["layers"],
                                         precision=reference_precision)
    torch.cuda.synchronize()
    stages = {name: error_metrics(expected.float(), observed.float())
              for name, (expected, observed) in view.comparable(outputs, engine.buffers).items()}
    tolerance = tolerances(engine.identity.precision)["shallow"]
    min_cosine = min(metrics["cosine_similarity"] for metrics in stages.values())
    max_rel_rms = max(metrics["rel_rms"] for metrics in stages.values())
    shallow = steps == 1 and layers == 1
    within = bool(min_cosine > tolerance["cosine_min"] and max_rel_rms < tolerance["rel_rms_max"])
    return {
        "identity": engine.identity.as_dict(),
        "measurement_context": engine.measurement_context,
        "config": {"plan": plan, "steps": steps, "layers": layers, "seed": seed,
                   "reference_precision": reference_precision, "options": options,
                   "oracle": "model_reference"},
        "stages": stages,
        "min_cosine": min_cosine,
        "max_rel_rms": max_rel_rms,
        "tolerance": dict(tolerance),
        "within_tolerance": within,
        "mode": "gate" if shallow else "report",
        "passed": within or not shallow,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--plan", default="shipped", help=f"one of {PLAN_NAMES}, a JSON object or a path")
    parser.add_argument("--steps", type=int, default=1, help="0 = the Target's full depth")
    parser.add_argument("--layers", type=int, default=1, help="0 = the Target's full depth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-precision", choices=("bfloat16", "float32"), default="float32",
                        help="float32 throughout, the gate (the reference then holds a float32 "
                             "copy of the weights: about twice their memory), or upstream's "
                             "bfloat16 inference dtypes, a report")
    parser.add_argument("--option", action="append", default=[],
                        help="target construction option as key=value")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)
    report = run(args.target, args.plan, steps=args.steps or None, layers=args.layers or None,
                 seed=args.seed, reference_precision=args.reference_precision,
                 **parse_options(args.option))
    text = json.dumps(report, indent=2)
    print(text)
    status = 0 if report["passed"] else 1
    if args.out is None:
        return status
    Path(args.out).write_text(text)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
