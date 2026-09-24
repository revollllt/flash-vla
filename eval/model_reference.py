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
side's bfloat16 roundings, which is what the engine approximates. It runs with
TF32 off for matmuls and cuDNN convolutions, whatever the process set: PyTorch
enables it for convolutions by default, and upstream code a plan loads may
enable it for matmuls (LingBot's policy does).
`--reference-precision bfloat16` runs upstream's own inference dtypes instead
and reports how far the engine is from upstream's numerics, which sit about as
far from the float32 math as the engine does. At one step and one layer
against the float32 reference the shared tolerances gate, and every compared
tensor must be finite; other runs report, because two implementations of a
deep denoising loop that are not bit-identical drift apart. The runner
contains no model or stage names.
"""
from __future__ import annotations

import argparse
import gc
from importlib import import_module
import json
from pathlib import Path
from typing import Literal, Mapping, Protocol, TypedDict

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.inference import PLAN_NAMES, TARGETS, build, resolve
from flash_vla.models.official import Precision
from flash_vla.runtime.vla import ConfigValue
from measurement.cli import parse_options


class WeightsSource(Protocol):
    """A model's `sources` module, as this runner reads it."""

    def official_weights(self, *, device: str, seed: int,
                         **construction: ConfigValue) -> dict[str, torch.Tensor]: ...


class ReferenceView(Protocol):
    """A model's `reference_view` module: the engine's observation fed to the
    reference, and each declared stage output paired as (reference, engine).
    `shape` is the engine's shape numbers (`identity.shape`), `seed` the
    construction's fixture seed."""

    def reference_outputs(self, weights: Mapping[str, torch.Tensor],
                          inputs: Mapping[str, torch.Tensor], buffers: Mapping[str, torch.Tensor], *,
                          shape: Mapping[str, int], seed: int, precision: Precision) -> object: ...

    def comparable(self, outputs: object, buffers: Mapping[str, torch.Tensor]
                   ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]: ...


class ReferenceConfig(TypedDict):
    plan: str
    steps: int | None
    layers: int | None
    seed: int
    reference_precision: Precision
    options: dict[str, ConfigValue]
    oracle: Literal["model_reference"]


class ReferenceReport(TypedDict):
    """One comparison: `stages` holds each declared stage output's error metrics."""
    identity: dict[str, object]
    measurement_context: dict[str, dict[str, str]]
    config: ReferenceConfig
    stages: dict[str, dict[str, float]]
    finite: bool
    min_cosine: float
    max_rel_rms: float
    tolerance: dict[str, float]
    within_tolerance: bool
    mode: Literal["gate", "report"]
    passed: bool


def run(target: str, plan: str = "shipped", *, steps: int | None = 1, layers: int | None = 1,
        seed: int = 0, reference_precision: Precision = "float32",
        **options: ConfigValue) -> ReferenceReport:
    """Compare Target `target` on `plan` with its model's reference, stage by stage."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this on a GPU node")
    entry = TARGETS[resolve(target)]
    shape_options = {name: value for name, value in (("steps", steps), ("layers", layers))
                     if value is not None}
    engine = build(target, plan, seed=seed, **shape_options, **options)
    inputs = engine.sample_inputs(seed)
    engine.forward(**inputs)
    torch.cuda.synchronize()
    identity, context, device = engine.identity, engine.measurement_context, str(engine.device)
    # Keep what the engine computed and release its weights and graphs: a
    # float32 reference of a 4B model needs that device memory.
    buffers = {name: tensor.clone() for name, tensor in engine.buffers.items()}
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    sources: WeightsSource = import_module(entry.sources_module)
    weights = sources.official_weights(device=device, seed=seed, **options)
    view: ReferenceView = import_module(entry.reference_view_module)
    matmul_tf32, convolution_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.inference_mode():
            outputs = view.reference_outputs(weights, inputs, buffers, shape=identity.shape,
                                             seed=seed, precision=reference_precision)
        torch.cuda.synchronize()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = convolution_tf32
    pairs = view.comparable(outputs, buffers)
    stages = {name: error_metrics(expected.float(), observed.float())
              for name, (expected, observed) in pairs.items()}
    # A non-finite value makes its metrics NaN, which `min` and `max` would skip.
    finite = all(bool(torch.isfinite(expected).all() and torch.isfinite(observed).all())
                 for expected, observed in pairs.values())
    tolerance = tolerances(identity.precision)["shallow"]
    min_cosine = min(metrics["cosine_similarity"] for metrics in stages.values())
    max_rel_rms = max(metrics["rel_rms"] for metrics in stages.values())
    gate = steps == 1 and layers == 1 and reference_precision == "float32"
    within = finite and bool(min_cosine > tolerance["cosine_min"]
                             and max_rel_rms < tolerance["rel_rms_max"])
    return {
        "identity": identity.as_dict(),
        "measurement_context": context,
        "config": {"plan": plan, "steps": steps, "layers": layers, "seed": seed,
                   "reference_precision": reference_precision, "options": options,
                   "oracle": "model_reference"},
        "stages": stages,
        "finite": finite,
        "min_cosine": min_cosine,
        "max_rel_rms": max_rel_rms,
        "tolerance": dict(tolerance),
        "within_tolerance": within,
        "mode": "gate" if gate else "report",
        "passed": within or not gate,
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
