"""Inspect a Pi0.5 checkpoint ABI for onboarding; no GPU execution or value hashing.

python -m tools.check_checkpoint --checkpoint PATH --checkpoint-id ID \
    --checkpoint-digest DIGEST --openpi-config CONFIG --out REPORT
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from flash_vla.inference import declare
from flash_vla.models.pi05 import openpi as openpi05
from flash_vla.runtime.identity import inference_signature


def run(checkpoint, *, checkpoint_id, checkpoint_digest, openpi_config, seed=0, **options):
    runner = declare("pi05", checkpoint=checkpoint, checkpoint_id=checkpoint_id,
                     checkpoint_digest=checkpoint_digest, openpi_config=openpi_config,
                     seed=seed, **options)
    config = openpi05.resolve_config(checkpoint, openpi_config)
    observed = openpi05.checkpoint_contract(checkpoint, config)
    if inference_signature(**observed["contract"]) != runner.identity.inference_signature:
        raise ValueError("inference signature mismatch; resolve a compatible Target/model revision")
    return dict(status="passed", identity=runner.identity.as_dict(),
                weights=runner.measurement_context["weights"],
                fixture=runner.measurement_context["fixture"],
                openpi_config=openpi_config, reference_model_config=asdict(config),
                scope="stored tensor ABI under the explicit reference configuration; not training origin or numerical correctness",
                **observed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--openpi-config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    from flash_vla.inference import parse_options
    report = run(args.checkpoint, checkpoint_id=args.checkpoint_id,
                 checkpoint_digest=args.checkpoint_digest, openpi_config=args.openpi_config,
                 seed=args.seed, **parse_options(args.option))
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
