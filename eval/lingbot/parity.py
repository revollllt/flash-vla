"""Compare the captured LingBot Target with the frozen upstream eager oracle."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from safetensors.torch import load_file
import torch
import yaml

from flash_vla.inference import resolve_assets
from flash_vla.inference import build
from eval.tolerances import tolerances
from eval.metrics import error_metrics

ORACLE_ASSET = "lingbot-robotwin-canonical-v1/seed-42/oracle"


def _physical_actions(actions: torch.Tensor, fixture: dict[str, torch.Tensor], assets) -> torch.Tensor:
    upstream, qwen = (Path(assets[role]) for role in ("upstream", "qwen"))
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.models import build_processor

    with (Path(assets["checkpoint"]) / "lingbotvla_cli.yaml").open() as source:
        training = yaml.safe_load(source)
    data = SimpleNamespace(**training["data"])
    data.max_state_dim = 75
    data.max_action_dim = 75
    data.resize_imgs_with_padding = [224, 224]
    data.tokenizer_max_length = 72
    processor = build_processor(str(qwen))
    transform = FeatureTransform(
        upstream / "configs/robot_configs/robotwin.yaml",
        data,
        processor.tokenizer,
        processor.image_processor,
        chunk_size=50,
        norm_stats_path=upstream / "assets/norm_stats/robotwin_50.json",
    )
    inverse = {
        "actions": actions[0].float().cpu(),
        "state": fixture["state"][0].float(),
        "joint_mask": fixture["joint_mask"],
    }
    return transform.unapply(inverse)["action"].contiguous()


def run(plan: str = "reference", oracle: Path | None = None,
        seed: int = 42, layers: int = 36, steps: int = 10, *,
        asset_config: str | None = None, source_checkout: str | None = None) -> dict[str, object]:
    if oracle is None:
        oracle = resolve_assets({"oracle": ORACLE_ASSET}, asset_config)["oracle"]
    oracle_metadata = json.loads((oracle / "official-eager.json").read_text())
    expected = load_file(oracle / "official-eager.safetensors")
    fixture = load_file(oracle / "fixture.safetensors")
    engine = build("h100/lingbot_vla", plan, seed=seed, layers=layers, steps=steps, asset_config=asset_config, source_checkout=source_checkout)
    inputs = engine.sample_inputs(seed)
    engine.stage(**inputs)
    for step in engine.program:
        if step.kind != "segment":
            raise RuntimeError(f"unexpected LingBot host step {step}")
        engine.replay(step.name)
    torch.cuda.synchronize()

    actual = {
        "vision_embeddings": engine.buffers["vision_embeddings"].detach().cpu(),
        "prefix_k": engine.buffers["prefix_k"][:layers].detach().cpu(),
        "prefix_v": engine.buffers["prefix_v"][:layers].detach().cpu(),
        "velocity_step_0": engine.buffers["velocity_step_0"].detach().cpu(),
        "actions": engine.buffers["actions"].detach().cpu(),
    }
    actual["physical_actions"] = _physical_actions(actual["actions"], fixture, engine.assets)
    metrics = {name: error_metrics(expected[name], value) for name, value in actual.items()}
    equal = {name: torch.equal(expected[name], value) for name, value in actual.items()}
    first = engine.forward(**inputs).clone()
    second = engine.forward(**inputs).clone()
    torch.cuda.synchronize()
    threshold = tolerances("bf16")["deepest"]
    passed = all(
        row["cosine_similarity"] > threshold["cosine_min"]
        and row["rel_rms"] < threshold["rel_rms_max"]
        for row in metrics.values()
    ) and torch.equal(first, second)
    return {
        "identity": engine.identity.as_dict(),
        "implementation_source": getattr(engine, "implementation_source", None),
        "measurement_context": engine.measurement_context,
        "oracle": str(oracle / "official-eager.safetensors"),
        "reference_provenance": dict(oracle_metadata["identity"],
            repository="https://github.com/Robbyant/lingbot-vla.git",
            commit=oracle_metadata["identity"]["upstream_commit"]),
        "seed": seed,
        "metrics": metrics,
        "bitwise_equal": equal,
        "replay_identical": torch.equal(first, second),
        "threshold": threshold,
        "passed": passed,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="reference")
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--asset-config")
    parser.add_argument("--option", action="append", default=[],
                        help="construction options forwarded by lab.optimize.gate; asset_config and source_checkout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    from flash_vla.inference import parse_options
    options = parse_options(args.option)
    unknown = set(options) - {"asset_config", "source_checkout"}
    if unknown:
        parser.error(f"unsupported official LingBot construction options: {sorted(unknown)}")
    asset_config = options.get("asset_config", args.asset_config)
    with redirect_stdout(sys.stderr):
        report = run(args.plan, args.oracle, args.seed, args.layers, args.steps,
                     asset_config=asset_config, source_checkout=options.get("source_checkout"))
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
