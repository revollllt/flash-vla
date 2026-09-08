"""Compare the captured LingBot Target with the frozen upstream eager oracle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from safetensors.torch import load_file
import torch
import yaml

from benchmarks.targets import build
from eval.acceptance import tolerances
from eval.metrics import error_metrics

DEFAULT_ORACLE = Path(
    "/data/user/jzou521/codes/cuda/flash-vla/artifacts/onboarding/"
    "lingbot-vla-4b-h100-bf16/official"
)


def _physical_actions(actions: torch.Tensor, fixture: dict[str, torch.Tensor]) -> torch.Tensor:
    upstream = Path(os.environ.get(
        "LINGBOT_UPSTREAM",
        "/data/user/jzou521/codes/cuda/flash-vla/artifacts/upstreams/lingbot-vla",
    )).resolve()
    qwen = Path(os.environ.get(
        "LINGBOT_QWEN",
        "/data/user/jzou521/codes/cuda/flash-vla/artifacts/upstreams/qwen2.5-vl-3b-instruct",
    )).resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.models import build_processor

    with Path(os.environ.get(
        "LINGBOT_CHECKPOINT",
        "/data/user/jzou521/models/lingbot-vla-4b-posttrain-robotwin-fb71a2c",
    )).joinpath("lingbotvla_cli.yaml").open() as source:
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


def run(plan: str = "reference", oracle: Path = DEFAULT_ORACLE,
        seed: int = 42) -> dict[str, object]:
    expected = load_file(oracle / "official-eager.safetensors")
    fixture = load_file(oracle / "fixture.safetensors")
    engine = build("h100/lingbot_vla", plan, seed=seed)
    inputs = engine.sample_inputs(seed)
    engine.stage(**inputs)
    for step in engine.program:
        if step.kind != "segment":
            raise RuntimeError(f"unexpected LingBot host step {step}")
        engine.replay(step.name)
    torch.cuda.synchronize()

    actual = {
        "vision_embeddings": engine.buffers["vision_embeddings"].detach().cpu(),
        "prefix_k": engine.buffers["prefix_k"].detach().cpu(),
        "prefix_v": engine.buffers["prefix_v"].detach().cpu(),
        "velocity_step_0": engine.buffers["velocity_step_0"].detach().cpu(),
        "actions": engine.buffers["actions"].detach().cpu(),
    }
    actual["physical_actions"] = _physical_actions(actual["actions"], fixture)
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
        "oracle": str(oracle / "official-eager.safetensors"),
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
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = run(args.plan, args.oracle, args.seed)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
