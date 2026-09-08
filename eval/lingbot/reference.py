"""Run the frozen LingBot-VLA Robotwin oracle on one deterministic fixture.

This module is an adapter around the upstream implementation.  It deliberately
keeps preprocessing and model execution in upstream code; flash-vla stage
outputs are compared against the tensors written here.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
from safetensors.torch import save_file
import torch


UPSTREAM_COMMIT = "4eb34b7693a0565c67433f8fac9c59a2e67eb60b"
CHECKPOINT_REVISION = "fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e"
QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
FIXTURE_SEED = 42
PROMPT = "adjust bottle"
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _raw_fixture(seed: int) -> dict[str, np.ndarray | str]:
    rng = np.random.default_rng(seed)
    fixture: dict[str, np.ndarray | str] = {
        "observation.state": rng.uniform(-0.25, 0.25, size=14).astype(np.float32),
        "task": PROMPT,
    }
    for camera in CAMERAS:
        fixture[f"observation.images.{camera}"] = rng.integers(
            0, 256, size=(224, 224, 3), dtype=np.uint8
        )
    return fixture


def _preprocess(server, seed: int) -> tuple[
    dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    raw = _raw_fixture(seed)
    raw_tensors = {
        key: torch.from_numpy(value.copy())
        for key, value in raw.items()
        if isinstance(value, np.ndarray)
    }
    observation = {key: value.copy() if isinstance(value, np.ndarray) else value
                   for key, value in raw.items()}
    server.resize_image(observation)
    for key, value in tuple(observation.items()):
        if isinstance(value, np.ndarray):
            observation[key] = torch.from_numpy(value)
    observation["action"] = torch.zeros(server.config.chunk_size, 14)
    observation["action_is_pad"] = torch.zeros(server.config.chunk_size)
    prepared = server.vla.feature_transform.apply(observation)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (1, server.config.n_action_steps, server.config.max_action_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    inputs = {
        "pixel_values": prepared["images"].to(device="cuda", dtype=torch.bfloat16),
        "image_masks": prepared["img_masks"].to(device="cuda"),
        "language_tokens": prepared["lang_tokens"].unsqueeze(0).to(device="cuda"),
        "language_masks": prepared["lang_masks"].unsqueeze(0).to(device="cuda"),
        "state": prepared["state"].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16),
        "noise": noise.to(device="cuda"),
    }
    fixture = {
        **raw_tensors,
        **{key: value.detach().cpu().contiguous() for key, value in inputs.items()},
        "joint_mask": prepared["joint_mask"].detach().cpu().contiguous(),
    }
    return inputs, fixture, prepared


@torch.no_grad()
def _stage_outputs(core, inputs: dict[str, torch.Tensor], layers: int,
                   steps: int) -> dict[str, torch.Tensor]:
    from lingbotvla.models.vla.pi0.utils import make_att_2d_masks

    image_embeddings = core.qwenvl_with_expert.embed_image(inputs["pixel_values"])
    image_prefix = image_embeddings.reshape(1, -1, image_embeddings.shape[-1])
    image_masks = inputs["image_masks"].unsqueeze(0).repeat_interleave(
        image_embeddings.shape[1], dim=1
    )
    language_embeddings = core.qwenvl_with_expert.embed_language_tokens(
        inputs["language_tokens"]
    )
    prefix_embeddings = torch.cat((image_prefix, language_embeddings), dim=1)
    prefix_masks = torch.cat((image_masks, inputs["language_masks"]), dim=1)
    attention = make_att_2d_masks(prefix_masks, torch.zeros_like(prefix_masks))
    positions = torch.cumsum(prefix_masks, dim=1) - 1
    _, cache = core.qwenvl_with_expert.forward(
        attention_mask=attention,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix_embeddings, None],
        use_cache=True,
        fill_kv_cache=True,
    )
    prefix_k = torch.stack([cache[layer]["key_states"][0] for layer in range(layers)])
    prefix_v = torch.stack([cache[layer]["value_states"][0] for layer in range(layers)])
    timestep = torch.ones(1, dtype=torch.bfloat16, device="cuda")
    velocity = core.predict_velocity(
        inputs["state"], prefix_masks, cache, inputs["noise"], timestep
    )
    actions = core.sample_actions(
        inputs["pixel_values"], inputs["image_masks"], inputs["language_tokens"],
        inputs["language_masks"], inputs["state"], noise=inputs["noise"].clone(),
        num_steps=steps,
    )
    return {
        "vision_embeddings": image_embeddings,
        "prefix_k": prefix_k,
        "prefix_v": prefix_v,
        "velocity_step_0": velocity,
        "actions": actions,
    }


@torch.no_grad()
def _latency(core, inputs: dict[str, torch.Tensor], warmup: int, reps: int,
             steps: int) -> list[float]:
    def call():
        return core.sample_actions(
            inputs["pixel_values"], inputs["image_masks"], inputs["language_tokens"],
            inputs["language_masks"], inputs["state"], noise=inputs["noise"].clone(),
            num_steps=steps,
        )

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("the LingBot oracle requires CUDA")
    upstream = args.upstream.resolve()
    if _git_head(upstream) != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream is not frozen commit {UPSTREAM_COMMIT}")
    if _git_head(args.qwen.resolve()) != QWEN_REVISION:
        raise RuntimeError(f"Qwen dependency is not frozen revision {QWEN_REVISION}")
    checkpoint = args.checkpoint.resolve() / "model.safetensors"
    if checkpoint.stat().st_size != 16_789_932_052:
        raise RuntimeError(f"checkpoint has unexpected size: {checkpoint.stat().st_size}")
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    os.environ["QWEN25_PATH"] = str(args.qwen.resolve())
    from deploy.lingbot_vla_policy import LingbotVLAServer

    started = time.time()
    server = LingbotVLAServer(
        str(args.checkpoint.resolve()), use_length=50, use_bf16=True,
        num_denoising_step=10, use_compile=args.mode == "compile",
    )
    server.reset("robotwin")
    inputs, fixture, prepared = _preprocess(server, args.seed)
    core = server.vla.model
    core.qwenvl_with_expert.qwenvl.config.num_hidden_layers = args.layers
    outputs = _stage_outputs(core, inputs, args.layers, args.steps)
    inverse = dict(prepared)
    inverse["actions"] = outputs["actions"][0].float().cpu()
    inverse["state"] = prepared["state"].float().cpu()
    physical = server.vla.feature_transform.unapply(inverse)
    outputs["physical_actions"] = physical["action"].contiguous()
    samples = _latency(core, inputs, args.warmup, args.reps, args.steps)

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(fixture, args.output / "fixture.safetensors")
    save_file({key: value.detach().cpu().contiguous() for key, value in outputs.items()},
              args.output / f"official-{args.mode}.safetensors")
    processor = server.processor.image_processor
    metadata = {
        "version": 1,
        "mode": args.mode,
        "identity": {
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "qwen_revision": QWEN_REVISION,
            "precision": "bf16",
            "fixture_seed": args.seed,
            "adapter_revision": _git_head(Path(__file__).resolve().parents[2]),
        },
        "fixture": {
            "prompt": PROMPT,
            "raw_images": [3, 224, 224, 3],
            "pixel_values": list(inputs["pixel_values"].shape),
            "image_grid_thw": [[1, 16, 16]] * 3,
            "visual_tokens_per_view": 64,
            "language_slots": int(inputs["language_tokens"].shape[1]),
            "valid_language_tokens": int(inputs["language_masks"].sum().item()),
            "prefix_length": 264,
            "layers": args.layers,
            "denoise_steps": args.steps,
            "state": list(inputs["state"].shape),
            "noise": list(inputs["noise"].shape),
        },
        "processor": {
            "class": type(processor).__name__,
            "module": type(processor).__module__,
            "do_resize": processor.do_resize,
            "patch_size": processor.patch_size,
            "temporal_patch_size": processor.temporal_patch_size,
            "merge_size": processor.merge_size,
        },
        "latency_ms": {
            "samples": samples,
            "min": min(samples),
            "median": statistics.median(samples),
            "reps": args.reps,
            "warmup": args.warmup,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": __import__("transformers").__version__,
            "lerobot": importlib.metadata.version("lerobot"),
            "datasets": importlib.metadata.version("datasets"),
            "numpy": np.__version__,
            "upstream": str(upstream),
            "checkpoint": str(args.checkpoint.resolve()),
            "qwen": str(args.qwen.resolve()),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
        },
        "elapsed_s": time.time() - started,
    }
    (args.output / f"official-{args.mode}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "compile"), default="eager")
    parser.add_argument("--seed", type=int, default=FIXTURE_SEED)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
