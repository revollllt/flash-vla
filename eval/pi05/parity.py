"""Official Pi0.5 parity across two interpreters.

`eval.pi05.reference` runs OpenPI and the Target in one process, which needs the
upstream stack importable beside `flash_vla`. Where it is not -- the RTX 5090's
environment is the pinned flash-vla one -- the same comparison splits in two, as
`eval.lingbot` splits its oracle:

    OPENPI_PYTHON -m eval.pi05.parity capture --checkpoint <dir> --out <dir>
    python -m eval.pi05.parity compare --oracle <dir> --checkpoint <dir> \
        --target rtx5090/pi05 --checkpoint-id <id>

`capture` writes the fixture it used beside the tensors it produced;
`OPENPI_PI05_MODULE` names where the official forward came from and the oracle
records it. Compared: the prefix KV cache over valid rows -- padded rows attend
normally here and are zeroed upstream, so only their finiteness is checked --
and the action chunk, at the registry's `deepest` pair since it is read at full
depth. The oracle keeps OpenPI's half-split K layout and `compare` permutes it
(`reference.to_pair_layout`). Fixture inputs are bf16-representable, so a
rounding difference cannot masquerade as a model difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from safetensors.torch import load_file, save_file
import torch

from flash_vla.inference import build
from eval.metrics import error_metrics
from eval.pi05 import official as official_pi05
from eval.pi05.reference import to_pair_layout
from eval.tolerances import tolerances

DEFAULT_PROMPT = "pick up the plate and put it in the sink"
VISION_TOKENS = 256
NUM_VIEWS = 3
ACTION_DIM = 32
STATE_DIM = 32


def fixture(seed: int, chunk: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Seeded images, state and noise, exactly representable in bfloat16.

    The same expressions `eval.pi05.reference` uses, rounded through bf16 so the
    two implementations cannot disagree about the inputs themselves.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    images = torch.rand((NUM_VIEWS, 224, 224, 3), generator=generator, device=device,
                        dtype=torch.float32) * 2.0 - 1.0
    state = torch.randn((STATE_DIM,), generator=generator, device=device, dtype=torch.float32)
    noise = torch.randn((chunk, ACTION_DIM), generator=generator, device=device,
                        dtype=torch.float32)
    return {name: value.bfloat16().float()
            for name, value in (("images", images), ("state", state), ("noise", noise))}


def _tokenize(prompt: str, state: torch.Tensor, tokenizer_path: str | None):
    from flash_vla.models.pi05.tokenize import Pi05Tokenizer

    tokenizer = Pi05Tokenizer(tokenizer_path)
    tokenizer.set_task(prompt)
    tokens, mask = tokenizer.encode(state.detach().to("cpu", torch.float32).numpy())
    return (torch.from_numpy(tokens.astype("int64")), torch.from_numpy(mask))


def _official_config(chunk: int):
    """The upstream config for this checkpoint, from whichever module supplied the model.

    A vendored snapshot carries its own JAX-free `Pi0Config`; upstream's lives in
    a JAX-importing module. Both take the two fields that decide the architecture
    here and default the rest, and `validate_config` then applies this adapter's
    compatibility rule to whichever one was built.
    """
    from flash_vla.models.pi05.openpi import validate_config

    config = official_pi05.official_attr("Pi0Config")(pi05=True, action_horizon=chunk)
    validate_config(config)
    return config


RANDOM_CHECKPOINT_REVISION = "flash-vla/openpi-pi05-random-checkpoint/v1"


def capture(checkpoint: str | None, out: str, *, prompt: str = DEFAULT_PROMPT, seed: int = 0,
            steps: int = 10, chunk: int = 50, device: str = "cuda",
            tokenizer_path: str | None = None, exact_rope: bool = True,
            checkpoint_id: str | None = None) -> dict[str, object]:
    """Run the official forward on one fixture and write the oracle directory.

    With no `checkpoint`, the official model is randomly initialized at `seed`
    and its state dict is written into the oracle beside the tensors, so
    `compare` runs the Target on the very same weights. Both sides then consume
    the same values and any difference is an implementation difference -- which
    is what gates a new route. It establishes no trained-policy correctness or
    quality; a real checkpoint additionally exercises converted trained values.
    """
    from safetensors.torch import load_model, save_model

    from flash_vla.models.pi05.openpi import restore_rope_precision

    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    module = official_pi05.model_module()
    config = _official_config(chunk)
    inputs = fixture(seed, chunk, torch_device)
    tokens, mask = _tokenize(prompt, inputs["state"], tokenizer_path)
    n_valid = NUM_VIEWS * VISION_TOKENS + int(mask.sum())

    torch.manual_seed(seed)
    model = official_pi05.official_attr("PI0Pytorch")(config).eval()
    if checkpoint is None:
        checkpoint_id = checkpoint_id or f"{RANDOM_CHECKPOINT_REVISION}/seed-{seed}"
        path = directory / "model.safetensors"
        save_model(model, str(path))
    else:
        path = Path(checkpoint)
        if path.is_dir():
            path = path / "model.safetensors"
        load_model(model, str(path), strict=True, device="cpu")
    model = model.to(torch_device)
    rope_buffers = restore_rope_precision(model) if exact_rope else 0

    past_key_values, _, pad_masks = official_pi05.prefix_kv_cache(
        model, inputs["images"], inputs["state"],
        tokens.to(torch_device), mask.to(torch_device))
    layers = official_pi05.cache_layers(past_key_values)
    # [batch, kv_heads, seq, head_dim] with one batch and one kv head, per layer.
    expected = {
        "prefix_k": torch.stack([k[0, 0] for k, _ in layers]).float().cpu(),
        "prefix_v": torch.stack([v[0, 0] for _, v in layers]).float().cpu(),
    }
    actions = official_pi05.denoise(model, inputs["state"], pad_masks, past_key_values,
                                    inputs["noise"].unsqueeze(0), num_steps=steps)
    expected["actions"] = actions[0].float().cpu()

    save_file({name: value.cpu() for name, value in inputs.items()}
              | {"prompt_tokens": tokens, "prompt_mask": mask},
              str(directory / "fixture.safetensors"))
    save_file(expected, str(directory / "official-eager.safetensors"))
    metadata = {
        "producer": "eval.pi05.parity capture",
        "official": official_pi05.module_provenance(module),
        "config": {name: getattr(config, name) for name in
                   ("pi05", "discrete_state_input", "action_dim", "action_horizon",
                    "max_token_len", "paligemma_variant", "action_expert_variant", "dtype")},
        "checkpoint": {"id": checkpoint_id, "file": path.name,
                       "weights": "random" if checkpoint is None else "converted",
                       "config_json": json.loads((path.parent / "config.json").read_text())
                       if (path.parent / "config.json").is_file() else None},
        "fixture": {"seed": seed, "prompt": prompt, "num_views": NUM_VIEWS,
                    "prompt_tokens": int(mask.sum()), "n_valid_prefix": n_valid,
                    "chunk": chunk, "steps": steps, "bf16_representable_inputs": True},
        "exact_rope": exact_rope,
        "rope_buffers_restored": rope_buffers,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else "cpu",
        "layers_captured": len(layers),
        "prefix_layout": "openpi half-split rope channels; [layers, seq, head_dim]",
        "finite": all(bool(torch.isfinite(value).all()) for value in expected.values()),
    }
    (directory / "official-eager.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return metadata


def compare(oracle: str, checkpoint: str, *, target: str = "rtx5090/pi05",
            checkpoint_id: str, checkpoint_digest: str | None = None,
            plan: str = "shipped", layers: int | None = None,
            device: str = "cuda", tokenizer_path: str | None = None) -> dict[str, object]:
    """Replay the oracle's fixture through the Target and report the difference."""
    directory = Path(oracle)
    metadata = json.loads((directory / "official-eager.json").read_text())
    expected = load_file(str(directory / "official-eager.safetensors"))
    inputs = load_file(str(directory / "fixture.safetensors"))
    chunk, steps = metadata["fixture"]["chunk"], metadata["fixture"]["steps"]
    n_valid = metadata["fixture"]["n_valid_prefix"]
    depth = layers if layers is not None else metadata["layers_captured"]

    engine = build(target, plan, converted_checkpoint=checkpoint,
                   checkpoint_id=checkpoint_id,
                   checkpoint_digest=checkpoint_digest or checkpoint_id,
                   chunk_size=chunk, steps=steps, layers=depth, device=device,
                   prompt=metadata["fixture"]["prompt"], tokenizer_path=tokenizer_path,
                   seed=metadata["fixture"]["seed"])
    actions = engine.forward(
        images=inputs["images"].to(device).bfloat16(),
        state=inputs["state"].to(device),
        noise=inputs["noise"].to(device).bfloat16()).float().cpu()
    torch.cuda.synchronize()

    report: dict[str, object] = {
        "stage": "official-parity",
        "identity": engine.identity.as_dict(),
        "measurement_context": engine.measurement_context,
        "plan": engine.plan,
        "oracle": metadata,
        "layers_compared": depth,
    }
    staged = engine.buffers["prompt_ids"].detach().cpu().to(torch.int64)
    if not torch.equal(staged, inputs["prompt_tokens"].to(torch.int64)):
        report["passed"] = False
        report["error"] = "the Target tokenized a different prompt than the oracle captured"
        print(json.dumps(report, indent=2))
        return report

    # Compared on the host in fp32: the oracle was written by another process on
    # another device, and the metrics are the same either way.
    keys = engine.buffers["prefix_k"].detach().float().cpu()
    values = engine.buffers["prefix_v"].detach().float().cpu()
    prefix_len = engine.derived["prefix_len"]
    per_layer = [
        {"layer": index,
         "k": error_metrics(to_pair_layout(expected["prefix_k"][index, :n_valid]),
                            keys[index, :n_valid]),
         "v": error_metrics(expected["prefix_v"][index, :n_valid], values[index, :n_valid])}
        for index in range(min(depth, expected["prefix_k"].shape[0]))
    ]
    padded = torch.cat([keys[:, n_valid:prefix_len], values[:, n_valid:prefix_len]])
    report["padded_rows_finite"] = bool(torch.isfinite(padded).all().item())
    report["per_layer"] = per_layer
    # `--layers` truncates both stages of the engine, and the oracle's chunk came
    # out of the full depth. Layers 0..N-1 of the prefix are still the same
    # tensors -- each depends only on the layers before it -- but the chunk is a
    # different function, so it is reported as not compared rather than judged.
    report["actions_finite"] = bool(torch.isfinite(actions).all().item())
    full_depth = depth == metadata["layers_captured"]
    report["actions"] = error_metrics(expected["actions"], actions) if full_depth else None
    if not full_depth:
        report["actions_note"] = (f"not compared: the engine ran {depth} layers and the "
                                  f"oracle {metadata['layers_captured']}")

    tol = tolerances(engine.identity.precision)
    worst = lambda row: (row["k"] if row["k"]["cosine_similarity"] <= row["v"]["cosine_similarity"]  # noqa: E731
                         else row["v"])
    within = lambda m, pair: bool(m["cosine_similarity"] > pair["cosine_min"]  # noqa: E731
                                  and m["rel_rms"] < pair["rel_rms_max"])
    cosines = [min(row["k"]["cosine_similarity"], row["v"]["cosine_similarity"])
               for row in per_layer]
    cosine_steps = [cosines[i] - cosines[i + 1] for i in range(len(cosines) - 1)]
    report["cosine"] = {"layer0": cosines[0], "deepest": cosines[-1],
                        "worst_step": max(cosine_steps) if cosine_steps else 0.0}
    report["tolerance"] = {"layer0": dict(tol["layer0"]), "deepest": dict(tol["deepest"]),
                           "step": tol["max_cosine_step"],
                           "actions": dict(tol["deepest"]) if full_depth else None}
    report["passed"] = bool(
        report["padded_rows_finite"] and report["actions_finite"]
        and within(worst(per_layer[0]), tol["layer0"])
        and within(worst(per_layer[-1]), tol["deepest"])
        and (not cosine_steps or max(cosine_steps) < tol["max_cosine_step"])
        and (not full_depth or within(report["actions"], tol["deepest"])))
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("capture", help="run the official forward; write the oracle")
    run.add_argument("--checkpoint", default=None,
                     help="a checkpoint OpenPI's converter wrote (directory or "
                          "model.safetensors); omitted, the model is randomly initialized "
                          "at --seed and its weights are written into the oracle")
    run.add_argument("--out", required=True, help="oracle directory to write")
    run.add_argument("--checkpoint-id", help="immutable id of the upstream checkpoint")
    run.add_argument("--prompt", default=DEFAULT_PROMPT)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--steps", type=int, default=10)
    run.add_argument("--chunk", type=int, default=50)
    run.add_argument("--device", default="cuda")
    run.add_argument("--tokenizer", dest="tokenizer_path", default=None)
    run.add_argument("--no-exact-rope", dest="exact_rope", action="store_false")

    check = sub.add_parser("compare", help="replay the oracle's fixture through the Target")
    check.add_argument("--oracle", required=True)
    check.add_argument("--checkpoint", required=True)
    check.add_argument("--checkpoint-id", required=True)
    check.add_argument("--checkpoint-digest", default=None)
    check.add_argument("--target", default="rtx5090/pi05")
    check.add_argument("--plan", default="shipped")
    check.add_argument("--layers", type=int, default=None)
    check.add_argument("--device", default="cuda")
    check.add_argument("--tokenizer", dest="tokenizer_path", default=None)

    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "capture":
        return 0 if capture(args.pop("checkpoint"), args.pop("out"), **args)["finite"] else 1
    return 0 if compare(args.pop("oracle"), args.pop("checkpoint"), **args)["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
