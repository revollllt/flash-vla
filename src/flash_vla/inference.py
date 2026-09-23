"""Runner factories by Target name, for the generic harnesses.

The one place a harness learns how to construct a Target: a named checkpoint or
random weights at a stated seed, the required tokenizer, and a task. Everything after
construction goes through the engine protocol (`flash_vla.runtime.engine`),
so the latency, profile, kernel, floor and correctness runners contain no
model names; this registry is where they enter.

Synthetic weights are seeded. Real Pi0.5 weights use an explicit upstream config
and immutable checkpoint ID/digest; the seed selects their input fixture. Every
construction rebuilds checkpoint-dependent folded tensors for its step schedule.

A plan is `"shipped"` (the Target's one deployed plan, the default),
`"reference"` (its correctness oracle route), a JSON object, or a path to a
JSON file (the candidate plans of the optimization workspace, `lab/plans/`).
"""
from __future__ import annotations

from typing import Any, Callable
from pathlib import Path
import json
import os
from typing import Mapping

from flash_vla.provenance import canonical_digest, git_revision
from flash_vla.runtime import ModelRunner

DEFAULT_PROMPT = "pick up the plate and put it in the sink"


def _pi05(plan: Any = "shipped", *, seed: int = 0, num_views: int = 3, chunk_size: int | None = None,
          steps: int = 10, layers: int = 18, prompt_len: int | None = None,
          device: str = "cuda", prompt: str = DEFAULT_PROMPT, tokenizer_path: str | None = None,
          declare: bool = False, checkpoint: str | None = None,
          converted_checkpoint: str | None = None,
          checkpoint_id: str | None = None, checkpoint_digest: str | None = None,
          openpi_config: str | None = None, quantization: str | None = None, target=None):
    """Build a Pi0.5 runner on synthetic or OpenPI weights.

    `target` defaults to H100; Pi0.5's checkpoint, tokenizer and fixture carry no
    hardware, so both Targets share this factory. Weights come from `seed`,
    from `checkpoint` (loaded through OpenPI, which validates the named upstream
    config), or from `converted_checkpoint` (already converted, read without
    OpenPI -- it resolves no config, so the caller supplies the shape profile
    and the immutable ID). `quantization` names one of the Target's recipes.
    """
    if target is None:
        from flash_vla.hardware.nvidia.h100.pi05 import TARGET as target
    from flash_vla.models.pi05 import openpi as openpi05
    from flash_vla.models.pi05.spec import MAX_TOKEN_LEN, random_checkpoint_revision
    from flash_vla.models.pi05.tokenize import Pi05Tokenizer
    from flash_vla.models.pi05.weights import fold, random_checkpoint

    engine_revision = git_revision()

    if checkpoint is not None and converted_checkpoint is not None:
        raise ValueError("pass checkpoint or converted_checkpoint, not both")
    # The chunk is the Target's own default unless an upstream config names one.
    default_chunk = target.configure().chunk_size
    if checkpoint is None and converted_checkpoint is None:
        if any(value is not None for value in (checkpoint_id, checkpoint_digest, openpi_config)):
            raise ValueError("checkpoint provenance/config requires a real checkpoint path")
        checkpoint_id = checkpoint_digest = random_checkpoint_revision(seed)
        chunk_size = default_chunk if chunk_size is None else chunk_size
    elif converted_checkpoint is not None:
        if not all((checkpoint_id, checkpoint_digest)):
            raise ValueError("a converted checkpoint requires checkpoint_id and checkpoint_digest")
        if openpi_config is not None:
            raise ValueError("a converted checkpoint resolves no OpenPI config; "
                             "pass `checkpoint` to have OpenPI resolve and validate one")
        chunk_size = default_chunk if chunk_size is None else chunk_size
    else:
        if not all((checkpoint_id, checkpoint_digest, openpi_config)):
            raise ValueError("real checkpoint requires checkpoint_id, checkpoint_digest and openpi_config")
        reference_config = openpi05.resolve_config(checkpoint, openpi_config)
        if chunk_size is not None and chunk_size != reference_config.action_horizon:
            raise ValueError("chunk_size differs from the checkpoint's explicit OpenPI config")
        if prompt_len is not None and prompt_len != reference_config.max_token_len:
            raise ValueError("prompt_len differs from the checkpoint's explicit OpenPI config")
        chunk_size = reference_config.action_horizon

    config = dict(num_views=num_views, chunk_size=chunk_size, steps=steps, layers=layers,
                  prompt_len=prompt_len or MAX_TOKEN_LEN, prompt=prompt)
    if declare:
        runner = ModelRunner(target, None, engine_revision=engine_revision,
                             checkpoint_id=checkpoint_id,
                             checkpoint_digest=checkpoint_digest, plan=plan,
                             quantization=quantization, device=device, capture=False, **config)
    else:
        if converted_checkpoint is not None:
            source = openpi05.converted_checkpoint(converted_checkpoint)
        elif checkpoint is None:
            source = random_checkpoint(seed=seed, device=device)
        else:
            model = openpi05.build_model(checkpoint, device, seed=seed, config=reference_config)
            source = openpi05.target_checkpoint(model)
            del model
        runner = ModelRunner(target, fold(source, steps=steps), engine_revision=engine_revision,
                             checkpoint_id=checkpoint_id,
                             checkpoint_digest=checkpoint_digest, plan=plan,
                             quantization=quantization, device=device,
                             tokenizer=Pi05Tokenizer(tokenizer_path), **config)
    fixture = {"producer": "flash-vla/pi05-inputs-v1", "seed": seed, "prompt": prompt}
    runner.measurement_context["fixture"] = {
        "id": fixture["producer"] + "/seed-" + str(seed), "digest": canonical_digest(fixture),
    }
    return runner


def _pi0(plan: Any = "shipped", *, seed: int = 0, num_views: int = 3, chunk_size: int = 50,
         steps: int = 10, layers: int = 18, prompt_len: int = 0, device: str = "cuda",
         declare: bool = False, target=None):
    """Build a Pi0 runner on seeded synthetic weights.

    `target` selects the hardware Target; it defaults to H100. Pi0's checkpoint
    and fixture are generated from `seed` and carry no hardware, so the two
    Targets share this factory rather than duplicating it.
    """
    if target is None:
        from flash_vla.hardware.nvidia.h100.pi0 import TARGET as target
    from flash_vla.models.pi0 import random_checkpoint
    from flash_vla.models.pi0.spec import random_checkpoint_revision

    config = dict(num_views=num_views, chunk_size=chunk_size, steps=steps, layers=layers)
    checkpoint_id = random_checkpoint_revision(seed)
    engine_revision = git_revision()
    if declare:
        runner = ModelRunner(target, None, engine_revision=engine_revision,
                             checkpoint_id=checkpoint_id,
                             checkpoint_digest=checkpoint_id, plan=plan,
                           device=device, capture=False,
                           prompt_len=prompt_len, **config)
    else:
        checkpoint = random_checkpoint(num_views=num_views, chunk_size=chunk_size,
                                       prompt_len=prompt_len, seed=seed, device=device)
        runner = ModelRunner(target, checkpoint, engine_revision=engine_revision,
                             checkpoint_id=checkpoint_id,
                             checkpoint_digest=checkpoint_id, plan=plan,
                           device=device, **config)
    fixture = {"producer": "flash-vla/pi0-inputs-v1", "seed": seed}
    runner.measurement_context["fixture"] = {
        "id": fixture["producer"] + "/seed-" + str(seed), "digest": canonical_digest(fixture),
    }
    return runner


def _lingbot(plan: Any = "shipped", *, seed: int = 42, steps: int = 10, layers: int = 36,
             device: str = "cuda", declare: bool = False,
             checkpoint: str | None = None, fixture: str | None = None,
             checkpoint_id: str | None = None, checkpoint_digest: str | None = None,
             fixture_id: str | None = None, fixture_digest: str | None = None,
             asset_config: str | None = None, source_checkout: str | None = None):
    from pathlib import Path
    from flash_vla.hardware.nvidia.h100.lingbot_vla import TARGET
    from flash_vla.models.lingbot import load_checkpoint

    if source_checkout is None:
        source = None
        engine_revision = git_revision()
    else:
        from flash_vla.source import lingbot_target
        TARGET, source = lingbot_target(source_checkout)
        engine_revision = source["revision"]

    if checkpoint_id is None:
        if checkpoint is not None or checkpoint_digest is not None:
            raise ValueError("checkpoint override needs checkpoint_id and checkpoint_digest")
        checkpoint_id = checkpoint_digest = TARGET.ASSETS["checkpoint"]
    if not checkpoint_digest:
        raise ValueError("checkpoint_digest must identify the immutable weights manifest")
    if fixture_id is None:
        if fixture is not None or fixture_digest is not None:
            raise ValueError("fixture override needs fixture_id and fixture_digest")
        fixture_id = fixture_digest = TARGET.ASSETS["fixture"]
    if not fixture_digest:
        raise ValueError("fixture_digest must identify the immutable fixture")
    assets = {}
    if not declare:
        identifiers = dict(TARGET.ASSETS, checkpoint=checkpoint_id, fixture=fixture_id)
        explicit = {role: Path(path).expanduser().resolve()
                    for role, path in (("checkpoint", checkpoint), ("fixture", fixture))
                    if path is not None}
        assets = resolve_assets({role: value for role, value in identifiers.items()
                                 if role not in explicit}, asset_config)
        assets.update(explicit)
    runner = ModelRunner(TARGET, None if declare else load_checkpoint(assets["checkpoint"]),
                         engine_revision=engine_revision, checkpoint_id=checkpoint_id, checkpoint_digest=checkpoint_digest,
                         plan=plan, device="cpu" if declare else device,
                         capture=not declare, assets=assets, steps=steps, layers=layers)
    runner.measurement_context["fixture"] = {"id": fixture_id, "digest": fixture_digest}
    if source is not None:
        runner.implementation_source = source
    return runner


def _pi0_rtx5090(plan: Any = "shipped", **overrides):
    """Pi0 on the RTX 5090. Same model, same synthetic weights, sm_120."""
    from flash_vla.hardware.nvidia.rtx5090.pi0 import TARGET
    return _pi0(plan, target=TARGET, **overrides)


def _pi05_rtx5090(plan: Any = "shipped", **overrides):
    """Pi0.5 on the RTX 5090. Same model, same checkpoint options, sm_120."""
    from flash_vla.hardware.nvidia.rtx5090.pi05 import TARGET
    return _pi05(plan, target=TARGET, **overrides)


def _groot_n17(plan: Any = "shipped", *, seed: int = 0, steps: int = 4, layers: int = 16,
               sequence_length: int = 156, device: str = "cuda", declare: bool = False,
               capture: bool = True, checkpoint: str | None = None, fixture: str | None = None,
               checkpoint_id: str | None = None, fixture_id: str | None = None,
               asset_config: str | None = None):
    from flash_vla.hardware.nvidia.rtx5090.groot_n17 import TARGET
    from flash_vla.models.groot_n17.weights import Checkpoint

    explicit = {"checkpoint": checkpoint, "fixture": fixture}
    ids = {"checkpoint": checkpoint_id, "fixture": fixture_id}
    engine_revision = git_revision()
    for role, path in explicit.items():
        if path is not None and ids[role] is None:
            raise ValueError(f"{role} override requires {role}_id")
    identifiers = {role: ids[role] or value for role, value in TARGET.ASSETS.items()}
    assets = {}
    if not declare:
        assets = resolve_assets({role: value for role, value in identifiers.items()
                                 if explicit[role] is None}, asset_config) if any(
                                     value is None for value in explicit.values()) else {}
        assets.update({role: Path(path).expanduser().resolve()
                       for role, path in explicit.items() if path is not None})
    runner = ModelRunner(TARGET, None if declare else Checkpoint(assets["checkpoint"]),
                         engine_revision=engine_revision, checkpoint_id=identifiers["checkpoint"],
                         checkpoint_digest=identifiers["checkpoint"],
                         plan=plan, device=device, capture=capture and not declare, assets=assets,
                         sequence_length=sequence_length, steps=steps, layers=layers)
    fixture_context = f"{identifiers['fixture']}/noise-seed-{seed}"
    runner.measurement_context["fixture"] = {"id": fixture_context, "digest": fixture_context}
    return runner


#: Target name -> factory. Short aliases resolve through `resolve`.
TARGETS: dict[str, Callable[..., Any]] = {
    "hardware/nvidia/rtx5090/groot_n17": _groot_n17,
    "hardware/nvidia/h100/lingbot_vla": _lingbot,
    "hardware/nvidia/h100/pi05": _pi05,
    "hardware/nvidia/h100/pi0": _pi0,
    "hardware/nvidia/rtx5090/pi0": _pi0_rtx5090,
    "hardware/nvidia/rtx5090/pi05": _pi05_rtx5090,
}
_ALIASES = {"rtx5090/groot_n17": "hardware/nvidia/rtx5090/groot_n17",
            "5090/groot_n17": "hardware/nvidia/rtx5090/groot_n17",
            "groot-n17": "hardware/nvidia/rtx5090/groot_n17",
            "rtx5090_groot_n17": "hardware/nvidia/rtx5090/groot_n17",
            "h100/pi05": "hardware/nvidia/h100/pi05", "pi05": "hardware/nvidia/h100/pi05",
            "h100/pi0": "hardware/nvidia/h100/pi0", "pi0": "hardware/nvidia/h100/pi0",
            "h100/lingbot_vla": "hardware/nvidia/h100/lingbot_vla",
            "lingbot_vla": "hardware/nvidia/h100/lingbot_vla",
            "rtx5090/pi0": "hardware/nvidia/rtx5090/pi0",
            "5090/pi0": "hardware/nvidia/rtx5090/pi0",
            "rtx5090/pi05": "hardware/nvidia/rtx5090/pi05",
            "5090/pi05": "hardware/nvidia/rtx5090/pi05",
            # The `lab/plans/<target>-<name>.json` prefix form, which cannot
            # carry a slash.
            "rtx5090_pi0": "hardware/nvidia/rtx5090/pi0",
            "rtx5090_pi05": "hardware/nvidia/rtx5090/pi05"}

#: The two plan names every Target understands.
PLAN_NAMES = ("shipped", "reference")


def resolve(name: str) -> str:
    """The full Target name for `name` or one of its aliases."""
    full = _ALIASES.get(name, name)
    if full not in TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {sorted(TARGETS)} "
                       f"and aliases {sorted(_ALIASES)}")
    return full


def build(name: str, plan: Any = "shipped", **overrides):
    """Construct the runner of Target `name` on `plan`."""
    return TARGETS[resolve(name)](plan, **overrides)


def declare(name: str, plan: Any = "shipped", **overrides):
    """Construct the runner of Target `name` up to its graph: no weights, no device."""
    return TARGETS[resolve(name)](plan, declare=True, device="cpu", **overrides)


__all__ = ["DEFAULT_PROMPT", "PLAN_NAMES", "TARGETS", "build", "declare", "resolve"]

def resolve_assets(identifiers: Mapping[str, str], config: str | Path | None = None) -> dict[str, Path]:
    """Map role -> logical ID to local paths; relative paths use the config directory."""
    path = Path(config or os.environ["FLASH_VLA_ASSETS"]).expanduser().resolve()
    locations = json.loads(path.read_text())
    return {role: (path.parent / Path(locations[identifier]).expanduser()).resolve()
            for role, identifier in identifiers.items()}


def parse_options(items: list[str]) -> dict[str, Any]:
    """`key=value` strings to a dict; true/false and integers are converted."""
    out: dict[str, Any] = {}
    for item in items:
        key, _, value = item.partition("=")
        low = value.lower()
        out[key] = (True if low == "true" else False if low == "false"
                    else int(value) if value.lstrip("-").isdigit() else value)
    return out
