"""Official OpenPI Pi0.5 execution and reference provenance."""
from __future__ import annotations
from pathlib import Path
import importlib
import os
import torch
from dataclasses import asdict
import inspect
import subprocess
from types import SimpleNamespace
from flash_vla.models.pi0.openpi import IMAGE_KEYS

#: Which module supplies the official PyTorch implementation. The default is
#: upstream OpenPI's own path; `OPENPI_PI05_MODULE` names another that exports
#: the same `PI0Pytorch` and `make_att_2d_masks`, which is how a machine without
#: the JAX training stack still runs the official forward -- a vendored snapshot
#: of the same PyTorch subtree. What ran is recorded by `module_provenance`.
OPENPI_PI05_MODULE = os.environ.get("OPENPI_PI05_MODULE",
                                    "openpi.models_pytorch.pi0_pytorch")


def model_module():
    """The module the official forward comes from."""
    return importlib.import_module(OPENPI_PI05_MODULE)


def official_attr(name: str):
    """`name` from the configured module, or from its `pi0_pytorch` submodule.

    Upstream's module *is* `pi0_pytorch`: it carries `PI0Pytorch` and
    `make_att_2d_masks` but not `Pi0Config`. A vendored snapshot is a package
    whose `__init__` re-exports the classes but not the module-level helper.
    Looking in both lets one environment variable name either shape.
    """
    module = model_module()
    found = getattr(module, name, None)
    if found is None:
        submodule = getattr(module, "pi0_pytorch", None) or importlib.import_module(
            f"{module.__name__}.pi0_pytorch")
        found = getattr(submodule, name, None)
    if found is None:
        raise AttributeError(f"{OPENPI_PI05_MODULE} provides no {name!r}")
    return found


def module_provenance(module) -> dict:
    """Record which official implementation ran, without hashing weights.

    `reference_provenance` above assumes the upstream OpenPI checkout and reads
    its Git state; a vendored snapshot lives in a different repository, so this
    records the module identity and whatever work tree it is in rather than
    asserting a repository it may not belong to.
    """
    source = Path(inspect.getfile(module)).resolve()
    provenance = {"module": module.__name__, "source": source.name,
                  "commit": None, "dirty": None, "repository": None}
    try:
        directory = str(source.parent)
        provenance["commit"] = subprocess.check_output(
            ["git", "-C", directory, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
        provenance["dirty"] = bool(subprocess.check_output(
            ["git", "-C", directory, "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL))
        provenance["repository"] = subprocess.check_output(
            ["git", "-C", directory, "remote", "get-url", "origin"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, OSError):
        pass
    vendor = source.parent / "VENDOR.md"
    if vendor.is_file():
        provenance["vendor_note"] = vendor.name
    return provenance


def observation(images: torch.Tensor, state: torch.Tensor,
                tokens: torch.Tensor, mask: torch.Tensor) -> SimpleNamespace:
    """The observation `_preprocess_observation` reads, as plain attributes.

    OpenPI's `Observation` is a dataclass in a JAX-importing module, and
    `preprocess_observation_pytorch` only reads attributes off it and returns a
    plain object of its own. Building it here keeps the official forward
    reachable from an environment that has the PyTorch subtree and not the rest.
    """
    device = images.device
    return SimpleNamespace(
        images={key: images[index].permute(2, 0, 1).unsqueeze(0)
                for index, key in enumerate(IMAGE_KEYS)},
        image_masks={key: torch.ones(1, dtype=torch.bool, device=device) for key in IMAGE_KEYS},
        state=state.unsqueeze(0),
        tokenized_prompt=tokens.unsqueeze(0),
        tokenized_prompt_mask=mask.unsqueeze(0),
        token_ar_mask=None,
        token_loss_mask=None,
    )

def cache_layers(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Normalize the several shapes a transformers cache can take."""
    if hasattr(past_key_values, "key_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache, strict=True))
    if hasattr(past_key_values, "layers"):
        return [(layer.keys, layer.values) for layer in past_key_values.layers]
    return [(k, v) for k, v in past_key_values]


def reference_provenance(model, config, *, exact_rope=False):
    """Record the loaded upstream class and adapter Git state, without hashing weights."""
    provenance = {}
    for role, source in (("upstream", Path(inspect.getfile(type(model)))),
                         ("adapter", Path(__file__))):
        directory = source.resolve().parent
        commit = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
            text=True)
        provenance[role] = dict(commit=commit, dirty=bool(dirty))
    provenance["upstream"]["module"] = type(model).__module__
    provenance.update(repository="https://github.com/Physical-Intelligence/openpi.git",
                      commit=provenance["upstream"]["commit"])
    provenance["config"] = asdict(config)
    provenance["exact_rope"] = exact_rope
    return provenance


@torch.inference_mode()
def prefix_kv_cache(model, images: torch.Tensor, state: torch.Tensor,
                    tokens: torch.Tensor, mask: torch.Tensor):
    """Run OpenPI's prefix prefill and return its per-layer K and V.

    Mirrors `PI0Pytorch.sample_actions` up to the point where the KV cache
    exists (`pi0_pytorch.py:185-201`), which is exactly what this target's
    prefix pass produces. `pad_masks` comes back too because `denoise_step`
    needs it to build the suffix mask and positions.
    """
    images_list, image_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(  # noqa: SLF001
        observation(images, state, tokens, mask), train=False)
    embeddings, pad_masks, att_masks = model.embed_prefix(
        images_list, image_masks, lang_tokens, lang_masks)
    make_att_2d_masks = official_attr("make_att_2d_masks")

    attention_mask = model._prepare_attention_masks_4d(  # noqa: SLF001
        make_att_2d_masks(pad_masks, att_masks))
    positions = torch.cumsum(pad_masks, dim=1) - 1
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=attention_mask, position_ids=positions, past_key_values=None,
        inputs_embeds=[embeddings, None], use_cache=True)
    return past_key_values, embeddings, pad_masks


@torch.inference_mode()
def denoise(model, state: torch.Tensor, prefix_pad_masks: torch.Tensor, past_key_values,
            noise: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
    """Run OpenPI's flow-matching loop against an existing KV cache.

    The body of `PI0Pytorch.sample_actions` (`pi0_pytorch.py:203-221`) with the
    prefill lifted out, so a caller can hand both implementations the *same*
    cache and compare only the decoder.
    """
    device = noise.device
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    for _ in range(num_steps):
        v_t = model.denoise_step(state.unsqueeze(0), prefix_pad_masks, past_key_values,
                                 x_t, time.expand(1))
        x_t = x_t + dt * v_t
        time = time + dt
    return x_t


def truncate_expert(model, layers: int) -> int:
    """Shorten the action expert to `layers`, for depth bisection.

    Both sides of a suffix comparison have to run the same depth. Our engine
    takes `layers` as a constructor argument; OpenPI's expert has to be cut
    here, and both its ModuleList and its config count matter -- `GemmaModel`
    iterates `self.layers[: self.config.num_hidden_layers]`.

    The prefix is untouched: it lives in a different module (`paligemma`), and
    a suffix comparison wants the full prefix regardless.
    """
    expert = model.paligemma_with_expert.gemma_expert.model
    if layers < len(expert.layers):
        expert.layers = expert.layers[:layers]
    expert.config.num_hidden_layers = min(layers, expert.config.num_hidden_layers)
    return len(expert.layers)
