"""Official OpenPI Pi0.5 execution and reference provenance."""
from __future__ import annotations
from pathlib import Path
import torch
from dataclasses import asdict
import inspect
import subprocess
from flash_vla.models.pi0.openpi import IMAGE_KEYS

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
    from openpi.models.model import Observation

    device = images.device
    observation = Observation(
        images={key: images[index].permute(2, 0, 1).unsqueeze(0)
                for index, key in enumerate(IMAGE_KEYS)},
        image_masks={key: torch.ones(1, dtype=torch.bool, device=device) for key in IMAGE_KEYS},
        state=state.unsqueeze(0),
        tokenized_prompt=tokens.unsqueeze(0),
        tokenized_prompt_mask=mask.unsqueeze(0),
    )
    images_list, image_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(  # noqa: SLF001
        observation, train=False)
    embeddings, pad_masks, att_masks = model.embed_prefix(
        images_list, image_masks, lang_tokens, lang_masks)
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

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
