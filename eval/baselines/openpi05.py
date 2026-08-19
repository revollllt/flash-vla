"""Thin adapter from the official OpenPI PyTorch Pi0.5 to this H100 target.

Same job as `openpi.py` does for Pi0, and it reuses that module's tensor helpers
rather than restating them -- the vision tower and the 2B prefix are identical
between the two models, down to the RoPE layout and the norm folding.

Three things differ, all in the action expert:

- no `state_proj` and no `action_time_mlp`; the action path is a bare
  `action_in_proj` and the timestep goes through `time_mlp_in/out`.
- the expert's norms are adaptive, so there is no per-channel `weight` to fold
  into the following GEMM (`modeling_gemma.py:57-64`). The decoder qkv, gate and
  up projections are therefore taken raw, unlike the encoder's.
- the vocabulary table is carried whole, because Pi0.5's prompt contains the
  state and cannot be embedded ahead of time.

The output is the `models.pi05.spec.weight_shapes()` layout -- OpenPI-aligned,
before the AdaRMSNorm fold. Run it through `models.pi05.weights.fold` to get
what the engine loads.
"""
from __future__ import annotations

from pathlib import Path

import torch

from eval.baselines.openpi import (
    IMAGE_KEYS,
    _bf16,
    _fold_norm,
    _interleave_rope,
    _linear,
    _stack_layers,
    _value,
)
from flash_vla.models.pi05.spec import DECODER_HEADS, ENCODER_LAYERS, VISION_LAYERS, weight_shapes

VISION = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
PROJECTOR = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
ENCODER = "paligemma_with_expert.paligemma.model.language_model"
DECODER = "paligemma_with_expert.gemma_expert.model"


def restore_rope_precision(model) -> int:
    """Undo OpenPI's bfloat16 cast of the rotary `inv_freq` buffers.

    `PaliGemmaWithExpertModel.to_bfloat16_for_selected_params` casts the whole
    module with `self.to(dtype=torch.bfloat16)`, which sweeps up `inv_freq` --
    a registered buffer, not a parameter. That quantizes the rotary frequencies
    to eight mantissa bits: `10000**(-1/128)` becomes 0.9296875 instead of
    0.9305720, a relative error of 1e-3.

    Phase is frequency times position, so the error grows with position. At a
    prefix position of 900 it is several radians -- the rotation is simply a
    different one. Pi0 never had to care: with an empty prompt its prefix stops
    at 768 and the damage is smaller, and its gate reads the final action after
    ten denoising steps rather than the KV cache directly.

    JAX OpenPI, which is what the checkpoints were trained with, computes the
    timescale in float32 (`models/gemma.py:424-440`), so float32 here is closer
    to the model as trained, not further.

    Note the frequencies have to be *recomputed*, not re-cast. The forward pass
    already does `self.inv_freq[...].float()`, so widening the stored buffer
    changes nothing -- 0.9296875 widened to float32 is still 0.9296875. The
    quantization happened once, at construction, and only re-running the rope
    initializer undoes it. Returns the number of buffers fixed.
    """
    fixed = 0
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        rope_init_fn = getattr(module, "rope_init_fn", None)
        if not isinstance(inv_freq, torch.Tensor) or rope_init_fn is None:
            continue
        exact, attention_scaling = rope_init_fn(module.config, inv_freq.device)
        module.inv_freq = exact.float()
        if getattr(module, "original_inv_freq", None) is not None:
            module.original_inv_freq = module.inv_freq
        module.attention_scaling = attention_scaling
        fixed += 1
    return fixed


def build_model(checkpoint: str | Path | None = None,
                device: str | torch.device = "cuda", seed: int = 0,
                exact_rope: bool = True):
    """Load a Pi0.5 model, or construct one with random weights if no path is given.

    Random weights are enough for an implementation gate: both sides run the same
    tensors, so any difference is ours. They are not enough for a policy-quality
    claim, which needs `pi05_base`.

    `exact_rope` restores the rotary frequencies to float32; see
    `restore_rope_precision` for why that is the honest default.
    """
    try:
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    except ImportError as error:
        raise RuntimeError(
            "OpenPI's PyTorch dependencies are required; install OpenPI using "
            "its official PyTorch setup instructions."
        ) from error

    torch.manual_seed(seed)
    model = PI0Pytorch(Pi0Config(pi05=True, pytorch_compile_mode=None))

    if checkpoint is not None:
        from safetensors.torch import load_model as load_safetensors_model

        checkpoint = Path(checkpoint)
        if checkpoint.is_dir():
            checkpoint = checkpoint / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        load_safetensors_model(model, str(checkpoint), strict=True)

    model = model.to(device).eval()
    if exact_rope:
        restore_rope_precision(model)
    return model


def _decoder_qkv(state: dict[str, torch.Tensor], layer: int) -> torch.Tensor:
    """Packed decoder QKV in the target's RoPE layout, with no norm to fold in."""
    attention = f"{DECODER}.layers.{layer}.self_attn"
    return torch.cat(
        [
            _interleave_rope(_linear(state, f"{attention}.q_proj"), DECODER_HEADS),
            _interleave_rope(_linear(state, f"{attention}.k_proj"), 1),
            _linear(state, f"{attention}.v_proj"),
        ],
        dim=1,
    )


def _encoder_qkv(state: dict[str, torch.Tensor], layer: int) -> torch.Tensor:
    """Packed encoder QKV, with the plain RMSNorm scale folded in as in Pi0."""
    attention = f"{ENCODER}.layers.{layer}.self_attn"
    norm = f"{ENCODER}.layers.{layer}.input_layernorm.weight"
    return torch.cat(
        [
            _interleave_rope(_fold_norm(state, f"{attention}.q_proj", norm), DECODER_HEADS),
            _interleave_rope(_fold_norm(state, f"{attention}.k_proj", norm), 1),
            _fold_norm(state, f"{attention}.v_proj", norm),
        ],
        dim=1,
    )


@torch.inference_mode()
def target_checkpoint(model) -> dict[str, torch.Tensor]:
    """Convert one official Pi0.5 state dict to the packed `weight_shapes()` layout."""
    state = model.state_dict()

    def vision_layer(index: int) -> str:
        return f"{VISION}.encoder.layers.{index}"

    def vision_qkv(index: int) -> torch.Tensor:
        return torch.cat([_linear(state, f"{vision_layer(index)}.self_attn.{p}_proj")
                          for p in ("q", "k", "v")], dim=1)

    def vision_qkv_bias(index: int) -> torch.Tensor:
        return torch.cat([_value(state, f"{vision_layer(index)}.self_attn.{p}_proj.bias")
                          for p in ("q", "k", "v")])

    def encoder_ffn(name: str):
        return lambda i: _fold_norm(
            state, f"{ENCODER}.layers.{i}.mlp.{name}",
            f"{ENCODER}.layers.{i}.post_attention_layernorm.weight")

    def decoder_linear(fmt: str):
        return lambda i: _linear(state, fmt.format(i=i))

    checkpoint = {
        # --- vision: identical to Pi0 ---
        "vision_patch_embedding_w": _bf16(
            _value(state, f"{VISION}.embeddings.patch_embedding.weight").permute(2, 3, 1, 0)),
        "vision_patch_embedding_b": _bf16(
            _value(state, f"{VISION}.embeddings.patch_embedding.bias")),
        "vision_position_embedding": _bf16(
            _value(state, f"{VISION}.embeddings.position_embedding.weight")),
        "vision_attn_qkv_w": _stack_layers(VISION_LAYERS, vision_qkv),
        "vision_attn_qkv_b": _stack_layers(VISION_LAYERS, vision_qkv_bias),
        "vision_attn_o_w": _stack_layers(
            VISION_LAYERS, lambda i: _linear(state, f"{vision_layer(i)}.self_attn.out_proj")),
        "vision_attn_o_b": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.self_attn.out_proj.bias")),
        "vision_ffn_up_w": _stack_layers(
            VISION_LAYERS, lambda i: _linear(state, f"{vision_layer(i)}.mlp.fc1")),
        "vision_ffn_up_b": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.mlp.fc1.bias")),
        "vision_ffn_down_w": _stack_layers(
            VISION_LAYERS, lambda i: _linear(state, f"{vision_layer(i)}.mlp.fc2")),
        "vision_ffn_down_b": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.mlp.fc2.bias")),
        "vision_pre_attn_norm_w": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.layer_norm1.weight")),
        "vision_pre_attn_norm_b": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.layer_norm1.bias")),
        "vision_pre_ffn_norm_w": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.layer_norm2.weight")),
        "vision_pre_ffn_norm_b": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.layer_norm2.bias")),
        "vision_final_norm_w": _bf16(_value(state, f"{VISION}.post_layernorm.weight")),
        "vision_final_norm_b": _bf16(_value(state, f"{VISION}.post_layernorm.bias")),

        # --- encoder: identical to Pi0, plain norms folded into the GEMMs ---
        "encoder_multi_modal_projector_w": _bf16(_linear(state, PROJECTOR)),
        "encoder_multi_modal_projector_b": _bf16(_value(state, f"{PROJECTOR}.bias")),
        "encoder_attn_qkv_w": _stack_layers(ENCODER_LAYERS, lambda i: _encoder_qkv(state, i)),
        "encoder_attn_o_w": _stack_layers(
            ENCODER_LAYERS, lambda i: _linear(state, f"{ENCODER}.layers.{i}.self_attn.o_proj")),
        "encoder_ffn_gate_w": _stack_layers(ENCODER_LAYERS, encoder_ffn("gate_proj")),
        "encoder_ffn_up_w": _stack_layers(ENCODER_LAYERS, encoder_ffn("up_proj")),
        "encoder_ffn_down_w": _stack_layers(
            ENCODER_LAYERS, lambda i: _linear(state, f"{ENCODER}.layers.{i}.mlp.down_proj")),

        # --- decoder: adaptive norms, so nothing folds into these ---
        "decoder_action_in_proj_w": _bf16(_linear(state, "action_in_proj")),
        "decoder_action_in_proj_b": _bf16(_value(state, "action_in_proj.bias")),
        "decoder_time_mlp_in_w": _bf16(_linear(state, "time_mlp_in")),
        "decoder_time_mlp_in_b": _bf16(_value(state, "time_mlp_in.bias")),
        "decoder_time_mlp_out_w": _bf16(_linear(state, "time_mlp_out")),
        "decoder_time_mlp_out_b": _bf16(_value(state, "time_mlp_out.bias")),
        "decoder_ada_rms_attn_w": _stack_layers(
            ENCODER_LAYERS, decoder_linear(DECODER + ".layers.{i}.input_layernorm.dense")),
        "decoder_ada_rms_attn_b": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _value(state, f"{DECODER}.layers.{i}.input_layernorm.dense.bias")),
        "decoder_ada_rms_ffn_w": _stack_layers(
            ENCODER_LAYERS, decoder_linear(DECODER + ".layers.{i}.post_attention_layernorm.dense")),
        "decoder_ada_rms_ffn_b": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _value(state, f"{DECODER}.layers.{i}.post_attention_layernorm.dense.bias")),
        "decoder_ada_rms_final_w": _bf16(_linear(state, f"{DECODER}.norm.dense")),
        "decoder_ada_rms_final_b": _bf16(_value(state, f"{DECODER}.norm.dense.bias")),
        "decoder_attn_qkv_w": _stack_layers(ENCODER_LAYERS, lambda i: _decoder_qkv(state, i)),
        "decoder_attn_o_w": _stack_layers(
            ENCODER_LAYERS, lambda i: _linear(state, f"{DECODER}.layers.{i}.self_attn.o_proj")),
        "decoder_ffn_gate_w": _stack_layers(
            ENCODER_LAYERS, lambda i: _linear(state, f"{DECODER}.layers.{i}.mlp.gate_proj")),
        "decoder_ffn_up_w": _stack_layers(
            ENCODER_LAYERS, lambda i: _linear(state, f"{DECODER}.layers.{i}.mlp.up_proj")),
        "decoder_ffn_down_w": _stack_layers(
            ENCODER_LAYERS, lambda i: _linear(state, f"{DECODER}.layers.{i}.mlp.down_proj")),
        "decoder_action_out_proj_w": _bf16(_linear(state, "action_out_proj")),
        "decoder_action_out_proj_b": _bf16(_value(state, "action_out_proj.bias")),

        # --- language: the whole table, gathered per inference ---
        "vocab_embeddings": _bf16(_value(state, f"{ENCODER}.embed_tokens.weight")),
    }

    expected = weight_shapes()
    if set(checkpoint) != set(expected):
        raise ValueError(
            f"target checkpoint keys do not match: "
            f"missing={sorted(set(expected) - set(checkpoint))}, "
            f"extra={sorted(set(checkpoint) - set(expected))}")
    for name, shape in expected.items():
        if tuple(checkpoint[name].shape) != shape:
            raise ValueError(
                f"{name} has shape {tuple(checkpoint[name].shape)}, expected {shape}")
    return checkpoint


@torch.inference_mode()
def prefix_kv_cache(model, images: torch.Tensor, state: torch.Tensor,
                    tokens: torch.Tensor, mask: torch.Tensor):
    """Run OpenPI's prefix prefill and return its per-layer K and V.

    Mirrors `PI0Pytorch.sample_actions` up to the point where the KV cache
    exists (`pi0_pytorch.py:185-201`), which is exactly what this target's
    prefix pass produces.
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
    return past_key_values, embeddings
