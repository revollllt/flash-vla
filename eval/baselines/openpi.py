"""Thin adapter from the official OpenPI PyTorch Pi0 to this H100 target."""

from __future__ import annotations

from pathlib import Path

import torch

from flash_vla.models.pi0.spec import (
    DECODER_HEADS,
    ENCODER_LAYERS,
    VISION_LAYERS,
    weight_shapes,
)


VISION = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
PROJECTOR = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
ENCODER = "paligemma_with_expert.paligemma.model.language_model"
DECODER = "paligemma_with_expert.gemma_expert.model"
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def load_model(checkpoint: str | Path, device: str | torch.device = "cuda"):
    """Load the official OpenPI PyTorch Pi0 model without torch.compile."""
    try:
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from safetensors.torch import load_model as load_safetensors_model
    except ImportError as error:
        raise RuntimeError(
            "OpenPI's PyTorch dependencies are required; install OpenPI using "
            "its official PyTorch setup instructions."
        ) from error

    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = PI0Pytorch(Pi0Config(pytorch_compile_mode=None))
    load_safetensors_model(model, str(checkpoint), strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def sample_actions(model, images: torch.Tensor, state: torch.Tensor, noise: torch.Tensor):
    """Run OpenPI on three valid images, an empty prompt, and explicit noise."""
    from openpi.models.model import Observation

    if images.shape != (3, 224, 224, 3):
        raise ValueError(f"expected images [3, 224, 224, 3], got {tuple(images.shape)}")
    if state.shape != (32,):
        raise ValueError(f"expected state [32], got {tuple(state.shape)}")
    if noise.shape != (50, 32):
        raise ValueError(f"expected noise [50, 32], got {tuple(noise.shape)}")

    device = images.device
    observation = Observation(
        images={
            key: images[index].permute(2, 0, 1).unsqueeze(0)
            for index, key in enumerate(IMAGE_KEYS)
        },
        image_masks={key: torch.ones(1, dtype=torch.bool, device=device) for key in IMAGE_KEYS},
        state=state.unsqueeze(0),
        tokenized_prompt=torch.empty((1, 0), dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.empty((1, 0), dtype=torch.bool, device=device),
    )
    return model.sample_actions(
        device, observation, noise=noise.unsqueeze(0), num_steps=10
    )[0]


def _value(state: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    try:
        return state[name].detach().float()
    except KeyError as error:
        raise KeyError(f"OpenPI checkpoint is missing {name!r}") from error


def _linear(state: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return _value(state, f"{name}.weight").T


def _fold_norm(
    state: dict[str, torch.Tensor], linear: str, norm: str
) -> torch.Tensor:
    scale = 1.0 + _value(state, norm)
    return _linear(state, linear) * scale[:, None]


def _bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.bfloat16).contiguous()


def _stack_layers(count: int, make_layer) -> torch.Tensor:
    return torch.stack([_bf16(make_layer(index)) for index in range(count)])


def _interleave_rope(weight: torch.Tensor, heads: int) -> torch.Tensor:
    """Match the target's adjacent-pair RoPE layout to OpenPI's half-split layout."""
    inputs, outputs = weight.shape
    head_dim = outputs // heads
    return (
        weight.reshape(inputs, heads, 2, head_dim // 2)
        .permute(0, 1, 3, 2)
        .reshape(inputs, outputs)
    )


def _qkv(
    state: dict[str, torch.Tensor], root: str, layer: int, norm: str
) -> torch.Tensor:
    attention = f"{root}.layers.{layer}.self_attn"
    return torch.cat(
        [
            _interleave_rope(
                _fold_norm(state, f"{attention}.q_proj", norm), DECODER_HEADS
            ),
            _interleave_rope(_fold_norm(state, f"{attention}.k_proj", norm), 1),
            _fold_norm(state, f"{attention}.v_proj", norm),
        ],
        dim=1,
    )


@torch.inference_mode()
def target_checkpoint(model) -> dict[str, torch.Tensor]:
    """Convert one official Pi0 state dict to the packed H100/Pi0 layout."""
    state = model.state_dict()

    def vision_layer(index: int) -> str:
        return f"{VISION}.encoder.layers.{index}"

    def encoder_norm(index: int, name: str) -> str:
        return f"{ENCODER}.layers.{index}.{name}.weight"

    def decoder_norm(index: int, name: str) -> str:
        return f"{DECODER}.layers.{index}.{name}.weight"

    def vision_qkv(index: int) -> torch.Tensor:
        return torch.cat(
            [
                _linear(state, f"{vision_layer(index)}.self_attn.{projection}_proj")
                for projection in ("q", "k", "v")
            ],
            dim=1,
        )

    def vision_qkv_bias(index: int) -> torch.Tensor:
        return torch.cat(
            [
                _value(state, f"{vision_layer(index)}.self_attn.{projection}_proj.bias")
                for projection in ("q", "k", "v")
            ]
        )

    checkpoint = {
        "vision_patch_embedding_w": _bf16(
            _value(state, f"{VISION}.embeddings.patch_embedding.weight").permute(2, 3, 1, 0)
        ),
        "vision_patch_embedding_b": _bf16(
            _value(state, f"{VISION}.embeddings.patch_embedding.bias")
        ),
        "vision_position_embedding": _bf16(
            _value(state, f"{VISION}.embeddings.position_embedding.weight")
        ),
        "vision_attn_qkv_w": _stack_layers(VISION_LAYERS, vision_qkv),
        "vision_attn_qkv_b": _stack_layers(VISION_LAYERS, vision_qkv_bias),
        "vision_attn_o_w": _stack_layers(
            VISION_LAYERS, lambda i: _linear(state, f"{vision_layer(i)}.self_attn.out_proj")
        ),
        "vision_attn_o_b": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.self_attn.out_proj.bias")
        ),
        "vision_ffn_up_w": _stack_layers(
            VISION_LAYERS, lambda i: _linear(state, f"{vision_layer(i)}.mlp.fc1")
        ),
        "vision_ffn_up_b": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.mlp.fc1.bias")
        ),
        "vision_ffn_down_w": _stack_layers(
            VISION_LAYERS, lambda i: _linear(state, f"{vision_layer(i)}.mlp.fc2")
        ),
        "vision_ffn_down_b": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.mlp.fc2.bias")
        ),
        "vision_pre_attn_norm_w": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.layer_norm1.weight")
        ),
        "vision_pre_attn_norm_b": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.layer_norm1.bias")
        ),
        "vision_pre_ffn_norm_w": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.layer_norm2.weight")
        ),
        "vision_pre_ffn_norm_b": _stack_layers(
            VISION_LAYERS, lambda i: _value(state, f"{vision_layer(i)}.layer_norm2.bias")
        ),
        "vision_final_norm_w": _bf16(_value(state, f"{VISION}.post_layernorm.weight")),
        "vision_final_norm_b": _bf16(_value(state, f"{VISION}.post_layernorm.bias")),
        "encoder_multi_modal_projector_w": _bf16(_linear(state, PROJECTOR)),
        "encoder_multi_modal_projector_b": _bf16(_value(state, f"{PROJECTOR}.bias")),
        "encoder_attn_qkv_w": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _qkv(state, ENCODER, i, encoder_norm(i, "input_layernorm")),
        ),
        "encoder_attn_o_w": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _linear(state, f"{ENCODER}.layers.{i}.self_attn.o_proj"),
        ),
        "encoder_ffn_gate_w": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _fold_norm(
                state,
                f"{ENCODER}.layers.{i}.mlp.gate_proj",
                encoder_norm(i, "post_attention_layernorm"),
            ),
        ),
        "encoder_ffn_up_w": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _fold_norm(
                state,
                f"{ENCODER}.layers.{i}.mlp.up_proj",
                encoder_norm(i, "post_attention_layernorm"),
            ),
        ),
        "encoder_ffn_down_w": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _linear(state, f"{ENCODER}.layers.{i}.mlp.down_proj"),
        ),
        "decoder_state_in_proj_w": _bf16(_linear(state, "state_proj")),
        "decoder_state_in_proj_b": _bf16(_value(state, "state_proj.bias")),
        "decoder_action_mlp_w": _bf16(_linear(state, "action_time_mlp_out")),
        "decoder_action_mlp_b": _bf16(_value(state, "action_time_mlp_out.bias")),
        "decoder_attn_qkv_w": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _qkv(state, DECODER, i, decoder_norm(i, "input_layernorm")),
        ),
        "decoder_attn_o_w": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _linear(state, f"{DECODER}.layers.{i}.self_attn.o_proj"),
        ),
        "decoder_ffn_gate_w": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _fold_norm(
                state,
                f"{DECODER}.layers.{i}.mlp.gate_proj",
                decoder_norm(i, "post_attention_layernorm"),
            ),
        ),
        "decoder_ffn_up_w": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _fold_norm(
                state,
                f"{DECODER}.layers.{i}.mlp.up_proj",
                decoder_norm(i, "post_attention_layernorm"),
            ),
        ),
        "decoder_ffn_down_w": _stack_layers(
            ENCODER_LAYERS,
            lambda i: _linear(state, f"{DECODER}.layers.{i}.mlp.down_proj"),
        ),
        "language_embeds": torch.empty(
            (0, 2048), dtype=torch.bfloat16, device=_value(state, "action_in_proj.weight").device
        ),
    }

    # The target fixes all ten timesteps, so action/time projection can be folded ahead of capture.
    action_projection = _linear(state, "action_in_proj")
    action_bias = _value(state, "action_in_proj.bias")
    mix_weight = _value(state, "action_time_mlp_in.weight")
    mix_bias = _value(state, "action_time_mlp_in.bias")
    action_mix = mix_weight[:, :1024].T
    time_mix = mix_weight[:, 1024:].T

    checkpoint["decoder_action_fused_in_proj_w"] = _bf16(action_projection @ action_mix)

    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding

    dt = torch.tensor(-0.1, dtype=torch.float32, device=action_projection.device)
    time = torch.tensor(1.0, dtype=torch.float32, device=action_projection.device)
    times = []
    for _ in range(10):
        times.append(time.clone())
        time += dt
    time_embeddings = create_sinusoidal_pos_embedding(
        torch.stack(times), 1024, min_period=4e-3, max_period=4.0, device=action_projection.device
    ).float()
    checkpoint["decoder_action_fused_time_biases"] = _bf16(
        action_bias @ action_mix + time_embeddings @ time_mix + mix_bias
    )

    # The target updates noise in place; fold both the final RMSNorm scale and Euler dt.
    final_norm = 1.0 + _value(state, f"{DECODER}.norm.weight")
    checkpoint["decoder_action_fused_out_proj_w"] = _bf16(
        dt * final_norm[:, None] * _linear(state, "action_out_proj")
    )
    checkpoint["decoder_action_fused_out_proj_b"] = _bf16(
        dt * _value(state, "action_out_proj.bias")
    )

    expected = weight_shapes(prompt_len=0)
    if set(checkpoint) != set(expected):
        missing = sorted(set(expected) - set(checkpoint))
        extra = sorted(set(checkpoint) - set(expected))
        raise ValueError(f"target checkpoint keys do not match: missing={missing}, extra={extra}")
    for name, shape in expected.items():
        if checkpoint[name].shape != shape:
            raise ValueError(f"{name} has shape {tuple(checkpoint[name].shape)}, expected {shape}")

    return checkpoint
