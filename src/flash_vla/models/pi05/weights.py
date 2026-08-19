"""Pi0.5 checkpoint construction and the AdaRMSNorm fold.

`fold` is the bridge between the two layouts in `spec`: it takes a checkpoint
shaped like OpenPI's and returns the one an execution target loads, with the
flow-matching schedule resolved.

Why it is legitimate. The AdaRMSNorm condition is

    cond(t) = swish(time_mlp_out(swish(time_mlp_in(posemb_sincos(t)))))

a function of the timestep and nothing else, and OpenPI's sampler walks a fixed
schedule `t = 1, 1-1/steps, ..., 1/steps` (`models/pi0.py:228,278`). So every
`(scale, shift, gate)` in the model is a compile-time constant of the target,
and the 116 M parameters of modulation Dense never reach the inference weight
stream. Pi0 does the same thing to its action/time MLP; Pi0.5 simply offers more
of it.

What survives the fold, per norm site:

    x_hat @ W  =  rstd(x) * ((x * (1 + scale)) @ W)  +  shift @ W

a per-K-column scale on the GEMM's A operand and a per-N constant bias, plus a
per-N gate vector in the residual. The final norm folds further: its scale, its
shift and the Euler `dt` all collapse into the output projection.

The fold runs in float32 and rounds once at the end, so it is slightly *more*
accurate than OpenPI, which computes the modulation Dense in bfloat16. That is
the same trade the Pi0 fused kernels make; `eval.correctness.pi05.fold_equivalence`
measures it rather than assuming it.
"""
from __future__ import annotations

import math

import torch

from .spec import (
    DECODER_DIM,
    DEFAULT_FLOW_STEPS,
    ENCODER_LAYERS,
    TIME_MAX_PERIOD,
    TIME_MIN_PERIOD,
    runtime_shapes,
    weight_shapes,
)


def flow_timesteps(steps: int = DEFAULT_FLOW_STEPS,
                   device: str | torch.device = "cpu") -> tuple[torch.Tensor, float]:
    """The timesteps OpenPI's sampler visits, and its Euler `dt`.

    Accumulated in float32 exactly as the sampler does (`models/pi0.py:271,278`,
    and its PyTorch mirror), so the last step matches bit for bit rather than
    approximately.
    """
    dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device=device)
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    times = []
    for _ in range(steps):
        times.append(time.clone())
        time = time + dt
    return torch.stack(times), float(dt)


def posemb_sincos(timesteps: torch.Tensor, dim: int = DECODER_DIM,
                  min_period: float = TIME_MIN_PERIOD,
                  max_period: float = TIME_MAX_PERIOD) -> torch.Tensor:
    """Sine-cosine timestep embedding, matching OpenPI's float64 construction.

    Mirrors `create_sinusoidal_pos_embedding` (`models_pytorch/pi0_pytorch.py:25-42`),
    including that the periods are built in float64 and the result is cast back
    to the timestep dtype.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim ({dim}) must be divisible by 2")
    device = timesteps.device
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling[None, :] * timesteps.double()[:, None]
    embedding = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return embedding.to(timesteps.dtype)


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def adarms_conditions(checkpoint: dict[str, torch.Tensor],
                      steps: int = DEFAULT_FLOW_STEPS) -> torch.Tensor:
    """The `(steps, DECODER_DIM)` AdaRMSNorm condition for the fixed schedule."""
    def value(name: str) -> torch.Tensor:
        return checkpoint[name].detach().float()

    device = value("decoder_time_mlp_in_w").device
    timesteps, _ = flow_timesteps(steps, device)
    embedding = posemb_sincos(timesteps)
    hidden = _swish(embedding @ value("decoder_time_mlp_in_w")
                    + value("decoder_time_mlp_in_b"))
    return _swish(hidden @ value("decoder_time_mlp_out_w")
                  + value("decoder_time_mlp_out_b"))


def _modulation(cond: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`Dense(cond)` split into (scale, shift, gate), in OpenPI's order."""
    scale, shift, gate = torch.chunk(cond @ weight + bias, 3, dim=-1)
    return scale, shift, gate


@torch.no_grad()
def fold(checkpoint: dict[str, torch.Tensor], steps: int = DEFAULT_FLOW_STEPS,
         dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Resolve the AdaRMSNorm condition against the fixed schedule.

    Takes a checkpoint matching `spec.weight_shapes()` and returns one matching
    `spec.runtime_shapes(steps)`. Only the computed tables are materialized in
    `dtype`; every other tensor is passed through by reference.
    """
    expected = weight_shapes()
    missing = sorted(set(expected) - set(checkpoint))
    if missing:
        raise KeyError(f"checkpoint is missing {missing}")

    def value(name: str) -> torch.Tensor:
        return checkpoint[name].detach().float()

    cond = adarms_conditions(checkpoint, steps)
    device = cond.device

    # Tensors the fold does not touch pass through by reference, in whatever
    # dtype and layout they arrived in. Copying them would cost several GB on a
    # real checkpoint for no reason -- only the computed tables are new.
    folded = {name: checkpoint[name] for name in expected
              if not name.startswith(("decoder_ada_rms_", "decoder_time_mlp_"))
              and name not in ("decoder_action_out_proj_w", "decoder_action_out_proj_b")}

    attn_scale = torch.empty((steps, ENCODER_LAYERS, DECODER_DIM), device=device)
    attn_gate = torch.empty_like(attn_scale)
    ffn_scale = torch.empty_like(attn_scale)
    ffn_gate = torch.empty_like(attn_scale)
    qkv_bias = torch.empty((steps, ENCODER_LAYERS, expected["decoder_attn_qkv_w"][2]),
                           device=device)
    gate_bias = torch.empty((steps, ENCODER_LAYERS, expected["decoder_ffn_gate_w"][2]),
                            device=device)
    up_bias = torch.empty_like(gate_bias)

    # Upcast one layer at a time: the projections are the largest tensors here
    # and materializing all of them in float32 at once is gratuitous.
    def layer_value(name: str, layer: int) -> torch.Tensor:
        return checkpoint[name][layer].detach().float()

    for layer in range(ENCODER_LAYERS):
        scale, shift, gate = _modulation(cond, layer_value("decoder_ada_rms_attn_w", layer),
                                         layer_value("decoder_ada_rms_attn_b", layer))
        attn_scale[:, layer] = 1.0 + scale
        attn_gate[:, layer] = gate
        qkv_bias[:, layer] = shift @ layer_value("decoder_attn_qkv_w", layer)

        scale, shift, gate = _modulation(cond, layer_value("decoder_ada_rms_ffn_w", layer),
                                         layer_value("decoder_ada_rms_ffn_b", layer))
        ffn_scale[:, layer] = 1.0 + scale
        ffn_gate[:, layer] = gate
        gate_bias[:, layer] = shift @ layer_value("decoder_ffn_gate_w", layer)
        up_bias[:, layer] = shift @ layer_value("decoder_ffn_up_w", layer)

    # Final norm: its gate is discarded upstream, and its scale, its shift and
    # the Euler dt all fold into the output projection, so the target keeps Pi0's
    # `out += bias + rms(x) @ weight` call shape.
    _, dt = flow_timesteps(steps, device)
    scale, shift, _ = _modulation(cond, value("decoder_ada_rms_final_w"),
                                  value("decoder_ada_rms_final_b"))
    out_w, out_b = value("decoder_action_out_proj_w"), value("decoder_action_out_proj_b")
    action_w = dt * (1.0 + scale)[:, :, None] * out_w[None]
    action_b = dt * (shift @ out_w + out_b)

    computed = {
        "decoder_ada_attn_scale": attn_scale,
        "decoder_ada_attn_gate": attn_gate,
        "decoder_ada_ffn_scale": ffn_scale,
        "decoder_ada_ffn_gate": ffn_gate,
        "decoder_qkv_shift_bias": qkv_bias,
        "decoder_ffn_gate_shift_bias": gate_bias,
        "decoder_ffn_up_shift_bias": up_bias,
        "decoder_action_out_proj_w": action_w,
        "decoder_action_out_proj_b": action_b,
    }
    folded.update({name: tensor.to(dtype).contiguous()
                   for name, tensor in computed.items()})

    target = runtime_shapes(steps)
    if set(folded) != set(target):
        raise ValueError(
            f"folded keys do not match: missing={sorted(set(target) - set(folded))}, "
            f"extra={sorted(set(folded) - set(target))}")
    for name, shape in target.items():
        if tuple(folded[name].shape) != shape:
            raise ValueError(
                f"{name} has shape {tuple(folded[name].shape)}, expected {shape}")
    return folded


def random_checkpoint(wscale: float = 0.05, seed: int = 0,
                      device: str = "cuda") -> dict[str, torch.Tensor]:
    """Create a synthetic Pi0.5 checkpoint in the `weight_shapes()` layout.

    The small default scale keeps deep random residual streams numerically
    useful, as in Pi0. Note this is the *unfolded* layout; run it through `fold`
    to get what a target loads.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        name: (torch.randn(shape, dtype=torch.float32, device=device,
                           generator=generator) * wscale).to(torch.bfloat16)
        for name, shape in weight_shapes().items()
    }


__all__ = ["adarms_conditions", "flow_timesteps", "fold", "posemb_sincos",
           "random_checkpoint"]
