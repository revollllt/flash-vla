"""Equivalence gate for the Pi0.5 AdaRMSNorm fold.

`models.pi05.weights.fold` removes 116 M parameters from the inference weight
stream by resolving the AdaRMSNorm condition against the fixed flow-matching
schedule. That is only sound if the folded tables reproduce the unfolded
algebra, so this gate evaluates both forms on the same random activations and
reports the error.

Three claims are checked separately, because they fail for different reasons:

1. **Schedule and time embedding.** The timesteps and their sine-cosine
   embedding must match OpenPI exactly, not approximately -- everything
   downstream is a constant derived from them. `--openpi` compares against
   `create_sinusoidal_pos_embedding` directly; it is off by default because
   importing it drags in transformers and scipy, which costs minutes on a shared
   filesystem for a check the full parity gate subsumes.
2. **Pre-norm sites.** `(rms(x)*(1+scale) + shift) @ W` versus
   `rstd(x) * ((x*ada_scale) @ W) + shift_bias`, for the qkv, ffn-gate and
   ffn-up projections of every layer and step.
3. **Final norm plus the Euler update.** `dt * ((rms(x)*(1+scale) + shift) @
   W_out + b_out)` versus `rstd(x) * (x @ W'_s) + b'_s`, which is the call shape
   Pi0's `tl_scaled_matmul_bias_res` already has.

Errors are reported at two precisions. In float32 the two forms are the same
expression evaluated in a different order, so the residual is rounding only and
should sit near 1e-6 relative. In bfloat16 -- what the target actually stores --
the error is dominated by rounding the tables once, and is reported so the
number is on the record rather than assumed.

No GPU, no checkpoint and no OpenPI checkout are required; the last only
tightens check 1, behind `--openpi`.
"""
from __future__ import annotations

import argparse
import json

import torch

from flash_vla.models.pi05.spec import DECODER_DIM, ENCODER_LAYERS, weight_shapes
from flash_vla.models.pi05.weights import (
    adarms_conditions,
    flow_timesteps,
    fold,
    posemb_sincos,
)

EPS = 1e-6

#: Only the action expert participates in the fold. Everything else -- the
#: vision tower, the 2B prefix, and the 1 GB vocabulary table -- is passed
#: through by reference, so this gate stands them up as zero-strided views and
#: never allocates them. A full `random_checkpoint` is several GB and would say
#: nothing extra.
_EXERCISED = (
    "decoder_time_mlp_", "decoder_ada_rms_", "decoder_attn_qkv_w",
    "decoder_ffn_gate_w", "decoder_ffn_up_w", "decoder_action_out_proj_",
)


def _checkpoint(seed: int) -> dict[str, torch.Tensor]:
    """Random weights where the fold reads, placeholders where it does not."""
    generator = torch.Generator().manual_seed(seed)
    placeholder = torch.zeros(1, dtype=torch.bfloat16)
    checkpoint = {}
    for name, shape in weight_shapes().items():
        if name.startswith(_EXERCISED):
            checkpoint[name] = (torch.randn(shape, dtype=torch.float32,
                                            generator=generator) * 0.05).bfloat16()
        else:
            checkpoint[name] = placeholder.expand(shape)
    return checkpoint


def _rstd(x: torch.Tensor) -> torch.Tensor:
    """The Gemma RMSNorm row factor, computed in float32 as upstream does."""
    return torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + EPS)


def _error(reference: torch.Tensor, folded: torch.Tensor) -> tuple[float, float]:
    """Max absolute and max relative deviation, both in float32."""
    reference, folded = reference.float(), folded.float()
    absolute = (folded - reference).abs()
    scale = reference.abs().max().clamp_min(1e-12)
    return absolute.max().item(), (absolute.max() / scale).item()


def check_schedule(checkpoint: dict[str, torch.Tensor], steps: int,
                   against_openpi: bool = False) -> dict[str, object]:
    """Pin the timestep schedule and its embedding to OpenPI."""
    timesteps, dt = flow_timesteps(steps)
    result: dict[str, object] = {
        "steps": steps,
        "dt": dt,
        "timesteps": [round(float(t), 6) for t in timesteps],
        "cond_norm": adarms_conditions(checkpoint, steps).norm().item(),
        "openpi_sincos_max_abs": None,
    }
    if not against_openpi:
        return result

    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding

    upstream = create_sinusoidal_pos_embedding(
        timesteps, DECODER_DIM, min_period=4e-3, max_period=4.0, device=timesteps.device)
    result["openpi_sincos_max_abs"] = (posemb_sincos(timesteps).float()
                                       - upstream.float()).abs().max().item()
    return result


def check_prenorms(checkpoint: dict[str, torch.Tensor], folded: dict[str, torch.Tensor],
                   steps: int, rows: int, seed: int) -> dict[str, dict[str, float]]:
    """Compare the two forms at every layer and step, for all three projections."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, DECODER_DIM), generator=generator)
    rstd = _rstd(x)
    cond = adarms_conditions(checkpoint, steps)

    sites = {
        "qkv": ("decoder_ada_rms_attn_w", "decoder_ada_rms_attn_b",
                "decoder_attn_qkv_w", "decoder_ada_attn_scale", "decoder_qkv_shift_bias"),
        "ffn_gate": ("decoder_ada_rms_ffn_w", "decoder_ada_rms_ffn_b",
                     "decoder_ffn_gate_w", "decoder_ada_ffn_scale",
                     "decoder_ffn_gate_shift_bias"),
        "ffn_up": ("decoder_ada_rms_ffn_w", "decoder_ada_rms_ffn_b",
                   "decoder_ffn_up_w", "decoder_ada_ffn_scale",
                   "decoder_ffn_up_shift_bias"),
    }

    report: dict[str, dict[str, float]] = {}
    for site, (mod_w, mod_b, proj, scale_key, bias_key) in sites.items():
        worst_abs = worst_rel = 0.0
        for layer in range(ENCODER_LAYERS):
            weight = checkpoint[proj][layer].float()
            modulation = cond @ checkpoint[mod_w][layer].float() + checkpoint[mod_b][layer].float()
            scale, shift, _ = torch.chunk(modulation, 3, dim=-1)
            for step in range(steps):
                reference = ((rstd * x) * (1.0 + scale[step]) + shift[step]) @ weight
                candidate = (rstd * ((x * folded[scale_key][step, layer].float()) @ weight)
                             + folded[bias_key][step, layer].float())
                absolute, relative = _error(reference, candidate)
                worst_abs, worst_rel = max(worst_abs, absolute), max(worst_rel, relative)
        report[site] = {"max_abs": worst_abs, "max_rel": worst_rel}
    return report


def check_output(checkpoint: dict[str, torch.Tensor], folded: dict[str, torch.Tensor],
                 steps: int, rows: int, seed: int) -> dict[str, float]:
    """Compare the final norm folded into the output projection, dt included."""
    generator = torch.Generator().manual_seed(seed + 1)
    x = torch.randn((rows, DECODER_DIM), generator=generator)
    rstd = _rstd(x)
    cond = adarms_conditions(checkpoint, steps)
    _, dt = flow_timesteps(steps)

    modulation = (cond @ checkpoint["decoder_ada_rms_final_w"].float()
                  + checkpoint["decoder_ada_rms_final_b"].float())
    scale, shift, _ = torch.chunk(modulation, 3, dim=-1)
    weight = checkpoint["decoder_action_out_proj_w"].float()
    bias = checkpoint["decoder_action_out_proj_b"].float()

    worst_abs = worst_rel = 0.0
    for step in range(steps):
        reference = dt * (((rstd * x) * (1.0 + scale[step]) + shift[step]) @ weight + bias)
        candidate = (rstd * (x @ folded["decoder_action_out_proj_w"][step].float())
                     + folded["decoder_action_out_proj_b"][step].float())
        absolute, relative = _error(reference, candidate)
        worst_abs, worst_rel = max(worst_abs, absolute), max(worst_rel, relative)
    return {"max_abs": worst_abs, "max_rel": worst_rel}


def run(steps: int = 10, rows: int = 50, seed: int = 0, tolerance: float = 1e-4,
        against_openpi: bool = False) -> dict[str, object]:
    """Fold a random checkpoint at both precisions and report the deviation."""
    checkpoint = _checkpoint(seed)

    report: dict[str, object] = {
        "schedule": check_schedule(checkpoint, steps, against_openpi)}
    for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        folded = fold(checkpoint, steps, dtype=dtype)
        report[name] = {
            "prenorms": check_prenorms(checkpoint, folded, steps, rows, seed),
            "output": check_output(checkpoint, folded, steps, rows, seed),
        }

    exact = report["schedule"]["openpi_sincos_max_abs"]
    worst = max(site["max_rel"] for site in report["float32"]["prenorms"].values())
    worst = max(worst, report["float32"]["output"]["max_rel"])
    report["worst_float32_rel"] = worst
    report["passed"] = worst < tolerance and (exact is None or exact == 0.0)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10, help="flow-matching steps")
    parser.add_argument("--rows", type=int, default=50, help="action-chunk rows to test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="max relative deviation allowed in float32")
    parser.add_argument("--openpi", action="store_true",
                        help="also pin the time embedding to OpenPI (slow import)")
    args = parser.parse_args(argv)
    passed = run(steps=args.steps, rows=args.rows, seed=args.seed,
                 tolerance=args.tolerance, against_openpi=args.openpi)["passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
