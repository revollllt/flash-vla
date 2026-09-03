"""Numerical gate for the fused CUDA encoder attention kernel.

`kernels/enc_attn.cu` is the encoder's multi-query attention: one KV head,
Q already flattened so row = seq * heads + head, an additive per-key mask, and
one launch in place of the QK^T / softmax / PV chain. It depends on the shared
SM90 tile primitives, so this is the regression gate the tile library's README
names for that header set -- editing `tile/sm90/*.cuh` must re-run it.

Three checks, each independently meaningful:

- **chain**: against the torch route it replaced (`encoder_attention`), which
  is what the pipeline ran before and still runs under `ENC_ATTN_ROUTE=torch`.
  This is the comparison a promotion claim rests on.
- **fp32**: against an fp32 recomputation of the same maths, over valid query
  rows only. Padded prefix rows are excluded by construction: this target lets
  them attend normally where OpenPI zeroes them, and they are masked out of
  every later attention (`prefix_parity` makes the same point).
- **finite**: every row of the output, padded rows included. This is not
  redundant. The mask is a large finite negative, and scaled by log2(e) it
  overflows fp32 to -inf; a fully-masked block would then give
  `exp2(-inf - -inf) = NaN`, which the kernel's running-max floor exists to
  prevent. A NaN here would survive the downstream mask (`0 * NaN = NaN`) and
  reach the decoder, so it is checked separately from the value comparison.

Shapes come from the model spec and the production prefix length rather than
being passed in, because the kernel is compiled for one geometry.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F  # noqa: N812

from flash_vla.hardware.nvidia.h100.pi05.backends.cuda import enc_attn
from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels.attention import (
    encoder_attention)
from flash_vla.hardware.nvidia.h100.pi05.buffers import MASK_NEG
from flash_vla.models.pi05.spec import DECODER_HEADS, HEAD_DIM, VISION_TOKENS

#: Production prefix: 3 views x 256 image tokens + a prompt padded to 200.
DEFAULT_SEQ = 3 * VISION_TOKENS + 200
#: Valid keys in the benchmark configuration; the rest carry the mask.
DEFAULT_VALID = 3 * VISION_TOKENS + 131

#: Cosine below this is a real difference rather than reduction-order drift.
MIN_COSINE = 0.99999
#: Elementwise tolerance, relative to the largest reference magnitude. bf16
#: output rounding alone is ~4e-3 relative.
MAX_REL = 1e-2


def error_metrics(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    g, r = got.float().flatten(), ref.float().flatten()
    scale = r.abs().max().item()
    return {"cosine_similarity": F.cosine_similarity(g, r, dim=0).item(),
            "max_abs": (g - r).abs().max().item(),
            "max_rel": (g - r).abs().max().item() / scale if scale else 0.0,
            "rms_error": ((g - r) ** 2).mean().sqrt().item()}


def fp32_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   scale: float, mask: torch.Tensor, seq: int) -> torch.Tensor:
    """The same maths in fp32: (seq*heads, hd) -> (seq*heads, hd)."""
    heads = q.shape[0] // seq
    qh = q.float().view(seq, heads, -1).transpose(0, 1)          # (H, S, D)
    logits = qh @ k.float().T * scale + mask.float()[None, None, :]
    attn = torch.softmax(logits, dim=-1) @ v.float()             # (H, S, D)
    return attn.transpose(0, 1).reshape(q.shape[0], -1)


def run(seq: int, valid: int, seed: int, device: str) -> dict:
    generator = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape, scale: float = 0.5):
        return (torch.randn(shape, generator=generator, device=device,
                            dtype=torch.float32) * scale).bfloat16()

    heads, head_dim = DECODER_HEADS, HEAD_DIM
    scale = head_dim ** -0.5
    q = rand(seq * heads, head_dim)
    k, v = rand(seq, head_dim), rand(seq, head_dim)
    mask = torch.zeros(seq, dtype=torch.bfloat16, device=device)
    mask[valid:] = MASK_NEG
    out = torch.empty_like(q)

    enc_attn.attention(q, k, v, scale, mask, out)
    torch.cuda.synchronize()

    chain = encoder_attention(q, k, v, scale, mask).view(seq * heads, head_dim)
    reference = fp32_reference(q, k, v, scale, mask, seq)
    rows = slice(0, valid * heads)

    report = {
        "config": {"seq": seq, "valid_keys": valid, "heads": heads,
                   "head_dim": head_dim, "seed": seed,
                   "geometry": enc_attn.geometry()},
        "vs_chain": error_metrics(out[rows], chain[rows]),
        "vs_fp32": error_metrics(out[rows], reference[rows]),
        "chain_vs_fp32": error_metrics(chain[rows], reference[rows]),
        "all_rows_finite": bool(torch.isfinite(out.float()).all()),
        "padded_rows_finite": bool(torch.isfinite(out[valid * heads:].float()).all()),
        "thresholds": {"min_cosine": MIN_COSINE, "max_rel": MAX_REL},
    }
    report["passed"] = bool(
        report["all_rows_finite"]
        and report["padded_rows_finite"]
        and all(report[key]["cosine_similarity"] >= MIN_COSINE
                and report[key]["max_rel"] <= MAX_REL
                for key in ("vs_chain", "vs_fp32")))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ,
                        help="prefix length; must be the compiled geometry's")
    parser.add_argument("--valid-keys", type=int, default=DEFAULT_VALID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    report = run(args.seq, args.valid_keys, args.seed, args.device)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
