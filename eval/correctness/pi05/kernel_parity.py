"""Numerical gate for the four Pi0.5 AdaRMSNorm decoder kernels.

Each variant of `specs/tile/pi05-adarms-decoder.md` against a plain-torch
recomputation of the same maths, on the same random tensors. This is the check
that localizes a failure to the kernel rather than to the weight conversion --
the split that found the RoPE bug during the prefix bring-up, where the kernel
matched torch to 2.7e-3 and the discrepancy turned out to be in the reference.

Deliberately not a comparison against OpenPI. OpenPI has no AdaRMSNorm kernel to
compare against, only a whole model; suffix parity against it is the *next*
gate, and it can only be read once these four are known good in isolation.

What "the same maths" means matters, so the references mirror the kernels'
precision rather than an idealized fp32:

- the A-tile scale happens in bfloat16 inside the mainloop, because that is what
  the kernel does and what Pi0's `tl_scaled_gate` did before it;
- the accumulator and every epilogue term are fp32;
- `tl_ada_qkv_gemm_rope` keeps the rms factor in the fp32 epilogue while
  `tl_ada_scaled_gate` folds it into the bf16 A-tile scale, each following its
  origin kernel.

A reference that rounded differently would report a difference that is real but
uninteresting, which is the failure mode this docstring exists to prevent.
"""
from __future__ import annotations

import argparse
import json

import torch

from eval.correctness.pi05.prefix_parity import error_metrics
from flash_vla.models.pi05.spec import (
    ACTION_DIM,
    DECODER_DIM,
    DECODER_FFN,
    DECODER_HEADS,
    HEAD_DIM,
    QKV_WIDTH,
)

CHUNK = 50
KEYS = 1018
EPS = 1e-6
ROPE_COLS = (DECODER_HEADS + 1) * HEAD_DIM      # Q and K rotate; V does not


def _rstd(x: torch.Tensor) -> torch.Tensor:
    return torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPS)


def _rope_pairs(x: torch.Tensor, rope: torch.Tensor, columns: int) -> torch.Tensor:
    """Rotate adjacent column pairs of the first `columns` columns, as the kernel does."""
    out = x.clone()
    cos = rope[:, 0::2].float()
    sin = rope[:, 1::2].float()
    heads = columns // HEAD_DIM
    view = out[:, :columns].view(-1, heads, HEAD_DIM // 2, 2)
    a, b = view[..., 0].clone(), view[..., 1].clone()
    view[..., 0] = a * cos[:, None, :] - b * sin[:, None, :]
    view[..., 1] = b * cos[:, None, :] + a * sin[:, None, :]
    return out


def check_qkv_rope(gen, device) -> dict[str, float]:
    """Variant A: rstd * ((x*s) @ W) + bias, then RoPE on Q and K."""
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers

    def rand(*shape):
        return (torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
                * 0.05).bfloat16()

    x = rand(CHUNK, DECODER_DIM)
    s = (1.0 + torch.randn((DECODER_DIM,), generator=gen, device=device) * 0.1).bfloat16()
    w = rand(DECODER_DIM, QKV_WIDTH)
    bias = rand(QKV_WIDTH)
    rope = rand(CHUNK, HEAD_DIM)

    Q = torch.empty((CHUNK * DECODER_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    K = torch.empty((CHUNK, HEAD_DIM), dtype=torch.bfloat16, device=device)
    V = torch.empty((CHUNK, HEAD_DIM), dtype=torch.bfloat16, device=device)
    factor = torch.empty((CHUNK,), dtype=torch.bfloat16, device=device)
    wrappers.decoder_norm_qkv_rope(x, s, w, bias, rope, Q, K, V, factor)
    torch.cuda.synchronize()

    scaled = (x * s[None, :])                                   # bf16, as in the mainloop
    acc = scaled.float() @ w.float()
    acc = acc * _rstd(x) + bias.float()[None, :]                # fp32 epilogue, then rope
    reference = _rope_pairs(acc, rope, ROPE_COLS)

    got = torch.cat([Q.view(CHUNK, DECODER_HEADS * HEAD_DIM), K, V], dim=1)
    return error_metrics(reference.bfloat16(), got)


def check_gated_ffn(gen, device) -> dict[str, float]:
    """Variant B: gelu(a @ W1 + b1) * (a @ W2 + b2) with a = bf16(x * F * s)."""
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels.base import (
        GELU_C0, GELU_C1)

    def rand(*shape):
        return (torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
                * 0.05).bfloat16()

    x = rand(CHUNK, DECODER_DIM)
    s = (1.0 + torch.randn((DECODER_DIM,), generator=gen, device=device) * 0.1).bfloat16()
    w1, w2 = rand(DECODER_DIM, DECODER_FFN), rand(DECODER_DIM, DECODER_FFN)
    b1, b2 = rand(DECODER_FFN), rand(DECODER_FFN)

    out = torch.empty((CHUNK, DECODER_FFN), dtype=torch.bfloat16, device=device)
    factor = torch.empty((CHUNK,), dtype=torch.bfloat16, device=device)
    wrappers.decoder_norm_gated_ffn(x, s, w1, w2, b1, b2, out, factor)
    torch.cuda.synchronize()

    a = (x * _rstd(x).bfloat16() * s[None, :])                  # bf16 throughout, as in the kernel
    c1 = a.float() @ w1.float() + b1.float()[None, :]
    c2 = a.float() @ w2.float() + b2.float()[None, :]
    gelu = c1 * torch.sigmoid(GELU_C0 * c1 * (1.0 + GELU_C1 * c1 * c1))
    return error_metrics((gelu * c2).bfloat16(), out)


def check_gated_res(gen, device, k: int) -> dict[str, float]:
    """Variant C: out = R + (x @ W) * g."""
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers

    def rand(*shape):
        return (torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
                * 0.05).bfloat16()

    x = rand(CHUNK, k)
    w = rand(k, DECODER_DIM)
    g = rand(DECODER_DIM)
    residual = rand(CHUNK, DECODER_DIM)

    reference = (residual.float() + (x.float() @ w.float()) * g.float()[None, :]).bfloat16()
    out = residual.clone()
    wrappers.decoder_out_proj_residual(x, w, g, out)
    torch.cuda.synchronize()
    return error_metrics(reference, out)


def check_attention(gen, device) -> dict[str, float]:
    """Variant D: multi-query attention with an additive per-key mask."""
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers

    def rand(*shape):
        return (torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
                * 0.05).bfloat16()

    flat = CHUNK * DECODER_HEADS
    q = rand(flat, HEAD_DIM)
    k = rand(KEYS, HEAD_DIM)
    v = rand(KEYS, HEAD_DIM)

    # The real mask shape: valid prefix, a hole of prompt padding, then the suffix.
    mask = torch.zeros((KEYS,), dtype=torch.bfloat16, device=device)
    mask[903:968] = -3.0e38

    logits = (q.float() @ k.float().T) * (HEAD_DIM ** -0.5) + mask.float()[None, :]
    reference = (torch.softmax(logits, dim=-1) @ v.float()).bfloat16()

    out = torch.empty_like(q)
    wrappers.decoder_attention(q, k, v, mask, out)
    torch.cuda.synchronize()
    return error_metrics(reference, out)


def run(seed: int = 0, device: str = "cuda", tolerance: float = 0.9999,
        only: str | None = None) -> dict[str, object]:
    """Run every variant, or just the one named by `only`, and report its metrics.

    `only` exists because a TileLang kernel that deadlocks does not fail, it
    hangs -- one variant per job keeps a hang from hiding the other three.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this command on an H100 GPU node")
    torch_device = torch.device(device)
    gen = torch.Generator(device=torch_device).manual_seed(seed)

    checks = {
        "A_qkv_rope": lambda: check_qkv_rope(gen, torch_device),
        "B_gated_ffn": lambda: check_gated_ffn(gen, torch_device),
        "C_gated_res_out_proj": lambda: check_gated_res(gen, torch_device, DECODER_HEADS * HEAD_DIM),
        "C_gated_res_ffn_down": lambda: check_gated_res(gen, torch_device, DECODER_FFN),
        "D_attention": lambda: check_attention(gen, torch_device),
    }
    if only is not None:
        checks = {k: v for k, v in checks.items() if k.startswith(only)}
        if not checks:
            raise SystemExit(f"no variant matches {only!r}")
    report = {}
    for name, fn in checks.items():
        print(f"[gate] {name} ...", flush=True)
        report[name] = fn()
        print(f"[gate] {name} cosine={report[name]['cosine_similarity']:.7f}", flush=True)
    report["worst_cosine"] = min(v["cosine_similarity"] for v in report.values()
                                 if isinstance(v, dict))
    report["tolerance"] = tolerance
    report["passed"] = bool(report["worst_cosine"] > tolerance)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=0.9999,
                        help="minimum cosine against the torch recomputation")
    parser.add_argument("--only", default=None,
                        help="run one variant by name prefix, e.g. A / B / C / D")
    args = parser.parse_args(argv)
    return 0 if run(args.seed, args.device, args.tolerance, args.only)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
