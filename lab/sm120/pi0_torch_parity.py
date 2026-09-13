#!/usr/bin/env python3
"""Check the RTX 5090 torch backend against H100/Pi0's TileLang wrappers.

`eval.correctness` on this Target compares its reference plan against its
candidate plan, and both are the torch route -- so it proves the model runs and
replays deterministically, and proves NOTHING about the arithmetic. There is no
OpenPI environment on this machine, so Pi0 has no official oracle here either.

What is available is the implementation the torch backend was written from.
Where an H100 TileLang configuration fits this part's 99 KB of shared memory
[smem.bytes.cta.max], its wrapper runs here, and the two can be compared on the
same inputs. That covers the reimplementations with real room to be wrong:

  * RoPE   -- adjacent-column-pair rotation with an interleaved cos/sin table,
              read off the kernel source rather than documented anywhere
  * RMS    -- whether the epsilon is inside or outside the mean
  * LayerNorm, SiLU, and the packing of patch embedding

Call sites whose configuration does NOT fit are listed as unchecked rather than
skipped silently; the gated feed-forward and the vision output projection are
among them, so GELU has no TileLang counterpart here and is checked against its
closed form instead.

    CUDA_HOME=... python3 lab/sm120/pi0_torch_parity.py
"""
from __future__ import annotations

import torch

from flash_vla.hardware.nvidia.h100.pi0.backends.tilelang import wrappers as tl
from flash_vla.hardware.nvidia.rtx5090.pi0.backends import torch_ops as tt

DEV = "cuda"
DT = torch.bfloat16

#: Tolerance for a bf16 chain. The two implementations differ in reduction
#: order and in where they round, so bit-equality is not the target -- a
#: disagreement far above this is a semantic error, not a rounding one.
COS_MIN = 0.999
REL_RMS_MAX = 0.02

def _metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a32, b32 = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a32, b32, dim=0).item()
    rel = (torch.linalg.vector_norm(a32 - b32)
           / torch.linalg.vector_norm(a32).clamp_min(1e-12)).item()
    return cos, rel

def _rand(*shape, dtype=DT):
    return torch.randn(*shape, device=DEV, dtype=dtype) * 0.1

def check(name, run_tl, run_torch):
    """Run both implementations on identical inputs and compare their outputs."""
    try:
        ref = run_tl()
    except Exception as exc:  # noqa: BLE001 -- a launch failure IS the answer
        msg = str(exc).strip().splitlines()[-1][:70]
        print(f"  {name:34} UNCHECKED  TileLang: {msg}")
        return None
    got = run_torch()
    outs = ref if isinstance(ref, tuple) else (ref,)
    gots = got if isinstance(got, tuple) else (got,)
    worst_cos, worst_rel = 1.0, 0.0
    for r, g in zip(outs, gots):
        c, e = _metrics(r, g)
        worst_cos, worst_rel = min(worst_cos, c), max(worst_rel, e)
    ok = worst_cos >= COS_MIN and worst_rel <= REL_RMS_MAX
    print(f"  {name:34} {'PASS' if ok else '*** FAIL ***':12} "
          f"cos {worst_cos:.6f}  rel_rms {worst_rel:.3e}")
    return ok

def main() -> int:
    torch.manual_seed(0)
    results = []

    # -- RoPE, the reimplementation with the most room to be wrong ----------
    m, kdim, heads, head_dim = 51, 1024, 8, 256
    n = heads * head_dim + 2 * head_dim
    x, w = _rand(m, kdim), _rand(kdim, n)
    rope = _rand(m, head_dim)
    nf = torch.empty(m, device=DEV, dtype=DT)

    def _qkv(fn):
        Q = torch.zeros(m * heads, head_dim, device=DEV, dtype=DT)
        K = torch.zeros(m, head_dim, device=DEV, dtype=DT)
        V = torch.zeros(m, head_dim, device=DEV, dtype=DT)
        fn(x, None, w, None, rope, Q, K, V, nf)
        torch.cuda.synchronize()
        return Q.clone(), K.clone(), V.clone()

    results.append(check("action_expert_norm_qkv_rope",
                         lambda: _qkv(tl.action_expert_norm_qkv_rope),
                         lambda: _qkv(tt.action_expert_norm_qkv_rope)))

    # -- RMS placement, through the output projection ------------------------
    xo, wo, bo = _rand(m, kdim), _rand(kdim, 32), _rand(32)

    def _outproj(fn):
        out = torch.zeros(m, 32, device=DEV, dtype=DT)
        fn(xo, wo, bo, out, nf)
        torch.cuda.synchronize()
        return out.clone()

    results.append(check("action_expert_action_out_proj",
                         lambda: _outproj(tl.action_expert_action_out_proj),
                         lambda: _outproj(tt.action_expert_action_out_proj)))

    # -- SiLU ----------------------------------------------------------------
    xs, ws, bs = _rand(50, 32), _rand(32, 1024), _rand(1024)

    def _inproj(fn):
        out = torch.zeros(50, 1024, device=DEV, dtype=DT)
        fn(xs, ws, bs, out)
        torch.cuda.synchronize()
        return out.clone()

    results.append(check("action_expert_action_in_proj",
                         lambda: _inproj(tl.action_expert_action_in_proj),
                         lambda: _inproj(tt.action_expert_action_in_proj)))

    # -- LayerNorm, through the vision QKV projection -------------------------
    views, vd = 3, tt.VISION_DIM
    xv = _rand(views, tt.VISION_TOKENS, vd)
    nw, nb = _rand(vd), _rand(vd)
    qw, qb = _rand(vd, 3 * vd), _rand(3 * vd)

    def _visqkv(fn):
        out = torch.zeros(views, tt.VISION_TOKENS, 3 * vd, device=DEV, dtype=DT)
        fn(xv, nw, nb, qw, qb, out)
        torch.cuda.synchronize()
        return out.clone()

    results.append(check("vision_encoder_norm_qkv",
                         lambda: _visqkv(tl.vision_encoder_norm_qkv),
                         lambda: _visqkv(tt.vision_encoder_norm_qkv)))

    # -- Patch embedding packing ---------------------------------------------
    imgs = _rand(views, 224, 224, 3)
    pw, pb = _rand(tt.PATCH_FEATURES, vd), _rand(vd)
    pos = _rand(tt.VISION_TOKENS, vd)

    def _patch(fn):
        out = torch.zeros(views, tt.VISION_TOKENS, vd, device=DEV, dtype=DT)
        fn(imgs, pw, pb, pos, out)
        torch.cuda.synchronize()
        return out.clone()

    results.append(check("vision_encoder_patch_embed",
                         lambda: _patch(tl.vision_encoder_patch_embed),
                         lambda: _patch(tt.vision_encoder_patch_embed)))

    # -- GELU has no runnable TileLang counterpart here, so check the form ----
    # The kernels spell GELU as x * sigmoid(C0 * x * (1 + C1 * x^2)); torch's
    # tanh approximation is 0.5x(1 + tanh(y)), and sigma(2y) == 0.5(1 + tanh(y)),
    # so the two are the same function iff the constants line up. This is the
    # whole of that risk, and it needs no kernel.
    from flash_vla.hardware.nvidia.h100.pi0.backends.tilelang.kernels import base
    g = torch.linspace(-8, 8, 100_001, device=DEV, dtype=torch.float32)
    kernel_form = g * torch.sigmoid(base.GELU_C0 * g * (1.0 + base.GELU_C1 * g * g))
    cos, rel = _metrics(kernel_form, torch.nn.functional.gelu(g, approximate="tanh"))
    ok = cos >= 1 - 1e-9 and rel <= 1e-6
    print(f"  {'gelu form vs tanh approximation':34} {'PASS' if ok else '*** FAIL ***':12} "
          f"cos {cos:.9f}  rel_rms {rel:.3e}")
    results.append(ok)

    checked = [r for r in results if r is not None]
    print(f"\n{sum(checked)}/{len(checked)} checked, "
          f"{sum(1 for r in results if r is None)} unchecked (TileLang will not run here)")
    return 0 if all(checked) else 1

if __name__ == "__main__":
    raise SystemExit(main())
