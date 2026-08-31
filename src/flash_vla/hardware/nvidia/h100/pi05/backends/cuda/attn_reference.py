"""Torch mirror of `attn_taskloop_launch`, task body by task body.

Two references exist for this block and they answer different questions.
`models/pi05/reference.py` is the ALGORITHM: clean shapes, no padding, no task
structure, and it is what a spec is written against. This one is the KERNEL's
shape: the same buffers in the same layouts, mutated in place, decomposed into
the same three task bodies over the same split structure. It is what you read
next to the CUDA source to check that a task computes what it should, and what
a parity gate bisects against when one task kind is wrong.

Not bit-exact by construction, and not trying to be. Accumulation is fp32 and
stores are bf16 at the same points the kernel rounds, but the kernel forms
`x * ada_scale` in bf16 inside its mainloop and applies RoPE to an fp32
accumulator held in registers; small residuals against this module are expected
and are not a defect. `eval/correctness/pi05/kernel_parity.py` owns the
rounding-exact references for the TileLang call sites.

Geometry is PARSED from `kernels/sm90_attn_task_desc.cuh` rather than restated.
The FFN prototype recorded source, spec and planner disagreeing about BK, CTA
geometry and activation layout as its most expensive drift; one source of truth
removes that failure mode for this kernel.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

_HEADER = Path(__file__).parent / "kernels" / "sm90_attn_task_desc.cuh"
_DECL = re.compile(r"^constexpr\s+int(?:64_t)?\s+(\w+)\s*=\s*([^;]+);", re.M)


def geometry(path: Path = _HEADER) -> dict[str, int]:
    """Every `constexpr int` in the ABI header, evaluated in declaration order.

    Declarations reference earlier ones, so a single forward pass with the
    accumulated namespace resolves all of them.
    """
    out: dict[str, int] = {}
    for name, expr in _DECL.findall(path.read_text()):
        out[name] = int(eval(expr, {"__builtins__": {}}, dict(out)))  # noqa: S307
    return out


G = geometry()
M, M_PAD, D, H, DH = G["M"], G["M_PAD"], G["D"], G["H"], G["DH"]
QKV_W, PREFIX_LEN, KEYS, KEYS_PAD = G["QKV_W"], G["PREFIX_LEN"], G["KEYS"], G["KEYS_PAD"]
QKV_BN, QKV_SPLIT, QKV_TILES = G["QKV_BN"], G["QKV_SPLIT"], G["QKV_TILES"]
ATTN_SPLIT = G["ATTN_SPLIT"]
OUT_BN, OUT_SPLIT, OUT_TILES = G["OUT_BN"], G["OUT_SPLIT"], G["OUT_TILES"]

#: Q and K rotate, V does not: RoPE covers the first (H+1)*DH projection columns.
ROPE_COLS = (H + 1) * DH
#: Additive mask value for a padded key. Finite, so a fully masked tile gives a
#: uniform softmax rather than NaN.
MASK_NEG = -3.0e38
SCALE = DH ** -0.5


# ---------------------------------------------------------------------------
# Task kind 0 -- QKV projection
# ---------------------------------------------------------------------------
def qkv_partial(column: int, split: int, *, x, ada_scale, w_qkv) -> torch.Tensor:
    """One (n-tile, k-split) CTA's contribution, fp32 and unreduced.

    Returns (M_PAD, QKV_BN). `ada_scale` is indexed by the CONTRACTION axis, so
    it multiplies x before the reduction and cannot ride the epilogue the way
    the bias and the gate do.
    """
    k0, k1 = split * (D // QKV_SPLIT), (split + 1) * (D // QKV_SPLIT)
    n0 = column * QKV_BN
    a = (x[:, k0:k1] * ada_scale[None, k0:k1]).bfloat16()
    return a.float() @ w_qkv[k0:k1, n0:n0 + QKV_BN].float()


def qkv_epilogue(column: int, acc: torch.Tensor, *, rms_factor, qkv_bias, rope) -> torch.Tensor:
    """Split 0's epilogue on the reduced accumulator: rstd, bias, then RoPE.

    Ordering is load-bearing and silent if wrong: the folded form is
    `rstd * ((x*s) @ W) + b` and only THEN the rotation. Rotating first computes
    a different function that still looks plausible.
    """
    n0 = column * QKV_BN
    acc = acc * rms_factor.float()[:, None] + qkv_bias.float()[None, n0:n0 + QKV_BN]
    if n0 >= ROPE_COLS:                      # the V slice passes through
        return acc
    d0 = n0 % DH                             # tiles are 64 wide, so pairs never straddle
    cos = rope.float()[:, d0:d0 + QKV_BN:2]
    sin = rope.float()[:, d0 + 1:d0 + QKV_BN:2]
    even, odd = acc[:, 0::2], acc[:, 1::2]
    rotated = torch.empty_like(acc)
    rotated[:, 0::2] = even * cos - odd * sin
    rotated[:, 1::2] = odd * cos + even * sin
    return rotated


def qkv_scatter(column: int, value: torch.Tensor, *, q_buf, k_cache, v_cache) -> None:
    """Write one finished n-tile to Q, or to the suffix rows of the KV cache.

    Q is HEAD-MAJOR so this store, attention's load, and the output
    projection's k-slice read are all contiguous. The cache's prefix rows are
    read-only here; only [PREFIX_LEN, KEYS) belong to this chunk, and rows
    [KEYS, KEYS_PAD) stay whatever the caller zeroed them to.
    """
    n0 = column * QKV_BN
    value = value.bfloat16()
    if n0 < H * DH:
        q_buf[n0 // DH, :, n0 % DH:n0 % DH + QKV_BN] = value
    elif n0 < H * DH + DH:
        d0 = n0 - H * DH
        k_cache[PREFIX_LEN:KEYS, d0:d0 + QKV_BN] = value[:M]
    else:
        d0 = n0 - H * DH - DH
        v_cache[PREFIX_LEN:KEYS, d0:d0 + QKV_BN] = value[:M]


# ---------------------------------------------------------------------------
# Task kind 1 -- multi-query attention, one (head, kv-split) per CTA
# ---------------------------------------------------------------------------
def attention_partial(head: int, split: int, *, q_buf, k_cache, v_cache, key_mask):
    """One CTA's online-softmax pass over its slice of the key axis.

    Returns `(o, m, l)` UNNORMALISED -- `o` is the weighted sum of V and `l`
    the softmax denominator, both relative to this slice's running maximum `m`.
    That triple is what the partial buffers carry; normalising here would make
    the combine below impossible.

    All H query heads share one key/value head, so the slice each CTA reads is
    the same for every head. That re-read is why the split count stops paying
    past its knee.
    """
    span = KEYS_PAD // ATTN_SPLIT
    k0, k1 = split * span, (split + 1) * span
    logits = q_buf[head].float() @ k_cache[k0:k1].float().T
    logits = logits * SCALE + key_mask.float()[None, k0:k1]

    m = logits.amax(dim=-1)                                  # (M_PAD,)
    p = torch.exp(logits - m[:, None])
    return p @ v_cache[k0:k1].float(), m, p.sum(dim=-1)


def attention_combine(partials) -> torch.Tensor:
    """Split 0 folds the other splits in: rescale to a common maximum, then sum.

    `partials` is the per-split `(o, m, l)` in split order.
    """
    m_all = torch.stack([m for _, m, _ in partials])         # (splits, M_PAD)
    m_max = m_all.amax(dim=0)
    o = sum(o_s * torch.exp(m_s - m_max)[:, None] for o_s, m_s, _ in partials)
    l = sum(l_s * torch.exp(m_s - m_max) for _, m_s, l_s in partials)
    return (o / l[:, None]).bfloat16()


# ---------------------------------------------------------------------------
# Task kind 2 -- gated output projection
# ---------------------------------------------------------------------------
def out_proj_partial(column: int, split: int, *, o_buf, w_o) -> torch.Tensor:
    """One (n-tile, head) CTA's contribution, fp32 and unreduced.

    The split count IS the head count: split `h` is exactly head `h`'s share of
    the H*DH-wide contraction, which is why its only dependency is "head h is
    combined".
    """
    n0 = column * OUT_BN
    k0 = split * DH
    return o_buf[split].float() @ w_o[k0:k0 + DH, n0:n0 + OUT_BN].float()


def out_proj_epilogue(column: int, acc: torch.Tensor, *, ada_gate, out) -> None:
    """Split 0's epilogue: gate the reduced accumulator into the residual.

    `out` arrives holding the residual and leaves holding the sum, aliased as
    the kernel does it.
    """
    n0 = column * OUT_BN
    gated = out[:, n0:n0 + OUT_BN].float() + acc * ada_gate.float()[None, n0:n0 + OUT_BN]
    out[:, n0:n0 + OUT_BN] = gated.bfloat16()


# ---------------------------------------------------------------------------
# The whole table
# ---------------------------------------------------------------------------
def run(*, x, rms_factor, ada_scale, w_qkv, qkv_bias, rope, key_mask, w_o, ada_gate,
        k_cache, v_cache, out, q_buf, o_buf) -> None:
    """Execute every task in dependency order, mutating the caller's buffers.

    Argument names, shapes and dtypes mirror `attn_taskloop_launch`; the
    scheduling arguments it also takes (task table, counters, partial buffers)
    are not part of the algorithm and have no analogue here. Buffers marked
    inout in the ABI are mutated in place, `out` included.

        x           (M_PAD, D)      bf16   rows M..M_PAD-1 must be zero
        rms_factor  (M_PAD,)        bf16   rsqrt(mean(x^2)+eps), computed outside
        ada_scale   (D,)            bf16
        w_qkv       (D, QKV_W)      bf16
        qkv_bias    (QKV_W,)        bf16
        rope        (M_PAD, DH)     bf16   cos in even columns, sin in odd
        key_mask    (KEYS_PAD,)     bf16   0 on real keys, MASK_NEG on padding
        w_o         (H*DH, D)       bf16
        ada_gate    (D,)            bf16
        k_cache     (KEYS_PAD, DH)  bf16   inout
        v_cache     (KEYS_PAD, DH)  bf16   inout
        out         (M_PAD, D)      bf16   inout: residual in, sum out
        q_buf       (H, M_PAD, DH)  bf16   scratch, head-major
        o_buf       (H, M_PAD, DH)  bf16   scratch, head-major
    """
    for column in range(QKV_TILES):
        acc = sum(qkv_partial(column, s, x=x, ada_scale=ada_scale, w_qkv=w_qkv)
                  for s in range(QKV_SPLIT))
        value = qkv_epilogue(column, acc, rms_factor=rms_factor, qkv_bias=qkv_bias, rope=rope)
        qkv_scatter(column, value, q_buf=q_buf, k_cache=k_cache, v_cache=v_cache)

    for head in range(H):
        partials = [attention_partial(head, s, q_buf=q_buf, k_cache=k_cache,
                                      v_cache=v_cache, key_mask=key_mask)
                    for s in range(ATTN_SPLIT)]
        o_buf[head] = attention_combine(partials)

    for column in range(OUT_TILES):
        acc = sum(out_proj_partial(column, s, o_buf=o_buf, w_o=w_o)
                  for s in range(OUT_SPLIT))
        out_proj_epilogue(column, acc, ada_gate=ada_gate, out=out)
