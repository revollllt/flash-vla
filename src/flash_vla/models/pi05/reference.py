"""Readable torch description of the Pi0.5 action-expert attention block.

This is the *algorithm*, written to be read: one operation per line, every
tensor annotated with its shape, nothing reused in place. It is the input a
fused-kernel spec is written against -- `specs/tile/` names dataflow, this
names the maths the dataflow has to reproduce.

It is deliberately NOT any of these:

- not the fast path (`hardware/nvidia/h100/pi05/` owns that);
- not the parity gate (`eval/correctness/pi05/kernel_parity.py` owns that, and
  its references are the authority on rounding -- this file mirrors them);
- not a training module: there are no parameters here, only folded constants.

Everything the action expert's AdaRMSNorm needs arrives as per-(step, layer)
vectors, folded at checkpoint load by `weights.fold`. Nothing is streamed:

    s = 1 + scale     indexed by K (input width)   scales the GEMM's A operand
    b = shift @ W     indexed by N (output width)  a plain bias
    g = gate          indexed by N (output width)  multiplies the residual

## Shapes, at the reference configuration

    M        50      action chunk (queries; the state token Pi0 carried is gone)
    D      1024      action-expert width (DECODER_DIM)
    H         8      query heads (DECODER_HEADS)
    Dh      256      head width (HEAD_DIM)
    QKV    2560      = H*Dh (Q) + Dh (K) + Dh (V) -- multi-query, one KV head
    S_pre   968      prefix length = 3 views * 256 image tokens + 200 prompt
    KEYS   1018      = S_pre + M; the decoder attends over prefix AND chunk

The block runs 18 layers x 10 flow steps = 180 times per inference.
"""
from __future__ import annotations

import torch

from .spec import DECODER_DIM, DECODER_HEADS, HEAD_DIM, QKV_WIDTH, ROPE_THETA

#: RMSNorm epsilon, matching the kernels.
EPS = 1e-6

#: Q and K rotate; V does not. RoPE therefore covers the first (H+1)*Dh columns
#: of the QKV projection, and the V slice passes through untouched.
ROPE_COLS = (DECODER_HEADS + 1) * HEAD_DIM      # 2304 of 2560

#: Additive mask value for a padded prompt column. Finite, not -inf: -inf
#: produces NaN when a whole tile is masked.
MASK_NEG = -3.0e38


# ---------------------------------------------------------------------------
# Inputs that are computed per inference rather than folded
# ---------------------------------------------------------------------------
def rope_table(chunk: int, n_valid: int) -> torch.Tensor:
    """Cosines and sines for the chunk's positions, interleaved.

    The action chunk sits immediately after the *valid* prefix, so its absolute
    positions are `n_valid + 0..chunk-1`. `n_valid` depends on the tokenized
    state, which is why this table is a per-inference input and not a weight.

    Returns (chunk, Dh), with cos in the even columns and sin in the odd ones,
    so that column pair `(2p, 2p+1)` carries the (cos, sin) of frequency `p` --
    the same pair the rotation below consumes.
    """
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    positions = torch.arange(chunk).float() + n_valid       # (chunk,)
    phase = positions[:, None] * inv_freq[None, :]          # (chunk, Dh/2)
    return torch.stack([phase.cos(), phase.sin()], dim=2).view(chunk, HEAD_DIM)


def key_mask(keys: int, prefix_len: int, n_valid: int) -> torch.Tensor:
    """Additive per-key bias: 0 on real keys, MASK_NEG on prompt padding.

    The prefix is bidirectional and the chunk attends over all of it, so the
    mask is per-KEY only -- one vector, identical for every query row. There is
    no causal structure anywhere in this block.

    Layout of the `keys` axis:  [ 0, n_valid )        real prefix   -> 0
                                [ n_valid, prefix_len ) prompt pad  -> MASK_NEG
                                [ prefix_len, keys )   action chunk -> 0
    """
    mask = torch.zeros(keys)
    mask[n_valid:prefix_len] = MASK_NEG
    return mask


# ---------------------------------------------------------------------------
# 1. QKV projection under AdaRMSNorm, then RoPE
# ---------------------------------------------------------------------------
def qkv_proj(x, s, w_qkv, b, rope):
    """AdaRMS-scale x, project to QKV, add the shift bias, rotate Q and K.

    The folded identity this implements is

        q = rstd(x) * ((x * s) @ W_q) + b_q     and only then    RoPE(q)

    Two orderings in there are load-bearing and silent if wrong:

    - `s` multiplies x *before* the contraction. It is indexed by K, so it sits
      inside the reduction and cannot ride the epilogue the way `b` and `g` do.
      This is the entire reason the AdaRMS kernels exist.
    - `b` is added *before* the rotation. Adding it after computes a different
      function that still looks plausible.

    Shapes
        x      (M, D)          action-expert hidden state
        s      (D,)            1 + scale, per (step, layer)
        w_qkv  (D, QKV)        QKV weight, q/k/v concatenated on the N axis
        b      (QKV,)          shift @ w_qkv, per (step, layer)
        rope   (M, Dh)         from `rope_table`
    Returns
        q      (M, H, Dh)      rotated
        k      (M, Dh)         rotated, one head (multi-query)
        v      (M, Dh)         not rotated
    """
    # rstd is per ROW, so it commutes with the contraction and rides the
    # epilogue. Computed on the UNSCALED x -- `s` is part of the projection,
    # not part of the norm.
    rstd = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPS)  # (M, 1)

    # The kernel forms this product in bf16, in shared memory, inside the
    # mainloop. Rounding here rather than in fp32 is what the reference has to
    # mirror to be comparable.
    a = (x * s[None, :]).bfloat16()                          # (M, D)

    acc = a.float() @ w_qkv.float()                          # (M, QKV)  fp32 accumulator
    acc = acc * rstd + b.float()[None, :]                    # (M, QKV)  epilogue: rstd then bias

    rotated = _rotate_pairs(acc, rope, ROPE_COLS)            # (M, QKV)  Q and K only

    q = rotated[:, :DECODER_HEADS * HEAD_DIM]                # (M, H*Dh)
    k = rotated[:, DECODER_HEADS * HEAD_DIM:][:, :HEAD_DIM]  # (M, Dh)
    v = rotated[:, DECODER_HEADS * HEAD_DIM + HEAD_DIM:]     # (M, Dh)
    return (q.reshape(-1, DECODER_HEADS, HEAD_DIM).bfloat16(),
            k.bfloat16(),
            v.bfloat16())


def _rotate_pairs(acc, rope, columns):
    """Rotate ADJACENT column pairs of the first `columns` columns.

    Pairs are (2p, 2p+1), not the (p, p + Dh/2) split some implementations use.
    OpenPI's checkpoint uses the split form; `spec.weight_shapes()` permutes the
    QKV weight columns offline so that this cheap adjacent-pair form is correct
    here -- the permutation is a weight relayout, never a runtime shuffle.

    `columns` stops the rotation at the V slice, which must pass through.
    """
    cos = rope[:, 0::2].float()                              # (M, Dh/2)
    sin = rope[:, 1::2].float()                              # (M, Dh/2)

    head = acc[:, :columns].reshape(-1, columns // HEAD_DIM, HEAD_DIM // 2, 2)
    even = head[..., 0]                                      # (M, columns/Dh, Dh/2)
    odd = head[..., 1]
    out = torch.empty_like(head)
    out[..., 0] = even * cos[:, None, :] - odd * sin[:, None, :]
    out[..., 1] = odd * cos[:, None, :] + even * sin[:, None, :]

    return torch.cat([out.reshape(acc.shape[0], columns), acc[:, columns:]], dim=1)


# ---------------------------------------------------------------------------
# 2. Multi-query attention over the KV cache
# ---------------------------------------------------------------------------
def append_to_cache(k_cache, v_cache, k, v, prefix_len):
    """Write this step's chunk K/V into the suffix rows of the cache.

    The prefix rows [0, prefix_len) were built once by the prefix stage and are
    read-only for all 180 block invocations. The suffix rows are rewritten every
    layer and every flow step, because the chunk itself changes.

    Shapes
        k_cache, v_cache  (KEYS, Dh)   KEYS = prefix_len + M
        k, v              (M, Dh)
    """
    k_cache = k_cache.clone()                                # readable, not the fast path
    v_cache = v_cache.clone()
    k_cache[prefix_len:] = k
    v_cache[prefix_len:] = v
    return k_cache, v_cache


def attention(q, k_cache, v_cache, mask):
    """Multi-query attention: H query heads against ONE key/value head.

    Bidirectional over the whole key axis. Every query row sees the same mask,
    so the softmax denominator differs across rows only through the logits.

    The H heads share k_cache/v_cache entirely -- that is the multi-query part,
    and it is why the arithmetic intensity here is high (each cached byte feeds
    8 query heads) while every other decoder op is weight-bandwidth bound.

    Shapes
        q                 (M, H, Dh)
        k_cache, v_cache  (KEYS, Dh)
        mask              (KEYS,)          additive, 0 or MASK_NEG
    Returns
        out               (M, H, Dh)
    """
    scale = HEAD_DIM ** -0.5                                 # 0.0625 at Dh=256

    logits = torch.einsum("mhd,kd->mhk", q.float(), k_cache.float())   # (M, H, KEYS)
    logits = logits * scale + mask.float()[None, None, :]

    probs = torch.softmax(logits, dim=-1)                    # (M, H, KEYS)  fp32

    out = torch.einsum("mhk,kd->mhd", probs, v_cache.float())          # (M, H, Dh)
    return out.bfloat16()


# ---------------------------------------------------------------------------
# 3. Output projection, gated, into the residual
# ---------------------------------------------------------------------------
def o_proj_residual(attn_out, w_o, g, residual):
    """residual + (attn @ W_o) * g.

    `g` is indexed by the output width, so it is a plain epilogue multiply --
    the cheap half of AdaRMSNorm. The heads are concatenated, not summed: the
    (M, H, Dh) attention output is reinterpreted as (M, H*Dh) and W_o contracts
    over that whole axis.

    Shapes
        attn_out  (M, H, Dh)
        w_o       (H*Dh, D)
        g         (D,)              gate, per (step, layer)
        residual  (M, D)
    Returns
        (M, D)
    """
    flat = attn_out.reshape(-1, DECODER_HEADS * HEAD_DIM)    # (M, H*Dh) = (M, 2048)
    delta = flat.float() @ w_o.float()                       # (M, D)  fp32 accumulator
    return (residual.float() + delta * g.float()[None, :]).bfloat16()


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------
def attention_block(x, s, w_qkv, b, rope, k_cache, v_cache, mask, w_o, g, prefix_len):
    """One layer's attention half, for one flow step: x -> x + gated attention.

    The FFN half that follows it (AdaRMS -> gated FFN -> down-projection into
    the same residual) is a separate chain; see `specs/tile/ffn_taskloop.md`.

    Returns (x_next, k_cache, v_cache) -- the caches are returned rather than
    mutated so the data flow is visible.
    """
    q, k, v = qkv_proj(x, s, w_qkv, b, rope)                 # (M,H,Dh) (M,Dh) (M,Dh)
    k_cache, v_cache = append_to_cache(k_cache, v_cache, k, v, prefix_len)
    attn = attention(q, k_cache, v_cache, mask)              # (M, H, Dh)
    x_next = o_proj_residual(attn, w_o, g, x)                # (M, D)
    return x_next, k_cache, v_cache


# ---------------------------------------------------------------------------
# Shape and cost trace: `python -m flash_vla.models.pi05.reference`
# ---------------------------------------------------------------------------
def _trace(chunk: int = 50, prefix_len: int = 968, n_valid: int = 903, seed: int = 0):
    """Run the block once at the reference shapes and print what it moved.

    Runs on CPU. The per-invocation costs below are what a fused kernel is
    budgeted against; multiply by 18 layers x 10 steps for the stage.
    """
    torch.manual_seed(seed)
    keys = prefix_len + chunk
    heads, dh, d, qkv = DECODER_HEADS, HEAD_DIM, DECODER_DIM, QKV_WIDTH

    def rand(*shape):
        return (torch.randn(shape) * 0.05).bfloat16()

    x = rand(chunk, d)
    s = (1.0 + torch.randn(d) * 0.1).bfloat16()
    w_qkv, b = rand(d, qkv), rand(qkv)
    w_o, g = rand(heads * dh, d), rand(d)
    k_cache, v_cache = rand(keys, dh), rand(keys, dh)
    rope = rope_table(chunk, n_valid).bfloat16()
    mask = key_mask(keys, prefix_len, n_valid)

    q, k, v = qkv_proj(x, s, w_qkv, b, rope)
    k_cache, v_cache = append_to_cache(k_cache, v_cache, k, v, prefix_len)
    attn = attention(q, k_cache, v_cache, mask)
    x_next = o_proj_residual(attn, w_o, g, x)

    rows = [
        ("qkv_proj", f"({chunk},{d}) @ ({d},{qkv})", tuple(q.shape),
         2 * chunk * d * qkv, d * qkv * 2),
        ("attention QK", f"({chunk},{heads},{dh}) x ({keys},{dh})", tuple(attn.shape),
         2 * chunk * heads * dh * keys, keys * dh * 2),
        ("attention PV", f"({chunk},{heads},{keys}) x ({keys},{dh})", tuple(attn.shape),
         2 * chunk * heads * keys * dh, keys * dh * 2),
        ("o_proj", f"({chunk},{heads * dh}) @ ({heads * dh},{d})", tuple(x_next.shape),
         2 * chunk * heads * dh * d, heads * dh * d * 2),
    ]
    print(f"M={chunk} D={d} H={heads} Dh={dh} QKV={qkv} "
          f"prefix={prefix_len} n_valid={n_valid} KEYS={keys}\n")
    print(f"{'op':<14}{'contraction':<34}{'out':<18}{'MFLOP':>8}{'MB read':>9}{'FLOP/B':>8}")
    for name, contraction, out, flop, byts in rows:
        print(f"{name:<14}{contraction:<34}{str(out):<18}"
              f"{flop / 1e6:>8.1f}{byts / 1e6:>9.2f}{flop / byts:>8.1f}")
    total_flop = sum(r[3] for r in rows)
    total_bytes = sum(r[4] for r in rows)
    print(f"\n{'block':<14}{'':<34}{'':<18}{total_flop / 1e6:>8.1f}"
          f"{total_bytes / 1e6:>9.2f}{total_flop / total_bytes:>8.1f}")
    print(f"\nx {tuple(x.shape)} -> x_next {tuple(x_next.shape)}, "
          f"finite={bool(torch.isfinite(x_next).all())}")
    print(f"cache suffix rows [{prefix_len}:{keys}) rewritten every invocation "
          f"({chunk * dh * 2 * 2 / 1e3:.1f} KB)")
    return x_next


if __name__ == "__main__":
    _trace()
