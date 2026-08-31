"""Torch mirror of `attn_taskloop_launch`, at the algorithm level.

Same inputs, same outputs, same buffers, same mutation -- and deliberately no
tiling: every contraction is one `F.linear` or one `scaled_dot_product_attention`
call. Reading this beside the kernel answers "does the kernel compute the right
function"; it says nothing about whether it computes it well.

It is not bit-exact and does not try to be. The kernel rounds the AdaRMS product
to bf16 inside its mainloop and accumulates in fp32; here the whole chain is
fp32 and rounds once at each buffer boundary, so expect agreement to a cosine of
a few 1e-4 rather than to the last bit.

Three other references exist; each answers a different question and none
replaces this one:

- `attn_reference` (beside this file) has the same ABI but is decomposed TASK BY
  TASK, over the same split structure as the kernel. Read that one to find WHICH
  task body is wrong; read this one to decide whether the block computes the
  right function at all. Geometry is imported from it so there is one mirror of
  the header, not two.
- `flash_vla.models.pi05.reference` is the hardware-independent algorithm with
  no padding and no ABI, split into the three call sites. This module is checked
  against it.
- `eval.correctness.pi05.kernel_parity` owns the ROUNDING contract and is the
  authority when a parity number is in dispute.

`kernels/sm90_attn_task_desc.cuh` remains authoritative for shapes, layouts and
geometry.

One invariant makes the padding safe and is worth checking while reviewing:
**the whole block is row-independent along M.** The projections, the attention
and the output projection all map row `m` to row `m`, so the padded rows
`[M, M_PAD)` cannot contaminate a real row and are simply never read.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Geometry comes from `attn_reference`, which evaluates it straight out of the
# ABI header, so there is exactly one mirror of `sm90_attn_task_desc.cuh` in
# python and this file cannot drift from it.
from .attn_reference import (
    DH, D, H, KEYS, KEYS_PAD, M, MASK_NEG, M_PAD, PREFIX_LEN, QKV_W, ROPE_COLS, SCALE,
)

#: RMSNorm epsilon, matching the kernels. Only `make_inputs` needs it: the block
#: itself takes the factor as an input.
EPS = 1e-6


class AttnBlockReference:
    """One layer-step of the action-expert attention block, in plain torch.

    Call `forward` with exactly the tensors `attn_taskloop_launch` takes. All
    tensors are 2-D or 3-D, contiguous, row-major, on one device, and bf16
    unless noted. `k_cache`, `v_cache` and `out` are mutated in place, as the
    kernel mutates them; `q_buf` and `o_buf` are filled so a reviewer can diff
    the kernel's scratch stage by stage.

    Not safe during CUDA-graph capture and not meant to be: it allocates.
    """

    def __init__(self, *, prefix_len: int = PREFIX_LEN, chunk: int = M):
        # Frozen for the object's lifetime, so derive once and name them.
        self.prefix_len = prefix_len
        self.chunk = chunk
        self.suffix = slice(prefix_len, prefix_len + chunk)
        self.q_cols = H * DH

    # -- stages ------------------------------------------------------------

    def _qkv(self, x, rms_factor, ada_scale, w_qkv, qkv_bias, rope):
        """AdaRMS-scale, project, apply the shift bias, then rotate Q and K.

        `rms_factor` is an INPUT, matching the kernel: the RMS reduction stays
        outside this launch and joins as a task kind in the full-layer scope.

        Two orderings here are load-bearing and silent if wrong: `ada_scale`
        multiplies x BEFORE the contraction (it is indexed by K, so it cannot
        ride the epilogue), and `qkv_bias` is added BEFORE the rotation.
        """
        a = x.float() * ada_scale.float()[None, :]              # (M_PAD, D)
        acc = F.linear(a, w_qkv.float().T)                      # (M_PAD, QKV_W)
        acc = acc * rms_factor.float()[:, None] + qkv_bias.float()[None, :]
        return self._rotate(acc, rope)

    def _rotate(self, acc, rope):
        """Rotate adjacent column pairs of the Q and K slices; V passes through.

        Pairs are (2p, 2p+1). OpenPI's checkpoint uses the (p, p+DH/2) split
        form; `spec.weight_shapes()` permutes the QKV weight columns offline so
        this cheap adjacent-pair form is the correct one here.
        """
        cos = rope[:, 0::2].float()                             # (M_PAD, DH/2)
        sin = rope[:, 1::2].float()
        head = acc[:, :ROPE_COLS].reshape(M_PAD, ROPE_COLS // DH, DH // 2, 2)
        even, odd = head[..., 0], head[..., 1]
        rot = torch.stack((even * cos[:, None, :] - odd * sin[:, None, :],
                           odd * cos[:, None, :] + even * sin[:, None, :]), dim=-1)
        return torch.cat((rot.reshape(M_PAD, ROPE_COLS), acc[:, ROPE_COLS:]), dim=1)

    def _attend(self, q_buf, k_cache, v_cache, key_mask):
        """Multi-query attention: H query heads against ONE key/value head.

        Bidirectional over the whole key axis -- no causal structure anywhere in
        this block. The mask is per-KEY only, so every query row sees the same
        additive bias.
        """
        q = q_buf.float().unsqueeze(0)                          # (1, H, M_PAD, DH)
        k = k_cache.float()[None, None]                         # (1, 1, KEYS_PAD, DH)
        v = v_cache.float()[None, None]
        bias = key_mask.float()[None, None, None, :]            # broadcast over rows
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                              scale=SCALE, enable_gqa=True)
        return attn[0]                                          # (H, M_PAD, DH)

    # -- the block ---------------------------------------------------------

    def forward(self, *, x, rms_factor, ada_scale, w_qkv, qkv_bias, rope,
                key_mask, w_o, ada_gate, k_cache, v_cache, out,
                q_buf=None, o_buf=None) -> dict[str, torch.Tensor]:
        """Run the block; mutate `k_cache`, `v_cache`, `out`; return the stages.

        Shapes, matching the header:
            x           (M_PAD, D)        rows [M, M_PAD) zeroed
            rms_factor  (M_PAD,)          rsqrt(mean(x^2) + eps), pad rows zero
            ada_scale   (D,)              1 + scale, per (step, layer)
            w_qkv       (D, QKV_W)
            qkv_bias    (QKV_W,)          shift @ w_qkv
            rope        (M_PAD, DH)       cos in even columns, sin in odd
            key_mask    (KEYS_PAD,)       0 on real keys, MASK_NEG on the prompt
                                          pad AND on [KEYS, KEYS_PAD)
            w_o         (H*DH, D)
            ada_gate    (D,)              g, per (step, layer)
            k_cache     (KEYS_PAD, DH)    rows [0, prefix_len) read-only
            v_cache     (KEYS_PAD, DH)
            out         (M_PAD, D)        in: residual;  out: residual + gated
            q_buf       (H, M_PAD, DH)    optional; allocated if omitted
            o_buf       (H, M_PAD, DH)    optional
        """
        if q_buf is None:
            q_buf = torch.empty((H, M_PAD, DH), dtype=x.dtype, device=x.device)
        if o_buf is None:
            o_buf = torch.empty((H, M_PAD, DH), dtype=x.dtype, device=x.device)

        # 1. projection + RoPE, split into the three slices
        qkv = self._qkv(x, rms_factor, ada_scale, w_qkv, qkv_bias, rope)
        q = qkv[:, :self.q_cols]                                # (M_PAD, H*DH)
        k = qkv[:, self.q_cols:self.q_cols + DH]                # (M_PAD, DH)
        v = qkv[:, self.q_cols + DH:]                           # (M_PAD, DH)

        # Q is HEAD-MAJOR in the buffer: it makes this store, the attention's
        # load and the output projection's k-slice read all contiguous.
        q_buf.copy_(q.reshape(M_PAD, H, DH).permute(1, 0, 2).to(q_buf.dtype))

        # 2. append this step's chunk K/V to the cache. Only the `chunk` real
        # rows are written; [KEYS, KEYS_PAD) stays padding and is masked.
        k_cache[self.suffix] = k[:self.chunk].to(k_cache.dtype)
        v_cache[self.suffix] = v[:self.chunk].to(v_cache.dtype)

        # 3. attention over the whole cache
        attn = self._attend(q_buf, k_cache, v_cache, key_mask)
        o_buf.copy_(attn.to(o_buf.dtype))

        # 4. gated output projection into the residual, in place
        flat = o_buf.permute(1, 0, 2).reshape(M_PAD, H * DH)    # head-major -> (M_PAD, H*DH)
        delta = F.linear(flat.float(), w_o.float().T)           # (M_PAD, D)
        out.copy_((out.float() + delta * ada_gate.float()[None, :]).to(out.dtype))

        return {"qkv": qkv, "q": q_buf, "k": k, "v": v, "attn": o_buf,
                "delta": delta, "out": out}


# ---------------------------------------------------------------------------
# Standalone inputs and the cross-check: `python -m ...attn_block_reference`
# ---------------------------------------------------------------------------
def make_inputs(*, n_valid: int = 903, seed: int = 0, alias_out: bool = True,
                device: str = "cpu") -> dict:
    """One layer-step of random inputs at the ABI's shapes, on `device`.

    `n_valid` is the number of real prefix rows; measured prompts put it at
    843-919, so keys [n_valid, PREFIX_LEN) are the masked prompt padding. The
    activation and rms-factor pad rows are zeroed, matching the caller contract
    and the FFN prototype's harness.

    `alias_out` makes `out` the same tensor as `x`, which is what the decoder
    pipeline does: one buffer is both the projection input and the residual.
    Pass False to exercise the unaliased case the ABI also permits.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape):
        return (torch.randn(shape, generator=gen, device=device) * 0.05).bfloat16()

    x = torch.zeros((M_PAD, D), dtype=torch.bfloat16, device=device)
    x[:M] = rand(M, D)
    rms_factor = torch.zeros((M_PAD,), dtype=torch.bfloat16, device=device)
    rms_factor[:M] = torch.rsqrt(x[:M].float().square().mean(-1) + EPS).bfloat16()

    key_mask = torch.zeros((KEYS_PAD,), dtype=torch.bfloat16, device=device)
    key_mask[n_valid:PREFIX_LEN] = MASK_NEG      # prompt padding
    key_mask[KEYS:] = MASK_NEG                   # split-alignment padding

    rope = torch.empty((M_PAD, DH), dtype=torch.bfloat16, device=device)
    inv = 1.0 / (10000.0 ** (torch.arange(0, DH, 2, device=device).float() / DH))
    phase = (torch.arange(M_PAD, device=device).float() + n_valid)[:, None] * inv[None, :]
    rope.copy_(torch.stack((phase.cos(), phase.sin()), dim=2).view(M_PAD, DH))

    return dict(
        x=x, rms_factor=rms_factor,
        ada_scale=(1.0 + torch.randn(D, generator=gen, device=device) * 0.1).bfloat16(),
        w_qkv=rand(D, QKV_W), qkv_bias=rand(QKV_W), rope=rope, key_mask=key_mask,
        w_o=rand(H * DH, D),
        ada_gate=(1.0 + torch.randn(D, generator=gen, device=device) * 0.1).bfloat16(),
        k_cache=rand(KEYS_PAD, DH), v_cache=rand(KEYS_PAD, DH),
        out=x if alias_out else rand(M_PAD, D),
        q_buf=torch.zeros((H, M_PAD, DH), dtype=torch.bfloat16, device=device),
        o_buf=torch.zeros((H, M_PAD, DH), dtype=torch.bfloat16, device=device),
    )


def _cosine(a, b) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def _cross_check(t: dict) -> dict[str, float]:
    """Against `models.pi05.reference`, the hardware-independent algorithm.

    That module runs unpadded (M rows, KEYS keys) and computes the RMS factor
    itself, so the comparison is over the real rows only -- which is also a
    demonstration of the row-independence invariant in the module docstring.
    """
    from flash_vla.models.pi05 import reference as algo

    # `models.pi05.reference.attention_block` models the pipeline's aliased
    # form, where the projection input is also the residual, so the comparison
    # requires it. The reference runs first, on copies: `forward` mutates the
    # caches, and mutates `x` too when `out` aliases it.
    if t["out"] is not t["x"]:
        raise ValueError("cross-check requires make_inputs(alias_out=True)")
    ref_out, ref_k, ref_v = algo.attention_block(
        t["x"][:M].clone(), t["ada_scale"], t["w_qkv"], t["qkv_bias"], t["rope"][:M],
        t["k_cache"][:KEYS].clone(), t["v_cache"][:KEYS].clone(),
        t["key_mask"][:KEYS], t["w_o"], t["ada_gate"], PREFIX_LEN)

    got = AttnBlockReference().forward(**t)
    return {
        "out": _cosine(ref_out, got["out"][:M]),
        "k_cache_suffix": _cosine(ref_k[PREFIX_LEN:], t["k_cache"][PREFIX_LEN:KEYS]),
        "v_cache_suffix": _cosine(ref_v[PREFIX_LEN:], t["v_cache"][PREFIX_LEN:KEYS]),
    }


if __name__ == "__main__":
    inputs = make_inputs()
    scores = _cross_check(inputs)
    print(f"M={M} M_PAD={M_PAD} D={D} H={H} DH={DH} QKV_W={QKV_W} "
          f"PREFIX_LEN={PREFIX_LEN} KEYS={KEYS} KEYS_PAD={KEYS_PAD}\n")
    for name, tensor in (("q_buf", inputs["q_buf"]), ("o_buf", inputs["o_buf"]),
                         ("k_cache", inputs["k_cache"]), ("out", inputs["out"])):
        print(f"  {name:<10}{str(tuple(tensor.shape)):<18}{tensor.dtype}")
    print("\nagainst models.pi05.reference (real rows only):")
    for name, value in scores.items():
        print(f"  {name:<16}cosine {value:.7f}  {'PASS' if value > 0.999 else 'FAIL'}")
