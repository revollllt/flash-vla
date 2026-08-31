"""Hand-written CUDA call sites for the Pi0.5 decoder attention and FFN halves.

Four call sites, with the TileLang wrappers' signatures, so a plan can move
them one pair at a time:

    {"decoder_norm_qkv_rope": "cuda", "decoder_attention": "cuda",
     "decoder_norm_gated_ffn": "cuda", "decoder_ffn_down_residual": "cuda"}

The attention pair runs the standalone kernels of `kernels/attn_taskloop.cu`
(the per-op form of the task-loop bodies,
`specs/tile/attention_block_contract.md`). The FFN pair runs the persistent
132-CTA task loop of `kernels/ffn_taskloop.cu` (`taskloop.FFNTaskloop`):
`decoder_norm_gated_ffn` resets the readiness counters and launches the
K-major XFS producer with a programmatic-dependent-launch trigger, and
`decoder_ffn_down_residual` immediately launches the persistent consumer with
`use_programmatic_dependency=True`. The two calls are adjacent in the
pipeline, which is exactly the direct-predecessor contract PDL requires; the
counter reset sits *before* the XFS producer so the producer stays the
persistent kernel's immediate predecessor on the stream.
The contract's tensors are the padded allocations behind the pipeline's
views (`buffers.py`): 64 query rows, 1024 cache keys. Each wrapper recovers
the padded base from the view it is handed and refuses a view that was not
cut from one.

Q crosses the two call sites head-major in implementation-owned scratch, not
through the token-major `Q` the pipeline passes, so `decoder_attention` here
requires `decoder_norm_qkv_rope` here as well; it checks. The attention
output is combined straight into the pipeline's token-major buffer, which is
what the TileLang `decoder_out_proj_residual` reads.

Compiled for one configuration: three views, a 200-token prompt, a chunk of
50 (`PREFIX_LEN` in `kernels/sm90_attn_task_desc.cuh`).
"""

from __future__ import annotations

import math
import os

import torch

from ..tilelang import wrappers as _tilelang
from . import attn_taskloop as _at
from . import taskloop as _ffn


class _FFNState:
    """The persistent-FFN library, its fixed schedule, and implementation-owned scratch."""

    def __init__(self, device: torch.device):
        self.kernel = _ffn.FFNTaskloop(verbose=bool(os.environ.get("FLASH_VLA_BUILD_VERBOSE")))
        self.table = _ffn.build_table("full").to(device)
        self.xfs = torch.empty((_ffn.D, _ffn.M_PAD), dtype=torch.bfloat16, device=device)
        self.counters = torch.empty((_ffn.N_COUNTERS,), dtype=torch.int32, device=device)
        # Legacy ABI slot (`F`); the kernel never reads it.
        self.legacy_factor = torch.empty((_at.M,), dtype=torch.bfloat16, device=device)
        # key -> (source weights, packed copy). Holding the sources keeps their
        # allocations alive, so a data_ptr key cannot be reused by a later
        # engine's weights while a stale packed copy is still cached.
        self.packed: dict[int, tuple[tuple[torch.Tensor, ...], torch.Tensor]] = {}
        # Armed by `decoder_norm_gated_ffn`, consumed by `decoder_ffn_down_residual`.
        self.pending: tuple | None = None


class _Runtime:
    """The built library and its scratch, one per device, built on first use."""

    def __init__(self, device: torch.device):
        self.kernel = _at.AttnTaskloop(verbose=bool(os.environ.get("FLASH_VLA_BUILD_VERBOSE")))
        self.workspace = _at.Workspace(device)
        # data_ptr of the pipeline's Q the last qkv call was asked to fill;
        # the head-major Q it really filled lives in `workspace.q_buf`.
        self.q_owner: int | None = None
        # Built on the first FFN call, so attention-only plans never compile it.
        self.ffn: _FFNState | None = None


_RUNTIMES: dict[tuple[str, int | None], _Runtime] = {}


def _runtime(device: torch.device) -> _Runtime:
    key = (device.type, device.index)
    if key not in _RUNTIMES:
        _RUNTIMES[key] = _Runtime(device)
    return _RUNTIMES[key]


def _padded_base(view: torch.Tensor, shape: tuple[int, ...], first_row: int = 0) -> torch.Tensor:
    """The contiguous `shape` allocation `view` is rows `[first_row, ...)` of."""
    strides = tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))
    if view.dim() != len(shape) or tuple(view.shape[1:]) != tuple(shape[1:]) \
            or tuple(view.stride()) != strides:
        raise ValueError(f"expected a leading-row view of a contiguous {shape} allocation, "
                         f"got shape {tuple(view.shape)} strides {tuple(view.stride())}")
    offset = view.storage_offset() - first_row * strides[0]
    capacity = view.untyped_storage().nbytes() // view.element_size()
    if offset < 0 or offset + math.prod(shape) > capacity:
        raise ValueError(f"view is not backed by a padded {shape} allocation "
                         f"(rows {first_row}..; see buffers.allocate_static_buffers)")
    return view.as_strided(shape, strides, offset)


def decoder_norm_qkv_rope(x, scale, weight_qkv, bias, rope, Q, K, V, norm_factor):
    """AdaRMS-scale x, project to QKV, add the shift bias, apply RoPE, scatter in place.

    `Q` is not written: the head-major Q lands in this backend's scratch for
    its own `decoder_attention`. K/V are written into the cache suffix rows.
    """
    if _at.QKV_WEIGHT_TRANSPOSED:
        raise NotImplementedError("ATTN_QKV_WT builds take W_qkv transposed; not a pipeline layout")
    M = x.shape[0]
    if M != _at.M or K.shape[0] != M or V.shape[0] != M:
        raise ValueError(f"cuda backend is compiled for M={_at.M}; got x rows {M}, K rows {K.shape[0]}")
    rt = _runtime(x.device)
    x_pad = _padded_base(x, (_at.M_PAD, _at.D))
    rope_pad = _padded_base(rope, (_at.M_PAD, _at.DH))
    factor_pad = _padded_base(norm_factor, (_at.M_PAD,))
    k_cache = _padded_base(K, (_at.KEYS_PAD, _at.DH), first_row=_at.PREFIX_LEN)
    v_cache = _padded_base(V, (_at.KEYS_PAD, _at.DH), first_row=_at.PREFIX_LEN)

    # Same rms_factor node as the TileLang call site (pad rows stay zero).
    _tilelang._rms_factor(x, norm_factor[:M])
    for op in _at.STANDALONE_OP_GROUPS["qkv"]:
        _at.launch_standalone(rt.kernel._lib, op, rt.workspace,
                              x=x_pad, rms_factor=factor_pad, ada_scale=scale, w_qkv=weight_qkv,
                              qkv_bias=bias, rope=rope_pad, key_mask=None, w_o=None, ada_gate=None,
                              k_cache=k_cache, v_cache=v_cache, out=None)
    rt.q_owner = Q.data_ptr()


def decoder_attention(Q, K, V, mask, out):
    """Split-K flash decoding over the padded cache, combined token-major into `out`.

    `Q` must be the tensor this backend's `decoder_norm_qkv_rope` was asked to
    fill (its real Q is head-major scratch). `out` may alias `Q`.
    """
    rt = _runtime(Q.device)
    if rt.q_owner != Q.data_ptr():
        raise RuntimeError("cuda decoder_attention needs cuda decoder_norm_qkv_rope on the same "
                           "plan: Q is head-major scratch, not the pipeline's buffer")
    if mask.shape[0] != _at.KEYS or K.shape[0] != _at.KEYS:
        raise ValueError(f"cuda backend is compiled for {_at.KEYS} keys; got {mask.shape[0]}")
    k_cache = _padded_base(K, (_at.KEYS_PAD, _at.DH))
    v_cache = _padded_base(V, (_at.KEYS_PAD, _at.DH))
    mask_pad = _padded_base(mask, (_at.KEYS_PAD,))
    common = dict(x=None, rms_factor=None, ada_scale=None, w_qkv=None, qkv_bias=None, rope=None,
                  key_mask=mask_pad, w_o=None, ada_gate=None, k_cache=k_cache, v_cache=v_cache)
    split_op, _combine_head_major = _at.STANDALONE_OP_GROUPS["attention"]
    _at.launch_standalone(rt.kernel._lib, split_op, rt.workspace, out=None, **common)
    _at.launch_standalone(rt.kernel._lib, _at.OP_ATTN_COMBINE_TOK, rt.workspace, out=out, **common)
    return out


def _pack_gate_up(gate_w: torch.Tensor, up_w: torch.Tensor) -> torch.Tensor:
    """One `[W_gate_tile | W_up_tile]` row pair per K tile; `taskloop` docstring."""
    def tiles(weight: torch.Tensor) -> torch.Tensor:
        return (weight.reshape(_ffn.D, _ffn.FF // _ffn.BN, _ffn.BN)
                .permute(1, 0, 2).contiguous().view(-1, _ffn.BN))
    return torch.cat((tiles(gate_w), tiles(up_w)), dim=1).contiguous()


def _pack_down(weight: torch.Tensor) -> torch.Tensor:
    return (weight.reshape(_ffn.FF, _ffn.D // _ffn.BN, _ffn.BN)
            .permute(1, 0, 2).contiguous().view(_ffn.FF, _ffn.D))


def _packed(state: _FFNState, key: int, sources: tuple[torch.Tensor, ...], pack):
    """The cached packed form of `sources`, packing on first sight.

    Packing allocates, so it must stay out of graph capture: every (step,
    layer) call site runs in the engine's warmup passes first, and capture
    then only replays cache hits.
    """
    entry = state.packed.get(key)
    if entry is None:
        entry = (sources, pack())
        state.packed[key] = entry
    return entry[1]


def decoder_norm_gated_ffn(x, scale, gate_w, up_w, gate_b, up_b, out, norm_factor):
    """Arm the persistent FFN: reset readiness counters, then produce XFS with a PDL trigger.

    Writes neither `out` nor `norm_factor`: the persistent kernel launched by
    the adjacent `decoder_ffn_down_residual` fills the padded hidden buffer
    behind `out`, and the row factor exists only inside the XFS producer.
    """
    if tuple(x.shape) != (_at.M, _ffn.D) or tuple(out.shape) != (_at.M, _ffn.FF):
        raise ValueError(f"cuda FFN is compiled for x[{_at.M},{_ffn.D}], "
                         f"hidden[{_at.M},{_ffn.FF}]; got {tuple(x.shape)}, {tuple(out.shape)}")
    rt = _runtime(x.device)
    if rt.ffn is None:
        rt.ffn = _FFNState(x.device)
    st = rt.ffn
    if st.pending is not None:
        raise RuntimeError("cuda decoder_norm_gated_ffn armed twice without a cuda "
                           "decoder_ffn_down_residual between: route both call sites here")
    packed_gate_up = _packed(st, gate_w.data_ptr(), (gate_w, up_w),
                             lambda: _pack_gate_up(gate_w, up_w))
    # Reset strictly before the producer: the producer must remain the
    # persistent kernel's direct stream predecessor or the PDL overlap is lost.
    st.kernel.reset_counters(st.counters)
    _tilelang.decoder_rms_xfs(x, scale, st.xfs, trigger_programmatic_launch=True)
    st.pending = (out.data_ptr(), scale, gate_b, up_b, packed_gate_up)


def decoder_ffn_down_residual(x, weight, gate, out):
    """Launch the persistent GatedUp+DownResidual task loop armed by `decoder_norm_gated_ffn`.

    `x` is the 50-row hidden view and `out` the 50-row `decoder_x` view; the
    kernel runs on the padded 64-row allocations behind both (`buffers.py`).
    """
    rt = _runtime(x.device)
    st = rt.ffn
    if st is None or st.pending is None or st.pending[0] != x.data_ptr():
        raise RuntimeError("cuda decoder_ffn_down_residual needs cuda decoder_norm_gated_ffn "
                           "immediately before it on the same plan: the XFS producer must be "
                           "the persistent kernel's direct stream predecessor")
    _hidden_owner, scale, gate_b, up_b, packed_gate_up = st.pending
    st.pending = None
    packed_down = _packed(st, weight.data_ptr(), (weight,), lambda: _pack_down(weight))
    hidden_pad = _padded_base(x, (_ffn.M_PAD, _ffn.FF))
    out_pad = _padded_base(out, (_ffn.M_PAD, _ffn.D))
    st.kernel.launch(
        st.table, st.xfs, st.legacy_factor, scale, packed_gate_up, packed_gate_up,
        gate_b, up_b, packed_down, gate, hidden_pad, out_pad, st.counters,
        zero_counters=False, use_programmatic_dependency=True)
    return out


ALL_WRAPPERS = {
    "decoder_norm_qkv_rope": decoder_norm_qkv_rope,
    "decoder_attention": decoder_attention,
    "decoder_norm_gated_ffn": decoder_norm_gated_ffn,
    "decoder_ffn_down_residual": decoder_ffn_down_residual,
}
FUSED_WRAPPERS: dict = {}

__all__ = ["ALL_WRAPPERS", "FUSED_WRAPPERS", "decoder_norm_qkv_rope", "decoder_attention",
           "decoder_norm_gated_ffn", "decoder_ffn_down_residual"]
