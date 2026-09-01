"""Hand-written CUDA call sites for the Pi0.5 decoder attention and FFN halves.

Five call sites carrying the TileLang wrappers' signatures, so a plan can move
the layer one half at a time:

    {"decoder_norm_qkv_rope": "cuda", "decoder_attention": "cuda"}
    {"decoder_norm_gated_ffn": "cuda", "decoder_ffn_down_residual": "cuda"}

with "decoder_out_proj_residual" joining the second pair on the fused
three-call route.

Attention runs the standalone kernels of `kernels/attn_taskloop.cu` -- the
per-op form of the task-loop bodies, under the tensor contract of
`specs/tile/attention_block_contract.md`. The fused persistent attention block
was built, is correct, and lost;
`.agents/notes/rejected/architecture/2026-08-27-attention-block-taskloop.md`
records the measured attribution.

The FFN pair runs the persistent 132-CTA task loop of `kernels/ffn_taskloop.cu`
(`taskloop.FFNTaskloop`). `decoder_norm_gated_ffn` launches the K-major XFS
producer, which resets the readiness counters itself, with a
programmatic-dependent-launch trigger; `decoder_ffn_down_residual` immediately
launches the persistent consumer with wait mode 1 (every warp waits at entry)
or, on the `pdl_chain` variant, mode 2 (weight loaders released early). The
two calls are adjacent in the pipeline, which is the direct-predecessor
contract PDL requires. The Phase-2 route starts one call earlier and treats
out-projection, norm/gated-FFN, and down-residual as an atomic three-call
transaction.

Both halves run on the padded allocations behind the pipeline's views
(`buffers.py`): 64 query rows, 1024 cache keys. Every wrapper recovers the
padded base from the view it is handed and refuses a view not cut from one.

Q crosses the two attention call sites head-major in implementation-owned
scratch rather than through the token-major `Q` the pipeline passes, so
`decoder_attention` here requires `decoder_norm_qkv_rope` here as well:
`ops.py` rejects the split when the plan is resolved and the wrapper checks
again at call time.

Weight packing is cached during warmup and is absent from CUDA graph replay.
Compiled for one configuration: three views, a 200-token prompt, a chunk of 50
(`PREFIX_LEN` in `kernels/sm90_attn_task_desc.cuh`).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

from ..tilelang import wrappers as _tilelang
from . import attn_taskloop as _at
from . import taskloop as _ffn

#: Both halves are compiled for the same chunk; the attention header owns it.
M = _at.M


class _AttnState:
    """The attention library and its scratch, built when a plan routes here."""

    def __init__(self, device: torch.device):
        self.device = device
        self.kernel = _at.AttnTaskloop(
            verbose=bool(os.environ.get("FLASH_VLA_BUILD_VERBOSE")))
        self.workspace = _at.Workspace(device)
        # data_ptr of the pipeline Q the last qkv call was asked to fill; the
        # head-major Q it really filled lives in `workspace.q_buf`.
        self.q_owner: int | None = None


class _FFNState:
    """Persistent library, fixed schedule, and owned scratch."""

    def __init__(self, device: torch.device):
        self.device = device
        self.kernel = _ffn.FFNTaskloop(
            verbose=bool(os.environ.get("FLASH_VLA_BUILD_VERBOSE")))
        self.table = _ffn.build_table("full").to(device)
        self.xfs = torch.empty(
            (_ffn.D, _ffn.M_PAD), dtype=torch.bfloat16, device=device)
        self.counters = torch.empty(
            (_ffn.N_COUNTERS,), dtype=torch.int32, device=device)
        self.legacy_factor = torch.empty(
            (M,), dtype=torch.bfloat16, device=device)
        self.square_partials = torch.empty(
            (4, 32, 16), dtype=torch.float32, device=device)
        self.packed: dict[tuple[int, ...], tuple[tuple[torch.Tensor, ...],
                                                 torch.Tensor]] = {}
        self.pending_out_proj: _PendingOutProj | None = None
        self.pending: _PendingFFN | None = None


@dataclass(frozen=True)
class _PendingOutProj:
    attention: torch.Tensor
    weight: torch.Tensor
    gate: torch.Tensor
    residual: torch.Tensor
    stream: int


@dataclass(frozen=True)
class _PendingFFN:
    hidden_ptr: int
    stream: int
    # FFN wait mode: 1 = entry wait (shipped), 2 = role-split wait (PDL chain).
    use_programmatic_dependency: int
    scale: torch.Tensor
    gate_bias: torch.Tensor
    up_bias: torch.Tensor
    packed_gate_up: torch.Tensor


def _padded_base(view: torch.Tensor, shape: tuple[int, ...],
                 first_row: int = 0) -> torch.Tensor:
    """The contiguous `shape` allocation `view` is rows `[first_row, ...)` of."""
    strides = tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))
    if (view.dim() != len(shape)
            or tuple(view.shape[1:]) != tuple(shape[1:])
            or tuple(view.stride()) != strides):
        raise ValueError(
            f"expected leading-row view of contiguous {shape}, got "
            f"shape={tuple(view.shape)} strides={tuple(view.stride())}")
    offset = view.storage_offset() - first_row * strides[0]
    capacity = view.untyped_storage().nbytes() // view.element_size()
    if offset < 0 or offset + math.prod(shape) > capacity:
        raise ValueError(f"view is not backed by padded {shape} allocation")
    return view.as_strided(shape, strides, offset)


def _pack_gate_up(gate_weight: torch.Tensor,
                  up_weight: torch.Tensor) -> torch.Tensor:
    """Pack one gate/up row pair per output-column tile."""

    def tiles(weight: torch.Tensor) -> torch.Tensor:
        return (weight.reshape(_ffn.D, _ffn.FF // _ffn.BN, _ffn.BN)
                .permute(1, 0, 2).contiguous().view(-1, _ffn.BN))

    return torch.cat((tiles(gate_weight), tiles(up_weight)), dim=1).contiguous()


def _pack_down(weight: torch.Tensor) -> torch.Tensor:
    return (weight.reshape(_ffn.FF, _ffn.D // _ffn.BN, _ffn.BN)
            .permute(1, 0, 2).contiguous().view(_ffn.FF, _ffn.D))


def _packed(state: _FFNState, sources: tuple[torch.Tensor, ...], pack):
    key = tuple(source.data_ptr() for source in sources)
    entry = state.packed.get(key)
    if entry is None:
        entry = (sources, pack())
        state.packed[key] = entry
    return entry[1]


#: The two attention call sites are one unit: `ops.py` refuses to split them.
ATTENTION_NAMES = ("decoder_norm_qkv_rope", "decoder_attention")

WRAPPER_NAMES = frozenset({
    "decoder_norm_qkv_rope",
    "decoder_attention",
    "decoder_out_proj_residual",
    "decoder_norm_gated_ffn",
    "decoder_ffn_down_residual",
})
FUSED_WRAPPERS: dict = {}


def make_wrappers(
        selected_names: set[str] | None = None,
        pdl_chain: bool = False) -> dict[str, object]:
    """Create one CUDA runtime whose lifetime follows its owning op table.

    ``pdl_chain`` extends PDL over every boundary this backend owns: the rms
    factor kernel triggers early, qkv / attention / combine launch with the
    programmatic attribute and wait at their first dependent read, and the
    persistent FFN uses the role-split wait (mode 2) that releases its
    dependency-free weight loaders ahead of the XFS producer. Off, launch
    semantics are exactly the shipped ones.
    """
    selected = set(WRAPPER_NAMES if selected_names is None else selected_names)
    fuse_out_proj = "decoder_out_proj_residual" in selected
    ffn_wait_mode = 2 if pdl_chain else 1
    attn: _AttnState | None = None
    ffn: _FFNState | None = None

    def attn_for(device: torch.device) -> _AttnState:
        nonlocal attn
        if attn is None:
            attn = _AttnState(device)
        elif attn.device != device:
            raise ValueError(
                f"one CUDA op table cannot span {attn.device} and {device}")
        return attn

    def ffn_for(device: torch.device) -> _FFNState:
        nonlocal ffn
        if ffn is None:
            ffn = _FFNState(device)
        elif ffn.device != device:
            raise ValueError(
                f"one CUDA op table cannot span {ffn.device} and {device}")
        return ffn

    def current_stream(device: torch.device) -> int:
        return torch.cuda.current_stream(device).cuda_stream

    def decoder_norm_qkv_rope(x, scale, weight_qkv, bias, rope, Q, K, V,
                              norm_factor):
        """AdaRMS-scale x, project to QKV, add the shift, apply RoPE, scatter.

        `Q` is not written: the head-major Q this backend's `decoder_attention`
        reads lands in implementation-owned scratch. K/V are written into the
        cache suffix rows.
        """
        if _at.QKV_WEIGHT_TRANSPOSED:
            raise NotImplementedError(
                "ATTN_QKV_WT builds take W_qkv transposed; not a pipeline layout")
        if x.shape[0] != M or K.shape[0] != M or V.shape[0] != M:
            raise ValueError(
                f"CUDA attention is compiled for M={M}; got x rows {x.shape[0]}, "
                f"K rows {K.shape[0]}")
        runtime = attn_for(x.device)
        x_pad = _padded_base(x, (_at.M_PAD, _at.D))
        rope_pad = _padded_base(rope, (_at.M_PAD, _at.DH))
        factor_pad = _padded_base(norm_factor, (_at.M_PAD,))
        k_cache = _padded_base(K, (_at.KEYS_PAD, _at.DH),
                               first_row=_at.PREFIX_LEN)
        v_cache = _padded_base(V, (_at.KEYS_PAD, _at.DH),
                               first_row=_at.PREFIX_LEN)

        # Same rms_factor node as the TileLang call site; pad rows stay zero.
        # On the PDL chain it triggers at entry and qkv waits before reading
        # the factor (its weights and x are not written by this predecessor).
        _tilelang._rms_factor(x, norm_factor[:M],
                              trigger_programmatic_launch=pdl_chain)
        for op in _at.STANDALONE_OP_GROUPS["qkv"]:
            _at.launch_standalone(
                runtime.kernel._lib, op, runtime.workspace,
                x=x_pad, rms_factor=factor_pad, ada_scale=scale,
                w_qkv=weight_qkv, qkv_bias=bias, rope=rope_pad, key_mask=None,
                w_o=None, ada_gate=None, k_cache=k_cache, v_cache=v_cache,
                out=None, use_programmatic_dependency=pdl_chain)
        runtime.q_owner = Q.data_ptr()

    def decoder_attention(Q, K, V, mask, out):
        """Split-K flash decoding over the padded cache, combined token-major.

        `Q` must be the tensor this backend's `decoder_norm_qkv_rope` was asked
        to fill; the real Q is head-major scratch. `out` may alias `Q`.
        """
        runtime = attn_for(Q.device)
        if runtime.q_owner != Q.data_ptr():
            raise RuntimeError(
                "CUDA decoder_attention requires CUDA decoder_norm_qkv_rope on "
                "the same plan: Q is head-major scratch, not the pipeline buffer")
        if mask.shape[0] != _at.KEYS or K.shape[0] != _at.KEYS:
            raise ValueError(
                f"CUDA attention is compiled for {_at.KEYS} keys; "
                f"got {mask.shape[0]}")
        common = dict(
            x=None, rms_factor=None, ada_scale=None, w_qkv=None, qkv_bias=None,
            rope=None, key_mask=_padded_base(mask, (_at.KEYS_PAD,)), w_o=None,
            ada_gate=None,
            k_cache=_padded_base(K, (_at.KEYS_PAD, _at.DH)),
            v_cache=_padded_base(V, (_at.KEYS_PAD, _at.DH)))
        split_op, _combine_head_major = _at.STANDALONE_OP_GROUPS["attention"]
        _at.launch_standalone(runtime.kernel._lib, split_op, runtime.workspace,
                              out=None, use_programmatic_dependency=pdl_chain,
                              **common)
        _at.launch_standalone(runtime.kernel._lib, _at.OP_ATTN_COMBINE_TOK,
                              runtime.workspace, out=out,
                              use_programmatic_dependency=pdl_chain, **common)
        return out

    def decoder_out_proj_residual(attention, weight, gate, out):
        """Defer out projection to the adjacent residual/partial producer."""
        runtime = ffn_for(out.device)
        if not fuse_out_proj:
            raise RuntimeError(
                "CUDA out_proj_residual requires the fused three-op route")
        if runtime.pending_out_proj is not None or runtime.pending is not None:
            raise RuntimeError(
                "decoder_out_proj_residual started with unfinished FFN state")
        if (tuple(attention.shape) != (M, 2048)
                or tuple(weight.shape) != (2048, _ffn.D)
                or tuple(gate.shape) != (_ffn.D,)
                or tuple(out.shape) != (M, _ffn.D)):
            raise ValueError("invalid fixed-shape CUDA out-projection tensors")
        tensors = (attention, weight, gate, out)
        if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
            raise ValueError("CUDA out-projection tensors must be BF16")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("CUDA out-projection tensors must be contiguous")
        if any(tensor.device != out.device for tensor in tensors):
            raise ValueError("CUDA out-projection tensors must share one device")
        runtime.pending_out_proj = _PendingOutProj(
            attention=attention, weight=weight, gate=gate,
            residual=out, stream=current_stream(out.device))
        return out

    def decoder_norm_gated_ffn(
            x, scale, gate_w, up_w, gate_b, up_b, out, norm_factor):
        """Reset readiness state, produce XFS, and arm the persistent FFN."""
        if tuple(x.shape) != (M, _ffn.D) or tuple(out.shape) != (M, _ffn.FF):
            raise ValueError(
                f"CUDA FFN requires x[{M},{_ffn.D}] and hidden[{M},{_ffn.FF}]")
        runtime = ffn_for(x.device)
        if runtime.pending is not None:
            raise RuntimeError(
                "decoder_norm_gated_ffn armed twice without down_residual")
        packed_gate_up = _packed(
            runtime, (gate_w, up_w), lambda: _pack_gate_up(gate_w, up_w))
        hidden_ready, down_ready = runtime.kernel.readiness_counter_buffers(
            runtime.counters)
        stream = current_stream(x.device)
        if fuse_out_proj:
            out_proj = runtime.pending_out_proj
            if (out_proj is None
                    or out_proj.residual.data_ptr() != x.data_ptr()
                    or out_proj.stream != stream):
                raise RuntimeError(
                    "fused CUDA norm_gated_ffn requires adjacent out projection "
                    "on the same residual and stream")
            _tilelang.decoder_out_proj_residual_rms_xfs(
                out_proj.attention, out_proj.weight, out_proj.gate, x, scale,
                hidden_ready, down_ready, runtime.square_partials,
                runtime.xfs, trigger_at_entry=pdl_chain)
            runtime.pending_out_proj = None
        else:
            if runtime.pending_out_proj is not None:
                raise RuntimeError("unexpected deferred out-projection state")
            _tilelang.decoder_rms_xfs(
                x, scale, hidden_ready, down_ready, runtime.xfs,
                trigger_programmatic_launch=True)
        runtime.pending = _PendingFFN(
            hidden_ptr=out.data_ptr(), stream=stream,
            use_programmatic_dependency=ffn_wait_mode,
            scale=scale, gate_bias=gate_b,
            up_bias=up_b, packed_gate_up=packed_gate_up)

    def decoder_ffn_down_residual(x, weight, gate, out):
        """Launch the persistent FFN armed by norm/gated-FFN."""
        runtime = ffn_for(x.device)
        pending = runtime.pending
        if (pending is None or pending.hidden_ptr != x.data_ptr()
                or pending.stream != current_stream(x.device)):
            raise RuntimeError(
                "CUDA down_residual requires adjacent CUDA norm_gated_ffn "
                "on the same stream")
        packed_down = _packed(runtime, (weight,), lambda: _pack_down(weight))
        hidden_pad = _padded_base(x, (_ffn.M_PAD, _ffn.FF))
        out_pad = _padded_base(out, (_ffn.M_PAD, _ffn.D))
        try:
            runtime.kernel.launch(
                runtime.table, runtime.xfs, runtime.legacy_factor,
                pending.scale, pending.packed_gate_up,
                pending.packed_gate_up, pending.gate_bias, pending.up_bias,
                packed_down, gate, hidden_pad, out_pad, runtime.counters,
                zero_counters=False,
                use_programmatic_dependency=(
                    pending.use_programmatic_dependency))
        finally:
            runtime.pending = None
        return out

    return {
        "decoder_norm_qkv_rope": decoder_norm_qkv_rope,
        "decoder_attention": decoder_attention,
        "decoder_out_proj_residual": decoder_out_proj_residual,
        "decoder_norm_gated_ffn": decoder_norm_gated_ffn,
        "decoder_ffn_down_residual": decoder_ffn_down_residual,
    }


__all__ = [
    "ATTENTION_NAMES", "WRAPPER_NAMES", "FUSED_WRAPPERS", "make_wrappers",
]
