"""Pipeline call sites for the fixed-shape persistent Pi0.5 FFN.

The two adjacent FFN calls preserve the pipeline signatures. The first resets
readiness state, produces contiguous K-major XFS, and arms the persistent
launch. The second consumes that state immediately. Weight packing is cached
during warmup and is absent from CUDA graph replay.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

from ..tilelang import wrappers as _tilelang
from . import taskloop as _ffn

M = 50


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
        self.rstd_per_cta = torch.empty(
            (4, 32, 16), dtype=torch.bfloat16, device=device)
        self.packed: dict[tuple[int, ...], tuple[tuple[torch.Tensor, ...],
                                                 torch.Tensor]] = {}
        self.pending_out_proj: _PendingOutProj | None = None
        self.pending: _PendingFFN | None = None


@dataclass(frozen=True)
class _PendingOutProj:
    attention: torch.Tensor
    weight: torch.Tensor
    gate: torch.Tensor
    residual_ptr: int
    stream: int


@dataclass(frozen=True)
class _PendingFFN:
    hidden_ptr: int
    stream: int
    use_programmatic_dependency: bool
    scale: torch.Tensor
    gate_bias: torch.Tensor
    up_bias: torch.Tensor
    packed_gate_up: torch.Tensor


def _padded_base(view: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Recover the contiguous padded allocation backing a leading-row view."""
    strides = tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))
    if (view.dim() != len(shape)
            or tuple(view.shape[1:]) != tuple(shape[1:])
            or tuple(view.stride()) != strides):
        raise ValueError(
            f"expected leading-row view of contiguous {shape}, got "
            f"shape={tuple(view.shape)} strides={tuple(view.stride())}")
    offset = view.storage_offset()
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


WRAPPER_NAMES = frozenset({
    "decoder_out_proj_residual",
    "decoder_norm_gated_ffn",
    "decoder_ffn_down_residual",
})
FUSED_WRAPPERS: dict = {}


def make_wrappers(
        selected_names: set[str] | None = None) -> dict[str, object]:
    """Create one FFN runtime whose lifetime follows its owning op table."""
    selected = set(WRAPPER_NAMES if selected_names is None else selected_names)
    fuse_out_proj = "decoder_out_proj_residual" in selected
    state: _FFNState | None = None

    def state_for(device: torch.device) -> _FFNState:
        nonlocal state
        if state is None:
            state = _FFNState(device)
        elif state.device != device:
            raise ValueError(
                f"one CUDA FFN op table cannot span {state.device} and {device}")
        return state

    def current_stream(device: torch.device) -> int:
        return torch.cuda.current_stream(device).cuda_stream

    def decoder_out_proj_residual(attention, weight, gate, out):
        """Defer the out projection to the adjacent cooperative XFS producer."""
        runtime = state_for(out.device)
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
        runtime.pending_out_proj = _PendingOutProj(
            attention=attention, weight=weight, gate=gate,
            residual_ptr=out.data_ptr(), stream=current_stream(out.device))
        return out

    def decoder_norm_gated_ffn(
            x, scale, gate_w, up_w, gate_b, up_b, out, norm_factor):
        """Reset readiness state, produce XFS, and arm the persistent FFN."""
        if tuple(x.shape) != (M, _ffn.D) or tuple(out.shape) != (M, _ffn.FF):
            raise ValueError(
                f"CUDA FFN requires x[{M},{_ffn.D}] and hidden[{M},{_ffn.FF}]")
        runtime = state_for(x.device)
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
            if (out_proj is None or out_proj.residual_ptr != x.data_ptr()
                    or out_proj.stream != stream):
                raise RuntimeError(
                    "fused CUDA norm_gated_ffn requires adjacent out projection "
                    "on the same residual and stream")
            _tilelang.decoder_out_proj_residual_rms_xfs(
                out_proj.attention, out_proj.weight, out_proj.gate, x, scale,
                hidden_ready, down_ready, runtime.square_partials,
                runtime.rstd_per_cta, runtime.xfs)
            runtime.pending_out_proj = None
        else:
            if runtime.pending_out_proj is not None:
                raise RuntimeError("unexpected deferred out-projection state")
            _tilelang.decoder_rms_xfs(
                x, scale, hidden_ready, down_ready, runtime.xfs,
                trigger_programmatic_launch=True)
        runtime.pending = _PendingFFN(
            hidden_ptr=out.data_ptr(), stream=stream,
            use_programmatic_dependency=not fuse_out_proj,
            scale=scale, gate_bias=gate_b,
            up_bias=up_b, packed_gate_up=packed_gate_up)

    def decoder_ffn_down_residual(x, weight, gate, out):
        """Launch the persistent FFN armed by norm/gated-FFN."""
        runtime = state_for(x.device)
        pending = runtime.pending
        if (pending is None or pending.hidden_ptr != x.data_ptr()
                or pending.stream != current_stream(x.device)):
            raise RuntimeError(
                "CUDA down_residual requires adjacent CUDA norm_gated_ffn "
                "on the same stream")
        runtime.pending = None
        packed_down = _packed(runtime, (weight,), lambda: _pack_down(weight))
        hidden_pad = _padded_base(x, (_ffn.M_PAD, _ffn.FF))
        out_pad = _padded_base(out, (_ffn.M_PAD, _ffn.D))
        runtime.kernel.launch(
            runtime.table, runtime.xfs, runtime.legacy_factor, pending.scale,
            pending.packed_gate_up, pending.packed_gate_up,
            pending.gate_bias, pending.up_bias, packed_down, gate,
            hidden_pad, out_pad, runtime.counters,
            zero_counters=False,
            use_programmatic_dependency=pending.use_programmatic_dependency)
        return out

    return {
        "decoder_out_proj_residual": decoder_out_proj_residual,
        "decoder_norm_gated_ffn": decoder_norm_gated_ffn,
        "decoder_ffn_down_residual": decoder_ffn_down_residual,
    }


__all__ = [
    "WRAPPER_NAMES", "FUSED_WRAPPERS", "make_wrappers",
]
