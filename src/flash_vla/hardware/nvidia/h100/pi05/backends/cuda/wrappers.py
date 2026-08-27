"""Pipeline call sites for the fixed-shape persistent Pi0.5 FFN.

The two adjacent FFN calls preserve the pipeline signatures. The first resets
readiness state, produces contiguous K-major XFS, and arms the persistent
launch. The second consumes that state immediately. Weight packing is cached
during warmup and is absent from CUDA graph replay.
"""

from __future__ import annotations

import math
import os

import torch

from ..tilelang import wrappers as _tilelang
from . import taskloop as _ffn

M = 50


class _FFNState:
    """Persistent library, fixed schedule, and owned scratch."""

    def __init__(self, device: torch.device):
        self.kernel = _ffn.FFNTaskloop(
            verbose=bool(os.environ.get("FLASH_VLA_BUILD_VERBOSE")))
        self.table = _ffn.build_table("full").to(device)
        self.xfs = torch.empty(
            (_ffn.D, _ffn.M_PAD), dtype=torch.bfloat16, device=device)
        self.counters = torch.empty(
            (_ffn.N_COUNTERS,), dtype=torch.int32, device=device)
        self.legacy_factor = torch.empty(
            (M,), dtype=torch.bfloat16, device=device)
        self.packed: dict[tuple[int, ...], tuple[tuple[torch.Tensor, ...],
                                                 torch.Tensor]] = {}
        self.pending: tuple | None = None


_STATES: dict[tuple[str, int | None], _FFNState] = {}


def _state(device: torch.device) -> _FFNState:
    key = (device.type, device.index)
    if key not in _STATES:
        _STATES[key] = _FFNState(device)
    return _STATES[key]


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


def decoder_norm_gated_ffn(
        x, scale, gate_w, up_w, gate_b, up_b, out, norm_factor):
    """Reset readiness state, produce XFS, and arm the persistent FFN."""
    if tuple(x.shape) != (M, _ffn.D) or tuple(out.shape) != (M, _ffn.FF):
        raise ValueError(
            f"CUDA FFN requires x[{M},{_ffn.D}] and hidden[{M},{_ffn.FF}]")
    state = _state(x.device)
    if state.pending is not None:
        raise RuntimeError(
            "decoder_norm_gated_ffn armed twice without down_residual")
    packed_gate_up = _packed(
        state, (gate_w, up_w), lambda: _pack_gate_up(gate_w, up_w))
    hidden_ready, down_ready = state.kernel.readiness_counter_buffers(
        state.counters)
    _tilelang.decoder_rms_xfs(
        x, scale, hidden_ready, down_ready, state.xfs,
        trigger_programmatic_launch=True)
    state.pending = (out.data_ptr(), scale, gate_b, up_b, packed_gate_up)


def decoder_ffn_down_residual(x, weight, gate, out):
    """Launch the persistent FFN armed by norm/gated-FFN."""
    state = _state(x.device)
    if state.pending is None or state.pending[0] != x.data_ptr():
        raise RuntimeError(
            "CUDA down_residual requires adjacent CUDA norm_gated_ffn")
    _, scale, gate_b, up_b, packed_gate_up = state.pending
    state.pending = None
    packed_down = _packed(state, (weight,), lambda: _pack_down(weight))
    hidden_pad = _padded_base(x, (_ffn.M_PAD, _ffn.FF))
    out_pad = _padded_base(out, (_ffn.M_PAD, _ffn.D))
    state.kernel.launch(
        state.table, state.xfs, state.legacy_factor, scale,
        packed_gate_up, packed_gate_up, gate_b, up_b, packed_down, gate,
        hidden_pad, out_pad, state.counters,
        zero_counters=False, use_programmatic_dependency=True)
    return out


ALL_WRAPPERS = {
    "decoder_norm_gated_ffn": decoder_norm_gated_ffn,
    "decoder_ffn_down_residual": decoder_ffn_down_residual,
}
FUSED_WRAPPERS: dict = {}

__all__ = [
    "ALL_WRAPPERS", "FUSED_WRAPPERS", "decoder_norm_gated_ffn",
    "decoder_ffn_down_residual",
]
