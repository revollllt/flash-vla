"""The call sites Pi0's graph emits, and its two ops beyond the standard vocabulary."""
from __future__ import annotations

from flash_vla.runtime.ops import STANDARD, OpSpec, gemm

#: The state token's projection and the second action MLP, both plain bias GEMMs.
OPS = (
    OpSpec("action_expert_state_proj", ("x", "weight", "bias", "out"), outputs=("out",),
           weights=("weight", "bias"), flops=gemm("x", "weight")),
    OpSpec("action_expert_action_mlp", ("x", "weight", "bias", "out"), outputs=("out",),
           weights=("weight", "bias"), flops=gemm("x", "weight")),
)

#: Every call site the Pi0 graph emits: the whole standard vocabulary and `OPS`.
CALL_SITES = frozenset(spec.name for spec in STANDARD) | frozenset(spec.name for spec in OPS)

__all__ = ["CALL_SITES", "OPS"]
