"""The call sites Pi0.5's graph emits, and its ops beyond the standard vocabulary.

The Pi0.5 graph emits every standard call site of `runtime/ops.py`, except
that a masked layout (`Pi05Layout.masked_backbone`) emits the backbone's three
dense call sites with the prefix mask as a last argument instead, so that a
device's kernels can skip the padded prompt rows the mask marks; those masked
variants are this model's extension ops.
"""
from __future__ import annotations

from flash_vla.runtime.ops import STANDARD, OpSpec, dual_gemm, gemm

#: The standard call sites: every one the unmasked Pi0.5 graph emits.
CALL_SITES = frozenset(spec.name for spec in STANDARD)

#: The backbone call sites a masked layout emits with the prefix mask appended,
#: standard name -> masked name.
MASKED_CALL_SITES = {
    "llm_backbone_norm_gated_ffn": "llm_backbone_norm_gated_ffn_masked",
    "llm_backbone_ffn_down_residual": "llm_backbone_ffn_down_residual_masked",
    "llm_backbone_out_proj_residual": "llm_backbone_out_proj_residual_masked",
}

#: The masked variants: each standard spec with `mask` (the prefix rows'
#: additive key mask, zero on valid rows) as its last parameter.
MASKED_OPS = (
    OpSpec("llm_backbone_norm_gated_ffn_masked",
           ("x", "gate_w", "up_w", "out", "x_norm", "mask"),
           outputs=("out", "x_norm"), weights=("gate_w", "up_w"), aux=("x_norm",),
           flops=dual_gemm("x", "gate_w")),
    OpSpec("llm_backbone_ffn_down_residual_masked",
           ("x", "weight", "out", "mask"), outputs=("out",), inout=("out",),
           weights=("weight",), flops=gemm("x", "weight")),
    OpSpec("llm_backbone_out_proj_residual_masked",
           ("x", "weight", "out", "mask"), outputs=("out",), inout=("out",),
           weights=("weight",), flops=gemm("x", "weight")),
)

__all__ = ["CALL_SITES", "MASKED_CALL_SITES", "MASKED_OPS"]
