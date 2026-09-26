"""Pi0.5 on one RTX 5090: native CUTLASS, Triton and CUDA fusions, and MXFP8.

Identity is what makes this a separate Target rather than a flag. A
measurement taken here must not be recorded against `h100-sxm5-80gb`: the two
machines differ by 1.8x on streaming bandwidth, 3.4x on tensor-core throughput
and 2.29x on shared memory per block, so a latency compared across them is not
a comparison at all.

The model is the same `flash_vla.models.pi05` as on H100. This Target captures
the backbone once per 64-row bucket of the prompt a workload's observations
reach (`replay_granularity`, `runtime/replay.py`), so
every GEMM plans for the rows its bucket holds and none runs the padding past
it. 64 rows is the finer of the CUTLASS tiles the BF16 backbone GEMMs pick
(`lab/pi05/bucket_granularity_screen.py`).
"""
from __future__ import annotations

from flash_vla.models.pi05.definition import Pi05Layout, Pi05Model
from flash_vla.runtime.vla import Pricing, QuantizationRecipe, Target

from .backends import REGISTRY

#: Bytes per MXFP8 element: one E4M3 value and a UE8M0 scale per 32.
MXFP8_ITEMSIZE = 1 + 1 / 32
#: The masked backbone call sites plans and reports saved before the replay
#: buckets name, and the standard sites that replaced them. A saved plan that
#: routes them to the removed `bucketed-backbone` still fails, by name.
LEGACY_MASKED_CALL_SITES = {
    "llm_backbone_norm_gated_ffn_masked": "llm_backbone_norm_gated_ffn",
    "llm_backbone_ffn_down_residual_masked": "llm_backbone_ffn_down_residual",
    "llm_backbone_out_proj_residual_masked": "llm_backbone_out_proj_residual",
}

TARGET = Target(
    name="hardware/nvidia/rtx5090/pi05",
    hardware="rtx5090-32gb",
    model=Pi05Model(Pi05Layout(row_pad=64)),
    registry=REGISTRY,
    # Native pointwise fusion around the existing bf16 GEMMs.
    plan={
        "vision_encoder_norm_qkv": ("cutlass-vision",),
        "vision_encoder_out_proj_residual": ("cutlass-vision",),
        "vision_encoder_norm_ffn_up": ("cutlass-vision",),
        "vision_encoder_ffn_down_residual": ("cutlass-vision",),
        "action_expert_norm_gated_ffn": ("dual-ffn",),
        "action_expert_norm_qkv_rope": ("triton-qkv-finish",),
        "llm_backbone_norm_qkv_rope": ("fused-prefix-qkv",),
        "llm_backbone_out_proj_residual": ("cutlass-backbone",),
        "llm_backbone_norm_gated_ffn": ("cutlass-backbone",),
        "llm_backbone_ffn_down_residual": ("cutlass-backbone",),
        "action_expert_attention": ("triton-qk-attention",),
        "action_expert_out_proj_residual": ("cutlass-expert-residual",),
        "action_expert_ffn_down_residual": ("cutlass-expert-residual",),
        "action_expert_action_out_proj": ("fused-qkv",),
    },
    workloads=("robodojo", "libero"),
    replay_granularity=64,
    reference_plan={},
    quantization={
        # The backbone FFN's three GEMMs in MXFP8 on every layer, approved on
        # LIBERO observations (results/quant-pi05-ffn-libero). The activation is
        # quantized from the BF16 value the BF16 route rounds to, and every other
        # rounding point is kept.
        "mxfp8-llm-ffn": QuantizationRecipe(
            spec={"mode": "mxfp8", "recipe": "mxfp8-llm-ffn-v1",
                  "weight": "e4m3, ue8m0 scale per 32 along K, quantized once from bf16",
                  "activation": "e4m3, ue8m0 scale per 32 along K, dynamic, from the bf16 "
                                "rms-norm output and the bf16-rounded gelu_tanh(gate)*up",
                  "rounding": "fp32 accumulation; gate and up round to bf16; down joins "
                              "the residual in fp32 and rounds once",
                  "quantizer": "flashinfer 0.7.0 fast path (quant_ops)"},
            plan={"llm_backbone_norm_gated_ffn": ("mxfp8-backbone",),
                  "llm_backbone_ffn_down_residual": ("mxfp8-backbone",)},
            reference_plan={"llm_backbone_norm_gated_ffn": ("fake-quant-mxfp8",),
                            "llm_backbone_ffn_down_residual": ("fake-quant-mxfp8",)},
            backends=frozenset({"mxfp8-backbone", "fake-quant-mxfp8"}),
            # Both call sites read MXFP8 weights and pass each other the MXFP8
            # hidden; everything else they read and write stays BF16.
            pricing=Pricing(call_sites=frozenset({"llm_backbone_norm_gated_ffn",
                                                  "llm_backbone_ffn_down_residual"}),
                            tensor="mxfp8", weight_itemsize=MXFP8_ITEMSIZE,
                            activation_itemsize=MXFP8_ITEMSIZE)),
    },
    # Plans saved against the masked backbone names still bind.
    call_site_aliases=LEGACY_MASKED_CALL_SITES,
    # H100's `action_expert_norm_gated_ffn` ceiling is a measured H100 number and
    # says nothing about this part; the floor model falls back to this machine's
    # own constants until a `tma_ring`-equivalent sweep has been run here.
    ceilings={},
)

__all__ = ["LEGACY_MASKED_CALL_SITES", "MXFP8_ITEMSIZE", "TARGET"]
