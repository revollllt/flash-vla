"""Pi0.5 on one RTX 5090: native CUTLASS, Triton and CUDA fusions, and MXFP8.

Identity is what makes this a separate Target rather than a flag. A
measurement taken here must not be recorded against `h100-sxm5-80gb`: the two
machines differ by 1.8x on streaming bandwidth, 3.4x on tensor-core throughput
and 2.29x on shared memory per block, so a latency compared across them is not
a comparison at all.

The model is the same `flash_vla.models.pi05` as on H100. This Target lays it
out with the backbone's dense call sites taking the prefix mask, which the
bucketed backbone GEMMs use to skip the padded prompt rows; plans saved with
the standard names still bind (`call_site_aliases`).
"""
from __future__ import annotations

from flash_vla.models.pi05.definition import Pi05Layout, Pi05Model
from flash_vla.models.pi05.ops import MASKED_CALL_SITES
from flash_vla.runtime.cost import Pricing
from flash_vla.runtime.vla import QuantizationRecipe, Target

from .backends import REGISTRY

#: Bytes per MXFP8 element: one E4M3 value and a UE8M0 scale per 32.
MXFP8_ITEMSIZE = 1 + 1 / 32

TARGET = Target(
    name="hardware/nvidia/rtx5090/pi05",
    hardware="rtx5090-32gb",
    model=Pi05Model(Pi05Layout(row_pad=64, masked_backbone=True)),
    registry=REGISTRY,
    # Native pointwise fusion around the existing bf16 GEMMs.
    plan={
        "vision_encoder_norm_qkv": "cutlass-vision",
        "vision_encoder_out_proj_residual": "cutlass-vision",
        "vision_encoder_norm_ffn_up": "cutlass-vision",
        "vision_encoder_ffn_down_residual": "cutlass-vision",
        "action_expert_norm_gated_ffn": "dual-ffn",
        "action_expert_norm_qkv_rope": "triton-qkv-finish",
        "llm_backbone_norm_qkv_rope": "fused-prefix-qkv",
        "llm_backbone_out_proj_residual_masked": "bucketed-backbone",
        "llm_backbone_norm_gated_ffn_masked": "bucketed-backbone",
        "llm_backbone_ffn_down_residual_masked": "bucketed-backbone",
        "action_expert_attention": "triton-qk-attention",
        "action_expert_out_proj_residual": "cutlass-expert-residual",
        "action_expert_ffn_down_residual": "cutlass-expert-residual",
        "action_expert_action_out_proj": "fused-qkv",
    },
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
            plan={"llm_backbone_norm_gated_ffn_masked": "mxfp8-backbone",
                  "llm_backbone_ffn_down_residual_masked": "mxfp8-backbone"},
            reference_plan={"llm_backbone_norm_gated_ffn_masked": "fake-quant-mxfp8",
                            "llm_backbone_ffn_down_residual_masked": "fake-quant-mxfp8"},
            backends=frozenset({"mxfp8-backbone", "fake-quant-mxfp8"}),
            # The hidden is the gated call site's output and the down GEMM's input.
            pricing={"llm_backbone_norm_gated_ffn_masked": Pricing("mxfp8", {
                         "gate_w": MXFP8_ITEMSIZE, "up_w": MXFP8_ITEMSIZE, "out": MXFP8_ITEMSIZE}),
                     "llm_backbone_ffn_down_residual_masked": Pricing("mxfp8", {
                         "x": MXFP8_ITEMSIZE, "weight": MXFP8_ITEMSIZE})}),
    },
    # Plans saved against the standard backbone names still bind.
    call_site_aliases=MASKED_CALL_SITES,
    # H100's `action_expert_norm_gated_ffn` ceiling is a measured H100 number and
    # says nothing about this part; the floor model falls back to this machine's
    # own constants until a `tma_ring`-equivalent sweep has been run here.
    ceilings={},
)

__all__ = ["MXFP8_ITEMSIZE", "TARGET"]
