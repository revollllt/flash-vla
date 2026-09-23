"""Pi0 on the RTX 5090: the same graph, a different machine underneath.

Pi0's forward graph is model semantics -- three stages, no host slot, bf16 --
and it does not change with the GPU. What changes is which backend each call
site routes to, and that is what this Target owns.

**This is the bring-up route and it is deliberately slow.** Every call site runs
in plain torch (`backends/torch_ops.py`). Nothing is fused, nothing is tuned,
and the number it measures is a starting point for the optimization workflow
rather than a performance claim.

Nothing is inherited from H100/Pi0's routing, and that is the point. Its
`siglip-cuda` and `gemma-cuda` backends are hand-written sm_90a CUDA whose GEMM
mainloop is `wgmma`, which ptxas refuses on `sm_120a` [isa.wgmma.absent]. Its
TileLang tile shapes were tuned against 227 KB of shared memory; ten of nineteen
exceed this part's 99 KB [smem.bytes.cta.max] and fail at launch rather than
running slower. Re-tiling those by hand is guesswork against another machine's
tuning, and the architecture asks for a route customised per hardware -- so the
kernels get written fresh, against this machine's measured constants, in the
optimization loop.

The graph is the model's (`flash_vla.models.pi0`), shared with H100 without
importing anything of that Target; the two torch attention helpers its torch
backend uses are the model's hardware-neutral references
(`flash_vla.models.pi0.attention`).
"""
from __future__ import annotations

from flash_vla.models.pi0.definition import Pi0Model
from flash_vla.runtime.vla import Target

from .backends import REGISTRY

TARGET = Target(
    name="hardware/nvidia/rtx5090/pi0",
    hardware="rtx5090-32gb",
    model=Pi0Model(),
    registry=REGISTRY,
    # Identity is what makes this a separate Target rather than a flag: a
    # measurement taken here must not be recorded against `h100-sxm5-80gb`.
    #
    # The expert's two norm-fed call sites on hand-written pointwise kernels;
    # everything else on the torch backend, which is the registry default.
    # Each measured paired against the version before it, in one job:
    #   the two norm-fed sites   46.828 -> 40.240 ms   -6.588
    #   the attention glue       40.177 -> 37.040      -3.137
    #   the backbone's two norms  37.030 -> 34.423      -2.607
    #   vision LayerNorm + addmm  34.324 -> 32.753      -1.571
    #   packed expert gate+up     32.886 -> 31.396      -1.490
    plan={
        "action_expert_norm_qkv_rope": "cuda",
        "action_expert_norm_gated_ffn": "cuda",
        "action_expert_attention": "cuda",
        "action_expert_out_proj_residual": "cuda",
        "action_expert_ffn_down_residual": "cuda",
        "action_expert_action_out_proj": "cuda",
        "llm_backbone_norm_qkv_rope": "cuda",
        "llm_backbone_norm_gated_ffn": "cuda",
        "llm_backbone_out_proj_residual": "cuda",
        "llm_backbone_ffn_down_residual": "cuda",
        "llm_backbone_projector": "cuda",
        "vision_encoder_norm_qkv": "cuda",
        "vision_encoder_norm_ffn_up": "cuda",
        "vision_encoder_out_proj_residual": "cuda",
        "vision_encoder_ffn_down_residual": "cuda",
    },
    # The numerical reference stays all-torch, so `eval.correctness` compares
    # the hand-written kernels against the implementation they replaced rather
    # than against themselves.
    reference_plan={},
)

__all__ = ["TARGET"]
