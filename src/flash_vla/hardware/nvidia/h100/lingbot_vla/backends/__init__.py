"""LingBot routes: upstream LingBot and the ladder of replacements built on it.

Every backend here is the upstream backend (`upstream.BACKEND`) with some of
its `UpstreamRoute` replacements switched on. Each rung of the ladder is the
previous rung plus one measured step, so a plan that names a rung names every
step below it; the three monolithic call sites always route together
(`upstream.ROUTE_CONSTRAINTS`).
"""
from __future__ import annotations

from dataclasses import replace


from flash_vla.runtime.registry import Registry

from .upstream import BACKEND as UPSTREAM, UpstreamRoute, route_backend

#: Invariant RoPE frequencies cached at warmup.
ROPE_FREQUENCY = UpstreamRoute(cache_rope_frequency=True)
#: A GEMM for the already expanded vision patches.
PATCH_LINEAR = replace(ROPE_FREQUENCY, linear_patch_embedding=True)
#: RoPE trig tables reused within each fixed-position upstream forward.
ROPE_TABLE = replace(PATCH_LINEAR, cache_rope_tables=True)
#: Pi0.5's fixed-timestep precomputation for the AdaRMS projections.
TIME_MODULATION = replace(ROPE_TABLE, precompute_time_modulation=True)
#: AdaRMS pointwise operations fused, FP32 intermediates retained.
FUSED_NORM = replace(TIME_MODULATION, fuse_norm=True)
#: The expert layer's redundant launches removed: packed GEMMs, grouped attention.
PACKED_PROJECTIONS = replace(FUSED_NORM, pack_expert_projections=True, grouped_attention=True)
#: The denoising steps through the Target's own loop instead of upstream's.
EXPERT_LOOP = replace(PACKED_PROJECTIONS, specialized_loop=True)
#: The expert's projection epilogue as one hand-written CUDA launch.
FUSED_ROPE = replace(EXPERT_LOOP, fused_rope=True)
#: Head-major attention: no transposing copies, fused score chain and epilogue.
FUSED_ATTENTION = replace(FUSED_ROPE, fused_attention=True)
#: The expert's fused route extended to the backbone's single prefix pass.
FUSED_PREFIX = replace(FUSED_ATTENTION, specialized_prefix=True)
#: An alignment-padded, packed vision feed-forward.
ALIGNED_VISION = replace(FUSED_PREFIX, pad_vision_ffn=True)
#: The single-launch vision RMSNorm.
VISION_NORM = replace(ALIGNED_VISION, fused_vision_norm=True)
#: The expert's residual adds folded into its AdaRMS, the gated activation fused;
#: this supersedes the separately fused AdaRMS.
FUSED_MLP = replace(VISION_NORM, fused_mlp=True, fuse_norm=False)
#: The fused pointwise kernels and the packed gated projection in the backbone.
FUSED_BACKBONE = replace(FUSED_MLP, fused_prefix_pointwise=True)
#: The expert's two 768-wide output projections on the hand-written GEMM.
SKINNY_PROJECTIONS = replace(FUSED_BACKBONE, skinny_gemm=True)
#: The expert's cuBLAS attention chain on the split-key CUDA kernel.
SPLIT_ATTENTION = replace(SKINNY_PROJECTIONS, split_attention=True)
#: The vision blocks' q/k/v prepared in one launch instead of about eleven.
VISION_ATTENTION = replace(SPLIT_ATTENTION, fused_vision_attention=True)
#: The expert's gated activation in the packed gate/up GEMM's epilogue.
FUSED_GATE = replace(VISION_ATTENTION, fused_gate=True)
#: A branch off `FUSED_BACKBONE`: both towers' attention chain as one
#: hand-written flash-form launch.
ATTENTION_KERNEL = replace(FUSED_BACKBONE, attention_kernel=True)

BACKENDS = {
    "upstream-reference": UPSTREAM,
    "upstream-shipped": UPSTREAM,
    "rope-frequency": route_backend(ROPE_FREQUENCY),
    "patch-linear": route_backend(PATCH_LINEAR),
    "rope-table": route_backend(ROPE_TABLE),
    "time-modulation": route_backend(TIME_MODULATION),
    "fused-norm": route_backend(FUSED_NORM),
    "packed-projections": route_backend(PACKED_PROJECTIONS),
    "expert-loop": route_backend(EXPERT_LOOP),
    "fused-rope": route_backend(FUSED_ROPE),
    "fused-attention": route_backend(FUSED_ATTENTION),
    "fused-prefix": route_backend(FUSED_PREFIX),
    "aligned-vision": route_backend(ALIGNED_VISION),
    "vision-norm": route_backend(VISION_NORM),
    "fused-mlp": route_backend(FUSED_MLP),
    "fused-backbone": route_backend(FUSED_BACKBONE),
    "skinny-projections": route_backend(SKINNY_PROJECTIONS),
    "split-attention": route_backend(SPLIT_ATTENTION),
    "vision-attention": route_backend(VISION_ATTENTION),
    "fused-gate": route_backend(FUSED_GATE),
    "attention-kernel": route_backend(ATTENTION_KERNEL),
}
REGISTRY = Registry(BACKENDS, default="upstream-reference")

__all__ = ["BACKENDS", "REGISTRY"]
