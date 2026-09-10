"""Register shared fused RMS builders in this Target's JIT namespace."""
from flash_vla.hardware.nvidia.h100.gemma_expert.backends.tilelang.kernels import fused_norm

from .base import kernel

tl_fused_rms_gate = kernel(fused_norm.tl_fused_rms_gate)
tl_fused_rms_matmul_bias_res = kernel(fused_norm.tl_fused_rms_matmul_bias_res)
