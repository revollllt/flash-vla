"""GEMM call sites of the VLA workloads the quantized-kernel survey times.

(M, K, N) are the deployed shapes: GR00T N1.7 from lab/groot_n17/roofline.py
(checked against a real forward's FLOPs), packed where the shipped route packs;
Pi0.5 from models/pi05/spec.py. Pi0.5's SigLIP FFN width 4304 is padded to
4352, which satisfies the 32-, 128- and alignment requirements of every format.
`count` is calls per observation.
"""
from typing import Literal, NamedTuple

GemmFormat = Literal["bf16", "fp8_1d2d", "mxfp8", "nvfp4"]


class GemmSite(NamedTuple):
    name: str
    model: str
    m: int
    k: int
    n: int
    count: int


SHAPES = [
    GemmSite("groot/vit_qkv",      "groot", 512,  1024,  3072,  24),
    GemmSite("groot/vit_o",        "groot", 512,  1024,  1024,  24),
    GemmSite("groot/vit_up",       "groot", 512,  1024,  4096,  24),
    GemmSite("groot/vit_down",     "groot", 512,  4096,  1024,  24),
    GemmSite("groot/llm_qkv",      "groot", 156,  2048,  4096,  16),
    GemmSite("groot/llm_o",        "groot", 156,  2048,  2048,  32),   # plus 16 refiner q/k/v/o
    GemmSite("groot/llm_gateup",   "groot", 156,  2048,  12288, 16),
    GemmSite("groot/llm_down",     "groot", 156,  6144,  2048,  16),
    GemmSite("groot/ref_up",       "groot", 156,  2048,  8192,  4),
    GemmSite("groot/ref_down",     "groot", 156,  8192,  2048,  4),
    GemmSite("groot/dit_crosskv",  "groot", 156,  2048,  3072,  16),   # once per observation
    GemmSite("groot/dit_qkv",      "groot", 41,   1536,  4608,  64),
    GemmSite("groot/dit_o",        "groot", 41,   1536,  1536,  192),  # self o + cross q/o
    GemmSite("groot/dit_up",       "groot", 41,   1536,  6144,  128),
    GemmSite("groot/dit_down",     "groot", 41,   6144,  1536,  128),
    GemmSite("pi05/vit_qkv",       "pi05",  768,  1152,  3456,  27),
    GemmSite("pi05/vit_o",         "pi05",  768,  1152,  1152,  27),
    GemmSite("pi05/vit_up",        "pi05",  768,  1152,  4352,  27),
    GemmSite("pi05/vit_down",      "pi05",  768,  4352,  1152,  27),
    GemmSite("pi05/llm_qkv",       "pi05",  968,  2048,  2560,  18),
    GemmSite("pi05/llm_o",         "pi05",  968,  2048,  2048,  18),
    GemmSite("pi05/llm_gateup",    "pi05",  968,  2048,  32768, 18),
    GemmSite("pi05/llm_down",      "pi05",  968,  16384, 2048,  18),
    GemmSite("pi05/exp_qkv",       "pi05",  50,   1024,  2560,  180),
    GemmSite("pi05/exp_o",         "pi05",  50,   2048,  1024,  180),
    GemmSite("pi05/exp_gateup",    "pi05",  50,   1024,  8192,  180),
    GemmSite("pi05/exp_down",      "pi05",  50,   4096,  1024,  180),
]

# Measured on this RTX 5090 (lab/quantization/mma_blockscale_clock.cu,
# 2026-09-23, ~2.89 GHz): dense tensor throughput with FP32 accumulate.
PEAK_TFLOPS = {"bf16": 253.3, "fp8_1d2d": 504.4, "mxfp8": 990.1, "nvfp4": 1926.3}

# Streaming roofline: weights (and their scales) read once from DRAM at the
# sustained rate measured on this part, activations reused on chip, against the
# measured tensor peak of the format. The project's per-launch floor adds a
# 3.35 us cold-read spin-up and a 0.45 us graph node; in a chain of back-to-back
# GEMM nodes (FlashInfer launches with PDL) that spin-up overlaps the previous
# kernel, and a smoke run beat it, so it is not charged.
DRAM_MIB_PER_US = 1.524   # ld.bw.dev.dram, hardware/nvidia/rtx5090/measured/constants.yaml


def weight_bytes(fmt: GemmFormat, k: int, n: int) -> int:
    """Bytes of an [N, K] weight and its scales in `fmt`."""
    blocks_128x128 = -(-n // 128) * -(-k // 128)
    return {"bf16": 2 * n * k,
            "fp8_1d2d": n * k + 4 * blocks_128x128,
            "mxfp8": n * k + n * (k // 32),
            "nvfp4": n * k // 2 + n * (k // 16) + 4}[fmt]


def floor_us(fmt: GemmFormat, m: int, k: int, n: int) -> tuple[float, str]:
    """(roofline µs, "compute" or "memory") of an [M, K] x [K, N] GEMM."""
    compute_us = 2 * m * n * k / (PEAK_TFLOPS[fmt] * 1e6)
    memory_us = weight_bytes(fmt, k, n) / 2**20 / DRAM_MIB_PER_US
    return max(compute_us, memory_us), ("compute" if compute_us > memory_us else "memory")
