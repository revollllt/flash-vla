"""Static specifications for the NVIDIA GeForce RTX 5090 (GB202, sm_120).

Capacities are bytes, bandwidths are bytes/second, floating-point throughput is
FLOP/second. Published peak throughputs are hardware ceilings, not expected
sustained performance: they are the roofline's denominator, and the measured
table beside this file is the ceiling's.

This is consumer Blackwell (compute capability 12.0), which is a different
instruction set from datacenter Blackwell (10.0) and from Hopper (9.0). See
``measured/isa-support.md`` for what that costs a kernel ported from sm_90a.

Fields marked DRIVER were read from ``cudaGetDeviceProperties`` on the RTX 5090
this file was written against; fields marked MEASURED come from a probe under
``lab/sm120/`` and name it.

The tensor-core ladder is MEASURED rather than derived, and it is stated as
FLOP per cycle per SM rather than as TFLOP/s. An earlier version of this file
derived it from the FP32 peak, omitted the accumulator width, and was wrong by
2x for every kernel in this repository -- all of which accumulate in fp32.
Consumer Blackwell runs fp32 accumulate at half the rate of fp16 accumulate,
which is the difference between NVIDIA's published 419 TF and the 209.6 a fp32
mainloop can reach.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


KB = 1 << 10
MB = 1 << 20
GB = 1 << 30


class RTX5090Spec:
    """Hardware constants for one NVIDIA GeForce RTX 5090 32 GB.

    This class describes the RTX 5090 specifically. The RTX 5090 D, 5080 and
    the RTX PRO 6000 Blackwell share compute capability 12.0 but differ in SM
    count, memory system and clocks, so none of the throughput figures here
    transfer to them.
    """

    SOURCES: Mapping[str, str] = MappingProxyType(
        {
            "product": "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/",
            "tuning_guide": "https://docs.nvidia.com/cuda/blackwell-tuning-guide/",
            "compatibility_guide": "https://docs.nvidia.com/cuda/blackwell-compatibility-guide/",
            "ptx_isa": "https://docs.nvidia.com/cuda/parallel-thread-execution/",
            "compute_capability": (
                "https://docs.nvidia.com/cuda/cuda-c-programming-guide/"
                "#compute-capabilities"
            ),
        }
    )

    # Identity and compiler target. `sm_120a` is architecture-conditional and,
    # per the compatibility guide, neither forward nor backward compatible; it
    # is required for block-scaled mma, setmaxnreg and ldmatrix .m16n16.
    NAME = "NVIDIA GeForce RTX 5090"
    ARCHITECTURE = "Blackwell (consumer)"
    DIE = "GB202"
    FORM_FACTOR = "PCIe add-in card"
    COMPUTE_CAPABILITY = (12, 0)
    BASELINE_CUDA_ARCH = "sm_120"
    CUDA_ARCH = "sm_120a"

    # SM topology. The RTX 5090 is a cut-down GB202; the full die has 192 SMs,
    # so full-die figures must not be used for this target. DRIVER: 170.
    SM_COUNT = 170
    FP32_CUDA_CORES_PER_SM = 128
    TENSOR_CORES_PER_SM = 4
    WARP_SCHEDULERS_PER_SM = 4
    FP32_CUDA_CORE_COUNT = SM_COUNT * FP32_CUDA_CORES_PER_SM  # 21_760
    TENSOR_CORE_COUNT = SM_COUNT * TENSOR_CORES_PER_SM  # 680
    TENSOR_CORE_GENERATION = 5

    MANUFACTURING_PROCESS = "TSMC 4N"
    MAX_TDP_WATTS = 575
    # Marketed boost clock. DRIVER reports a 2_550_000 kHz maximum clock; the
    # published boost is the figure the FP32 peak below is quoted at.
    BOOST_CLOCK_HZ = 2_407_000_000
    DRIVER_REPORTED_MAX_CLOCK_HZ = 2_550_000_000

    # Published dense peak for non-Tensor CUDA cores:
    #   21_760 lanes x 2 FLOP/FMA x 2.407 GHz = 104.8 TFLOP/s.
    CUDA_CORE_PEAK_FLOPS: Mapping[str, int] = MappingProxyType(
        {"fp32": 104_800_000_000_000}
    )

    # MEASURED dense tensor-core ladder, expressed the way the hardware
    # actually works: FLOP per cycle per SM. That is clock-free, and every
    # TFLOP/s figure -- NVIDIA's or anyone's -- is it multiplied by a clock.
    #
    # THE ACCUMULATOR WIDTH IS PART OF THE PEAK, and an earlier version of this
    # file omitted it. Consumer Blackwell runs fp32 accumulate at HALF the rate
    # of fp16 accumulate, which is why NVIDIA's published "419 TF dense" for
    # this part -- confirmed on their forums as the FP16-with-FP16-accumulate
    # figure -- is twice what a kernel accumulating in fp32 can reach. Every
    # mainloop in this repository accumulates in fp32.
    #
    # Measured by lab/sm120/mma_clock.cu; see measured/unit-mma.md.
    TENSOR_FLOP_PER_CYCLE_PER_SM: Mapping[str, int] = MappingProxyType(
        {
            "fp16_acc_fp16": 1024,  # measured 1023.9
            "fp16_acc_fp32": 512,   # measured 511.5
            "bf16_acc_fp32": 512,   # measured 511.5
            "fp8_acc_fp32": 1024,   # measured 1023.0
            # kind::mxf8f6f4.block_scale, e4m3 x e4m3 with ue8m0 per 32, fp32
            # accumulate: measured 2016.3 at 8 and 16 warps per SM
            # (lab/quantization/mma_blockscale_clock.cu, 2026-09-23).
            "mxfp8_block_scaled": 2048,
        }
    )

    #: Dense peaks at the marketed boost clock. Quoted for comparison with
    #: NVIDIA's published figures, NOT as a target: this part was observed
    #: running at 2.84-2.91 GHz under a pure tensor load, well above both the
    #: marketed boost and the driver-reported maximum, so a real ceiling is
    #: higher than these and varies with the clock. Use
    #: TENSOR_FLOP_PER_CYCLE_PER_SM with an observed clock instead.
    TENSOR_CORE_DENSE_PEAK_FLOPS: Mapping[str, int] = MappingProxyType(
        {
            "fp16_acc_fp16": 419_200_000_000_000,   # matches NVIDIA's 419 TF
            "fp16_acc_fp32": 209_600_000_000_000,
            "bf16_acc_fp32": 209_600_000_000_000,
            "fp8_acc_fp32": 419_200_000_000_000,
            # 2048 FLOP/cycle/SM at the marketed boost; NVIDIA's 3352 sparse
            # FP4 TOPS halved twice. The block-scaled instruction measures it.
            "mxfp8_block_scaled": 838_000_000_000_000,
        }
    )
    #: The marketed "3352 AI TOPS" is FP4 with 2:4 sparsity. Halving twice gives
    #: 838 TFLOP/s dense fp8, which needs 2048 FLOP/cycle/SM -- twice what fp8
    #: with fp32 accumulate measures. The block-scaled MXFP8 instruction measures
    #: 2016 FLOP/cycle/SM, so that row is recorded above; the fp4 and sparse rows
    #: are still left out rather than derived.
    MARKETED_FP4_SPARSE_OPS = 3_352_000_000_000_000

    TENSOR_CORE_SUPPORTED_INPUT_DTYPES = (
        "tf32",
        "bf16",
        "fp16",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp6_e2m3",
        "fp6_e3m2",
        "fp4_e2m1",
    )

    # GDDR7 and L2. DRIVER totalGlobalMem is 33_660_010_496 B; the marketed
    # capacity is 32 GB. The driver reports a 14_001 MHz memory clock, which is
    # half the 28 Gbps per-pin data rate GDDR7 signals at -- 28e9 x 512 / 8 is
    # the 1792 GB/s below, and quoting the driver figure directly halves it.
    DRAM_TYPE = "GDDR7"
    DRAM_CAPACITY_MARKETED_GB = 32
    DRAM_RUNTIME_VISIBLE_BYTES = 33_660_010_496  # DRIVER
    DRAM_BUS_WIDTH_BITS = 512  # DRIVER
    DRAM_DATA_RATE_BITS_PER_SECOND_PER_PIN = 28_000_000_000
    DRAM_BANDWIDTH_BYTES_PER_SECOND = 1_792_000_000_000
    L2_CACHE_SIZE_BYTES = 100_663_296  # DRIVER: 96 MiB
    L2_BANDWIDTH_BYTES_PER_SECOND: int | None = None

    # Shared memory. Two numbers that are easy to conflate: the tuning guide
    # states a 128 KB unified data cache per SM, and the DRIVER reports only
    # 100 KB of it reachable as shared memory, of which a block may opt in to
    # 99 KB. The 99 KB figure is the one a tile budget is built from -- and it
    # is 2.29x SMALLER than the 227 KB an sm_90a kernel may assume, which is
    # the single most disruptive number in this file for a port.
    UNIFIED_L1_TEXTURE_SHARED_CACHE_BYTES_PER_SM = 128 * KB
    SHARED_MEMORY_BYTES_PER_SM = 102_400  # DRIVER
    SHARED_MEMORY_BYTES_PER_BLOCK_OPTIN = 101_376  # DRIVER
    STATIC_SHARED_MEMORY_BYTES_PER_BLOCK = 48 * KB  # DRIVER default
    SHARED_MEMORY_BANK_COUNT = 32
    SHARED_MEMORY_BANK_WIDTH_BYTES = 4

    # Register file and occupancy limits for compute capability 12.0. The warp
    # ceiling is 48, NOT Hopper's 64: a 2048-thread-per-SM occupancy assumption
    # carried over from sm_90 overshoots by 1.33x.
    REGISTER_WIDTH_BITS = 32
    REGISTERS_PER_SM = 64 * KB  # DRIVER: 65_536
    REGISTERS_PER_BLOCK = 64 * KB
    MAX_REGISTERS_PER_THREAD = 255
    WARP_SIZE = 32
    MAX_THREADS_PER_BLOCK = 1_024  # DRIVER
    MAX_THREADS_PER_SM = 1_536  # DRIVER
    MAX_WARPS_PER_SM = 48
    MAX_BLOCKS_PER_SM = 32

    # Cluster limits. MEASURED on this device rather than assumed: clusters of
    # 1, 2, 4 and 8 launch, synchronise and exchange distributed shared memory;
    # 16 is refused with `cluster misconfiguration`. See measured/unit-launch.md.
    MAX_PORTABLE_BLOCKS_PER_CLUSTER = 8
    MAX_NONPORTABLE_BLOCKS_PER_CLUSTER = 8
    SUPPORTS_THREAD_BLOCK_CLUSTERS = True
    SUPPORTS_DISTRIBUTED_SHARED_MEMORY = True

    # Instruction-set facts that change kernel structure. Each is decided by
    # ptxas from the CUDA 13.1 toolkit, not inferred; measured/isa-support.md
    # carries the verbatim diagnostic for every False below.
    SUPPORTS_TMA = True
    TMA_MAX_TENSOR_RANK = 5
    SUPPORTS_WGMMA = False  # "not supported on .target 'sm_120a'"
    SUPPORTS_TCGEN05 = False  # datacenter Blackwell (sm_100a) only
    SUPPORTS_TENSOR_MEMORY = False
    SUPPORTS_SETMAXNREG = True  # requires sm_120a, NOT sm_120
    SUPPORTS_BLOCK_SCALED_MMA = True  # requires sm_120a
    SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH = True

    PCIE_GENERATION = 5
    PCIE_LANE_COUNT = 16
    NVLINK_GENERATION = None  # no NVLink on this SKU

    @classmethod
    def theoretical_shared_memory_bandwidth(
        cls, clock_hz: int | float | None = None
    ) -> int | float:
        """Return the conflict-free shared-memory bank ceiling at ``clock_hz``."""
        if clock_hz is None:
            clock_hz = cls.BOOST_CLOCK_HZ
        if clock_hz <= 0:
            raise ValueError("clock_hz must be positive")
        return (
            cls.SM_COUNT
            * cls.SHARED_MEMORY_BANK_COUNT
            * cls.SHARED_MEMORY_BANK_WIDTH_BYTES
            * clock_hz
        )


__all__ = ["RTX5090Spec"]
