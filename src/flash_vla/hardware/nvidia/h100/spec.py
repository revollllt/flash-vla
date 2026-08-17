"""Static specifications for the NVIDIA H100 SXM5 80 GB target.

Capacities are bytes, bandwidths are bytes/second, floating-point throughput
is FLOP/second, and integer throughput is operation/second. Published peak
throughputs are hardware ceilings, not expected sustained performance.

NVIDIA does not publish a reliable L1 or L2 bandwidth for this SKU. Those
fields intentionally remain ``None`` and should only be populated from a
device-specific microbenchmark with its clock and profiler provenance.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


KB = 1 << 10
MB = 1 << 20
GB = 1 << 30


class H100Spec:
    """Hardware constants for one NVIDIA H100 SXM5 80 GB GPU.

    This class deliberately describes the SXM5 SKU used by this repository;
    H100 PCIe and H100 NVL have different SM counts, throughput, memory, and
    interconnect characteristics.
    """

    SOURCES: Mapping[str, str] = MappingProxyType(
        {
            "product": "https://www.nvidia.com/en-us/data-center/h100/",
            "data_sheet": (
                "https://resources.nvidia.com/en-us-hopper-architecture/"
                "nvidia-tensor-core-gpu-datasheet"
            ),
            "architecture": (
                "https://developer.nvidia.com/blog/"
                "nvidia-hopper-architecture-in-depth/"
            ),
            "tuning_guide": "https://docs.nvidia.com/cuda/hopper-tuning-guide/",
            "compute_capability": (
                "https://docs.nvidia.com/cuda/cuda-c-programming-guide/"
                "#compute-capabilities"
            ),
            "best_practices": (
                "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/"
                "#shared-memory-and-memory-banks"
            ),
        }
    )

    # Identity and compiler target.
    NAME = "NVIDIA H100 SXM5 80GB"
    ARCHITECTURE = "Hopper"
    DIE = "GH100"
    FORM_FACTOR = "SXM5"
    COMPUTE_CAPABILITY = (9, 0)
    BASELINE_CUDA_ARCH = "sm_90"
    CUDA_ARCH = "sm_90a"

    # Physical and SM topology. H100 SXM5 is a cut-down 132-SM GH100; the full
    # die has 144 SMs, so full-die figures must not be used for this target.
    GPC_COUNT = 8
    TPC_COUNT = 66
    SM_COUNT = 132
    SMS_PER_TPC = 2
    FP32_CUDA_CORES_PER_SM = 128
    FP64_CUDA_CORES_PER_SM = 64
    INT32_CORES_PER_SM = 64
    TENSOR_CORES_PER_SM = 4
    TEXTURE_UNITS_PER_SM = 4
    WARP_SCHEDULERS_PER_SM = 4
    FP32_CUDA_CORE_COUNT = SM_COUNT * FP32_CUDA_CORES_PER_SM
    FP64_CUDA_CORE_COUNT = SM_COUNT * FP64_CUDA_CORES_PER_SM
    INT32_CORE_COUNT = SM_COUNT * INT32_CORES_PER_SM
    TENSOR_CORE_COUNT = SM_COUNT * TENSOR_CORES_PER_SM
    TEXTURE_UNIT_COUNT = SM_COUNT * TEXTURE_UNITS_PER_SM

    TRANSISTOR_COUNT = 80_000_000_000
    DIE_SIZE_SQUARE_MILLIMETERS = 814
    MANUFACTURING_PROCESS = "TSMC 4N"
    MAX_TDP_WATTS = 700
    MAX_BOOST_CLOCK_HZ = 1_980_000_000

    # Final published dense peaks for non-Tensor CUDA cores. NVIDIA does not
    # publish final scalar/SIMT BF16 and FP16 peaks on the current product page.
    CUDA_CORE_PEAK_FLOPS: Mapping[str, int] = MappingProxyType(
        {
            "fp64": 34_000_000_000_000,
            "fp32": 67_000_000_000_000,
        }
    )

    # Final Tensor Core peaks. The current product page publishes the values
    # marked with sparsity; dense values are the corresponding non-sparse H100
    # data-sheet figures. Marketing rounding makes some sparse values differ
    # from exactly 2x dense by 1 TFLOP/s.
    TENSOR_CORE_DENSE_PEAK_FLOPS: Mapping[str, int] = MappingProxyType(
        {
            "fp64": 67_000_000_000_000,
            "tf32": 494_000_000_000_000,
            "bf16": 989_000_000_000_000,
            "fp16": 989_000_000_000_000,
            "fp8": 1_979_000_000_000_000,
        }
    )
    TENSOR_CORE_2_TO_4_SPARSE_PEAK_FLOPS: Mapping[str, int] = MappingProxyType(
        {
            "tf32": 989_000_000_000_000,
            "bf16": 1_979_000_000_000_000,
            "fp16": 1_979_000_000_000_000,
            "fp8": 3_958_000_000_000_000,
        }
    )
    TENSOR_CORE_DENSE_PEAK_OPS: Mapping[str, int] = MappingProxyType(
        {"int8": 1_979_000_000_000_000}
    )
    TENSOR_CORE_2_TO_4_SPARSE_PEAK_OPS: Mapping[str, int] = MappingProxyType(
        {"int8": 3_958_000_000_000_000}
    )
    TENSOR_CORE_SUPPORTED_INPUT_DTYPES = (
        "fp64",
        "tf32",
        "bf16",
        "fp16",
        "fp8_e4m3",
        "fp8_e5m2",
    )

    # HBM3 and L2. NVIDIA markets this SKU as 80 GB; the nominal physical
    # capacity is 80 GiB. cudaDeviceProp.totalGlobalMem is the authority for
    # allocatable capacity and is slightly smaller due to reserved memory.
    HBM_TYPE = "HBM3"
    HBM_CAPACITY_MARKETED_GB = 80
    HBM_CAPACITY_BYTES = 80 * GB
    HBM_RUNTIME_VISIBLE_BYTES: int | None = None
    HBM_STACK_COUNT = 5
    HBM_MEMORY_CONTROLLER_COUNT = 10
    HBM_MEMORY_CONTROLLER_WIDTH_BITS = 512
    HBM_BUS_WIDTH_BITS = 5_120
    HBM_BANDWIDTH_BYTES_PER_SECOND = 3_350_000_000_000
    DRAM_BANDWIDTH_BYTES_PER_SECOND = HBM_BANDWIDTH_BYTES_PER_SECOND
    HBM_EFFECTIVE_DATA_RATE_BITS_PER_SECOND_PER_PIN = (
        HBM_BANDWIDTH_BYTES_PER_SECOND * 8 // HBM_BUS_WIDTH_BITS
    )

    L2_CACHE_SIZE_BYTES = 50 * MB
    L2_CACHE_PARTITION_COUNT = HBM_MEMORY_CONTROLLER_COUNT
    L2_CACHE_BYTES_PER_PARTITION = L2_CACHE_SIZE_BYTES // L2_CACHE_PARTITION_COUNT
    L2_BANDWIDTH_BYTES_PER_SECOND: int | None = None
    PERSISTING_L2_CACHE_MAX_BYTES: int | None = None

    # The 256 KiB unified data cache is partitioned between L1/texture and
    # shared memory. CUDA reserves 1 KiB per block at the 228 KiB carveout,
    # leaving at most 227 KiB addressable by a block.
    UNIFIED_L1_TEXTURE_SHARED_CACHE_BYTES_PER_SM = 256 * KB
    UNIFIED_L1_TEXTURE_CACHE_BANDWIDTH_BYTES_PER_SECOND: int | None = None
    SHARED_MEMORY_BYTES_PER_SM = 228 * KB
    SHARED_MEMORY_BYTES_PER_BLOCK = 227 * KB
    STATIC_SHARED_MEMORY_BYTES_PER_BLOCK = 48 * KB
    RESERVED_SHARED_MEMORY_BYTES_PER_BLOCK = 1 * KB
    SHARED_MEMORY_CARVEOUT_OPTIONS_BYTES = tuple(
        value * KB for value in (0, 8, 16, 32, 64, 100, 132, 164, 196, 228)
    )
    SHARED_MEMORY_BANK_COUNT = 32
    SHARED_MEMORY_BANK_WIDTH_BYTES = 4

    # This is the conflict-free, one-direction bank ceiling at the maximum
    # boost clock, not an observed application bandwidth.
    SHARED_MEMORY_THEORETICAL_BANDWIDTH_BYTES_PER_SECOND = (
        SM_COUNT
        * SHARED_MEMORY_BANK_COUNT
        * SHARED_MEMORY_BANK_WIDTH_BYTES
        * MAX_BOOST_CLOCK_HZ
    )

    # Register file and occupancy limits for compute capability 9.0.
    REGISTER_WIDTH_BITS = 32
    REGISTERS_PER_SM = 64 * KB
    REGISTERS_PER_BLOCK = 64 * KB
    MAX_REGISTERS_PER_THREAD = 255
    REGISTER_FILE_BYTES_PER_SM = REGISTERS_PER_SM * REGISTER_WIDTH_BITS // 8
    WARP_SIZE = 32
    WARP_GROUP_SIZE = 4 * WARP_SIZE
    MAX_THREADS_PER_BLOCK = 1_024
    MAX_THREADS_PER_SM = 2_048
    MAX_WARPS_PER_SM = 64
    MAX_BLOCKS_PER_SM = 32
    MAX_RESIDENT_GRIDS = 128
    MAX_BLOCK_DIM = (1_024, 1_024, 64)
    MAX_GRID_DIM = (2**31 - 1, 65_535, 65_535)
    MAX_LOCAL_MEMORY_BYTES_PER_THREAD = 512 * KB
    CONSTANT_MEMORY_SIZE_BYTES = 64 * KB
    CONSTANT_CACHE_WORKING_SET_BYTES_PER_SM = 8 * KB

    # Hopper cluster limits. A cluster of 16 requires explicit non-portable
    # opt-in and can reduce occupancy; runtime occupancy APIs remain authoritative.
    MAX_PORTABLE_BLOCKS_PER_CLUSTER = 8
    MAX_NONPORTABLE_BLOCKS_PER_CLUSTER = 16

    # Interconnect figures are NVIDIA's aggregate marketed bandwidths.
    NVLINK_GENERATION = 4
    NVLINK_LINK_COUNT = 18
    NVLINK_AGGREGATE_BANDWIDTH_BYTES_PER_SECOND = 900_000_000_000
    PCIE_GENERATION = 5
    PCIE_LANE_COUNT = 16
    PCIE_AGGREGATE_BANDWIDTH_BYTES_PER_SECOND = 128_000_000_000
    PCIE_BANDWIDTH_BYTES_PER_SECOND_PER_DIRECTION = (
        PCIE_AGGREGATE_BANDWIDTH_BYTES_PER_SECOND // 2
    )

    # Pipeline-relevant fixed-function resources and architecture features.
    NVDEC_COUNT = 7
    JPEG_DECODER_COUNT = 7
    MAX_MIG_INSTANCE_COUNT = 7
    MIG_INSTANCE_MEMORY_MARKETED_BYTES = 10_000_000_000
    TMA_MAX_TENSOR_RANK = 5
    SUPPORTS_TMA = True
    SUPPORTS_WGMMA = True
    SUPPORTS_THREAD_BLOCK_CLUSTERS = True
    SUPPORTS_DISTRIBUTED_SHARED_MEMORY = True
    SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH = True
    SUPPORTS_L2_RESIDENCY_CONTROL = True
    SUPPORTS_COMPUTE_DATA_COMPRESSION = True
    SUPPORTS_DPX = True

    @classmethod
    def theoretical_shared_memory_bandwidth(
        cls, clock_hz: int | float | None = None
    ) -> int | float:
        """Return the conflict-free shared-memory bank ceiling at ``clock_hz``."""
        if clock_hz is None:
            clock_hz = cls.MAX_BOOST_CLOCK_HZ
        if clock_hz <= 0:
            raise ValueError("clock_hz must be positive")
        return (
            cls.SM_COUNT
            * cls.SHARED_MEMORY_BANK_COUNT
            * cls.SHARED_MEMORY_BANK_WIDTH_BYTES
            * clock_hz
        )


__all__ = ["H100Spec"]
