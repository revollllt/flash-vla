// hut/tma.cuh -- bulk tensor copies and their descriptor.
//
// Names are the driver API's: cuTensorMapEncodeTiled takes globalDim,
// globalStrides, boxDim, elementStrides, swizzle, oobFill, and this header uses
// those words rather than inventing any. [references/vocabulary.md]

#pragma once

#include <cuda.h>
#include <cstdint>

#include "hut/barrier.cuh"
#include "hut/common.cuh"

namespace hut {

// PTX `cp.async.bulk.tensor.2d.shared::cluster.global.tile
//      .mbarrier::complete_tx::bytes`. The `.tile` load mode is the default and
// is written out for the same reason the rest of the mnemonic is: so the source
// says which instruction it means.
__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    const CUtensorMap* tensor_map, void* smem_dst,
    int32_t coord_0, int32_t coord_1, TransactionBarrier* bar) {
  uint32_t dst = smem_addr(smem_dst), mbar = smem_addr(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :: "r"(dst), "l"(tensor_map), "r"(coord_0), "r"(coord_1), "r"(mbar)
      : "memory");
}

// Encode a 2-D tiled tensor map. Returns the CUresult so a caller can
// ENUMERATE legality rather than assume it -- which swizzle widths and how many
// box rows the driver accepts is measured, not recalled. [tma.bytes.txn.max]
//
// Host-side: cuTensorMapEncodeTiled is a driver call.
inline int encode_tensor_map(CUtensorMap* tensor_map, const void* global_address,
                             uint64_t global_dim_0, uint64_t global_dim_1,
                             uint32_t box_dim_0, uint32_t box_dim_1,
                             int32_t swizzle, int32_t tensor_data_type,
                             int32_t elem_bytes) {
  // The driver's own typedefs: cuuint64_t is `unsigned long` on LP64, so
  // passing `unsigned long long` here is a hard type error, not a warning.
  cuuint64_t global_dim[2] = {static_cast<cuuint64_t>(global_dim_0),
                              static_cast<cuuint64_t>(global_dim_1)};
  cuuint64_t global_strides[1] = {
      static_cast<cuuint64_t>(global_dim_0 * static_cast<uint64_t>(elem_bytes))};
  cuuint32_t box_dim[2] = {box_dim_0, box_dim_1};
  cuuint32_t element_strides[2] = {1, 1};
  return static_cast<int>(cuTensorMapEncodeTiled(
      tensor_map, static_cast<CUtensorMapDataType>(tensor_data_type), 2,
      const_cast<void*>(global_address), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      static_cast<CUtensorMapSwizzle>(swizzle),
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

}  // namespace hut
