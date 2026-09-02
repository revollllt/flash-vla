#pragma once

// Host-side tensor-map construction for the device loads in copy_g2s.cuh
// and the stores in copy_s2g.cuh.  Dims and box are innermost first;
// strides are bytes for dims 1.. (dim 0 is contiguous).  The swizzle here
// must equal the SmemTileLayout swizzle the consumer reads through, and a
// swizzled box row (box[0] * sizeof(T)) may not exceed the swizzle width.
// Not safe during CUDA Graph capture: build maps before capture.

#include <cuda.h>
#include <cstdint>

#include <cutlass/numeric_types.h>

namespace flash_vla::sm90 {

template <class T>
constexpr CUtensorMapDataType tma_data_type();
template <> constexpr CUtensorMapDataType tma_data_type<cutlass::bfloat16_t>() {
  return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
}
template <> constexpr CUtensorMapDataType tma_data_type<cutlass::half_t>() {
  return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
}
template <> constexpr CUtensorMapDataType tma_data_type<float>() {
  return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
}
// fp8 moves as opaque bytes; the tensor core interprets it.
template <> constexpr CUtensorMapDataType tma_data_type<cutlass::float_e4m3_t>() {
  return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}
template <> constexpr CUtensorMapDataType tma_data_type<cutlass::float_e5m2_t>() {
  return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}
template <> constexpr CUtensorMapDataType tma_data_type<uint8_t>() {
  return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}

constexpr CUtensorMapSwizzle tma_swizzle(int bytes) {
  return bytes == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
         : bytes == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
         : bytes == 32 ? CU_TENSOR_MAP_SWIZZLE_32B
                       : CU_TENSOR_MAP_SWIZZLE_NONE;
}

// 2-D: {inner, outer} elements, outer_stride_bytes between outer rows.
template <class T>
inline CUresult encode_tensor_map_2d(
    CUtensorMap* out, const void* base, uint64_t inner, uint64_t outer,
    uint64_t outer_stride_bytes, uint32_t box_inner, uint32_t box_outer,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
  uint64_t dims[2] = {inner, outer};
  uint64_t strides[1] = {outer_stride_bytes};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t elem_strides[2] = {1, 1};
  return cuTensorMapEncodeTiled(
      out, tma_data_type<T>(), 2, const_cast<void*>(base), dims, strides, box,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, promotion, oob);
}

template <class T>
inline CUresult encode_tensor_map_2d(
    CUtensorMap* out, const void* base, uint64_t inner, uint64_t outer,
    uint64_t outer_stride_bytes, uint32_t box_inner, uint32_t box_outer,
    int swizzle_bytes,
    CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
  return encode_tensor_map_2d<T>(out, base, inner, outer, outer_stride_bytes,
                                 box_inner, box_outer, tma_swizzle(swizzle_bytes),
                                 promotion, oob);
}

// 3-D: {d0, d1, d2} elements with d0 contiguous; stride1/stride2 in bytes.
template <class T>
inline CUresult encode_tensor_map_3d(
    CUtensorMap* out, const void* base, uint64_t d0, uint64_t d1, uint64_t d2,
    uint64_t stride1_bytes, uint64_t stride2_bytes, uint32_t box0,
    uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
  uint64_t dims[3] = {d0, d1, d2};
  uint64_t strides[2] = {stride1_bytes, stride2_bytes};
  uint32_t box[3] = {box0, box1, box2};
  uint32_t elem_strides[3] = {1, 1, 1};
  return cuTensorMapEncodeTiled(
      out, tma_data_type<T>(), 3, const_cast<void*>(base), dims, strides, box,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, promotion, oob);
}

template <class T>
inline CUresult encode_tensor_map_3d(
    CUtensorMap* out, const void* base, uint64_t d0, uint64_t d1, uint64_t d2,
    uint64_t stride1_bytes, uint64_t stride2_bytes, uint32_t box0,
    uint32_t box1, uint32_t box2, int swizzle_bytes,
    CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
  return encode_tensor_map_3d<T>(out, base, d0, d1, d2, stride1_bytes,
                                 stride2_bytes, box0, box1, box2,
                                 tma_swizzle(swizzle_bytes), promotion, oob);
}

}  // namespace flash_vla::sm90
