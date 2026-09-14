// Fixed lab mapping: 16 queries x 32 features, four warps, K64, three V stages.
// Full-row softmax -> BF16 shared P -> CuTe BF16/FP32 MMA. No online rescaling.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>

namespace {
using namespace cute;
using Element = cute::bfloat16_t;
using PAtom = decltype(composition(
    Swizzle<3,3,3>{},
    Layout<Shape<_8,Shape<_8,_8>>,Stride<_8,Stride<_1,_64>>>{}));
using PLayout = decltype(tile_to_shape(PAtom{}, Shape<_16,_64,_16>{}));
using VLayout = decltype(composition(
    Swizzle<2,3,3>{},
    Layout<Shape<_32,_64,_3>,Stride<_1,_32,_2048>>{}));
struct alignas(16) Shared {
  Element p[cosize_v<PLayout>];
  Element v[cosize_v<VLayout>];
  __nv_bfloat16 mask[1024];
};

__device__ void stage_v(const Element* values, Element* shared,
                        int32_t tile, int32_t stage, int32_t column) {
  // Each warp copies contiguous eight-element BF16 vectors. Invalid K is zero.
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    const int32_t vector = threadIdx.x + 128 * i;
    const int32_t n = (vector & 3) * 8;
    const int32_t k = vector >> 2;
    const int32_t key = tile * 64 + k;
    const bool valid = key < 1018;
    const Element* src = values + (valid ? key * 256 + column + n : 0);
    Element* dst = shared + VLayout{}(n, k, stage);
    SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
        *reinterpret_cast<const uint128_t*>(src),
        *reinterpret_cast<uint128_t*>(dst), valid);
  }
  cp_async_fence();
}

__global__ __launch_bounds__(128)
void softmax_pv(const float* __restrict__ logits,
                const __nv_bfloat16* __restrict__ mask,
                const Element* __restrict__ values, Element* __restrict__ output,
                __nv_bfloat16* __restrict__ diagnostic_p) {
  __shared__ Shared smem;
  const int32_t tid = threadIdx.x;
  const int32_t lane = tid & 31;
  const int32_t warp = tid >> 5;
  const int32_t row_base = blockIdx.x * 16;
  const int32_t col_base = blockIdx.y * 32;
  for (int32_t key = tid; key < 1024; key += 128)
    smem.mask[key] = key < 1018 ? mask[key] : __float2bfloat16_rn(0.0f);
  __syncthreads();

  // One warp owns one complete row. Four sequential rows/warp limit held state.
  // The XOR sum tree differs from the original 256-thread native softmax.
#pragma unroll 1
  for (int32_t group = 0; group < 4; ++group) {
    const int32_t row = warp + 4 * group;
    float held[32];
    float maximum = -CUDART_INF_F;
#pragma unroll
    for (int32_t i = 0; i < 32; ++i) {
      const int32_t key = lane + 32 * i;
      held[i] = key < 1018
          ? logits[int64_t(row_base + row) * 1018 + key] * 0.0625f
              + __bfloat162float(smem.mask[key])
          : -CUDART_INF_F;
      maximum = fmaxf(maximum, held[i]);
    }
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    float sum = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < 32; ++i) {
      held[i] = expf(held[i] - maximum);
      sum += held[i];
    }
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    const float reciprocal = 1.0f / sum;
#pragma unroll
    for (int32_t i = 0; i < 32; ++i) {
      const int32_t key = lane + 32 * i;
      const __nv_bfloat16 p = __float2bfloat16_rn(held[i] * reciprocal);
      reinterpret_cast<__nv_bfloat16*>(smem.p)[
          PLayout{}(row, key & 63, key >> 6)] = p;
      // Only one feature CTA writes diagnostic P; nullptr is the timed path.
      if (diagnostic_p && blockIdx.y == 0 && key < 1018)
        diagnostic_p[int64_t(row_base + row) * 1018 + key] = p;
    }
  }
  // All warps must publish their probability rows before any MMA reads P.
  __syncthreads();

  auto mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                            Layout<Shape<_1,_4,_1>>{});
  auto thr_mma = mma.get_slice(tid);
  auto sP = make_tensor(make_smem_ptr(smem.p), PLayout{});
  auto sV = make_tensor(make_smem_ptr(smem.v), VLayout{});
  auto gOut = make_tensor(make_gmem_ptr(output + row_base * 256 + col_base),
                          Shape<_16,_32>{}, Stride<_256,_1>{});
  auto tOut = thr_mma.partition_C(gOut);
  auto rP = thr_mma.partition_fragment_A(sP(_,_,0));
  auto rV = thr_mma.partition_fragment_B(sV(_,_,0));
  auto accum = thr_mma.make_fragment_C(tOut);
  clear(accum);

  Copy_Atom<SM75_U32x4_LDSM_N, Element> atom_p;
  Copy_Atom<SM75_U16x4_LDSM_T, Element> atom_v;
  auto p_copy = make_tiled_copy_A(atom_p, mma).get_slice(tid);
  auto v_copy = make_tiled_copy_B(atom_v, mma).get_slice(tid);
  auto tPsP = p_copy.partition_S(sP);
  auto tPrP = p_copy.retile_D(rP);
  auto tVsV = v_copy.partition_S(sV);
  auto tVrV = v_copy.retile_D(rV);

  stage_v(values, smem.v, 0, 0, col_base);
  stage_v(values, smem.v, 1, 1, col_base);
#pragma unroll 1
  for (int32_t tile = 0; tile < 16; ++tile) {
    // Keep three committed groups, including zero-fill drain groups.
    stage_v(values, smem.v, tile + 2, (tile + 2) % 3, col_base);
    cp_async_wait<2>();
    __syncthreads();
    copy(atom_p, tPsP(_,_,_,tile), tPrP);
    copy(atom_v, tVsV(_,_,_,tile % 3), tVrV);
    CUTE_UNROLL
    for (int32_t k = 0; k < size<2>(rP); ++k)
      gemm(mma, rP(_,_,k), rV(_,_,k), accum);
    // Readers finish before the next iteration recycles this V stage.
    __syncthreads();
  }
  cp_async_wait<0>();
  auto rounded = make_tensor<Element>(shape(accum));
  CUTE_UNROLL
  for (int32_t i = 0; i < size(accum); ++i)
    rounded(i) = Element(accum(i));
  copy(rounded, tOut);
}
}  // namespace

extern "C" int32_t softmax_pv_launch(
    const void* logits, const void* mask, const void* values, void* output,
    void* diagnostic_p, void* stream) {
  softmax_pv<<<dim3(25,8),128,0,static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(logits),
      static_cast<const __nv_bfloat16*>(mask),
      static_cast<const Element*>(values), static_cast<Element*>(output),
      static_cast<__nv_bfloat16*>(diagnostic_p));
  return static_cast<int32_t>(cudaGetLastError());
}
