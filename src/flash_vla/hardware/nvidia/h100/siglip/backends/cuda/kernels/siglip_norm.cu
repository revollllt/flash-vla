// Vision LayerNorm over the feature axis, sm90.
//
//   out[m, k] = (x[m, k] - mean_m) * rstd_m * weight[k] + bias[k]
//
// One CTA per row. At M = 768 that is 768 CTAs, comfortably past the point
// where extra blocks become warps [sched.ctas.sm.knee: grid >= 3x SMs], and
// each row is 1152 bf16 = 2304 B, which one warpgroup reads in nine 16-byte
// chunks per thread.
//
// The op is memory bound and the target is to match, not beat, the tuned
// TileLang body it replaces: 1.77 MB in and 1.77 MB out at [ld.bw.dev.dram]
// (1.85 us fixed + MB/2.77) is about 3.1 us, and torch's generic F.layer_norm
// measured about 3.4 us/call slower than that on this shape (job 599790),
// which is the whole reason this kernel exists.
//
// Statistics are accumulated in fp32 and the affine is applied in fp32 with a
// single rounding to bf16 on output, which is the rounding point both Targets'
// shipped TileLang bodies use; an fp32 reduction is not optional at K=1152,
// since a bf16 sum of 1152 terms loses the mean.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

// Fixed by models/pi05/spec.py; siglip/geometry.py is the single mirror and
// siglip_norm.py checks these against it at load time.
constexpr int32_t DIM = 1152;
constexpr int32_t kThreads = 128;
constexpr int32_t kChunk = 8;                       // bf16 per 16-byte access
constexpr int32_t kChunks = DIM / kChunk;           // 144
static_assert(DIM % kChunk == 0, "a row is a whole number of 16-byte chunks");
// 144 chunks over 128 threads: the first 16 threads take a second chunk. Two
// registers of payload per thread is the whole working set, so the imbalance
// costs one extra iteration rather than a spill.
constexpr int32_t kPerThread = (kChunks + kThreads - 1) / kThreads;  // 2
constexpr int32_t kWarps = kThreads / 32;

struct __align__(16) Chunk { __nv_bfloat16 v[kChunk]; };

// Sum across the block: warp shuffles, then one round through shared memory.
// Returns the total on every thread.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
#pragma unroll
  for (int32_t off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
  const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
  if (lane == 0) { scratch[warp] = v; }
  __syncthreads();
  float total = 0.f;
#pragma unroll
  for (int32_t w = 0; w < kWarps; ++w) { total += scratch[w]; }
  return total;
}

}  // namespace

// grid = (rows,); one CTA owns one row of `x`.
__global__ __launch_bounds__(kThreads, 1)
void siglip_norm_kernel(const __nv_bfloat16* __restrict__ x,
                        const __nv_bfloat16* __restrict__ weight,
                        const __nv_bfloat16* __restrict__ bias,
                        __nv_bfloat16* __restrict__ out, float eps) {
  __shared__ float reduce[kWarps];
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const long row = static_cast<long>(blockIdx.x) * DIM;
  const Chunk* const xr = reinterpret_cast<const Chunk*>(x + row);
  Chunk* const orow = reinterpret_cast<Chunk*>(out + row);

  // One pass over the row, held in registers: the sum and the sum of squares
  // come from the same read, so the second traversal below is from registers
  // rather than from memory. Two traversals is this op's roofline; the only
  // way past it is fusing into a neighbour.
  Chunk held[kPerThread];
  float sum = 0.f, sumsq = 0.f;
#pragma unroll
  for (int32_t i = 0; i < kPerThread; ++i) {
    const int32_t c = tid + i * kThreads;
    if (c < kChunks) {
      held[i] = xr[c];
#pragma unroll
      for (int32_t j = 0; j < kChunk; ++j) {
        const float v = __bfloat162float(held[i].v[j]);
        sum += v;
        sumsq += v * v;
      }
    }
  }
  const float mean = block_sum(sum, reduce) * (1.f / DIM);
  __syncthreads();  // `reduce` is reused by the second reduction
  const float meansq = block_sum(sumsq, reduce) * (1.f / DIM);
  // E[x^2] - mean^2 is the form the TileLang body uses; it is one pass rather
  // than two and the catastrophic-cancellation case needs a variance far below
  // what a normalized activation reaches.
  const float rstd = rsqrtf(fmaxf(meansq - mean * mean, 0.f) + eps);

  const Chunk* const wr = reinterpret_cast<const Chunk*>(weight);
  const Chunk* const br = reinterpret_cast<const Chunk*>(bias);
#pragma unroll
  for (int32_t i = 0; i < kPerThread; ++i) {
    const int32_t c = tid + i * kThreads;
    if (c < kChunks) {
      const Chunk wc = wr[c];
      const Chunk bc = br[c];
      Chunk o;
#pragma unroll
      for (int32_t j = 0; j < kChunk; ++j) {
        const float v = (__bfloat162float(held[i].v[j]) - mean) * rstd;
        o.v[j] = __float2bfloat16(v * __bfloat162float(wc.v[j])
                                  + __bfloat162float(bc.v[j]));
      }
      orow[c] = o;
    }
  }
}

// ------------------------------------------------------------------ host ABI
// A plain launch: nothing here calls the driver, so the path is safe inside
// CUDA-graph capture.
extern "C" int siglip_norm_launch(const void* x, const void* weight,
                                  const void* bias, void* out, int rows,
                                  float eps, void* stream) {
  if (rows <= 0) { return 1; }
  siglip_norm_kernel<<<static_cast<unsigned>(rows), kThreads, 0,
                       static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias), static_cast<__nv_bfloat16*>(out), eps);
  return static_cast<int>(cudaGetLastError());
}

extern "C" void siglip_norm_geometry(int* dim, int* threads) {
  *dim = DIM;
  *threads = kThreads;
}
