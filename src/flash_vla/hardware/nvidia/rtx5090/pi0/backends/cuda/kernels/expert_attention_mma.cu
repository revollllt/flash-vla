// Pi0's action-expert attention as one tensor-core kernel, on sm_120.
//
// The CUDA-core attempt in `expert_attention.cu` is correct and 8x slower than
// splitting the work across cuBLAS and a softmax kernel. ncu settled why: 32.09M
// instructions for 342 MFLOP, because a dot product out of shared memory costs
// two loads and an FFMA per two FLOP. Even perfectly spread over the machine
// that is 32M / (170 SMs x 4 schedulers) = 16 us, which is what the split form
// already costs -- so no amount of tiling or occupancy work on CUDA cores can
// win, and only `mma.sync` clears it [mma.rate.sm.bf16].
//
// One `mma.sync.m16n8k16` does 4096 FLOP for 12 shared-memory loads, against the
// CUDA-core form's 512 for the same FLOP: a factor of 40 in instruction count.
//
// Shape: 51 tokens x 8 query heads = 408 flat queries, 768 prefix + 51 suffix =
// 819 keys, head dimension 256, multi-query so all eight heads share one KV.
//
// Tiling, and why. BLOCK_M is 16 because that is one mma M tile. BLOCK_N is 32
// so the whole working set -- Q, K, V-transposed, the scores and the
// probabilities -- fits the 48 KB of STATIC shared memory rather than needing
// the 99 KB opt-in [smem.bytes.cta.max]. That gives 408/16 = 26 CTAs, and the
// compute floor at 26 SMs is 342 MFLOP / (26 x 512 FLOP/cycle x 2.89 GHz) =
// 8.9 us. Splitting keys across CTAs would buy more SMs but a 16 x 256 fp32
// partial per split costs more in round-trip than it saves.
#include <cstdint>
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "mma_bf16.cuh"

namespace {

using flash_vla::rtx5090::load_a;
using flash_vla::rtx5090::load_b;
using flash_vla::rtx5090::mma_m16n8k16;

constexpr int32_t kHeadDim = 256;
constexpr int32_t kBlockM = 16;   // one mma M tile
constexpr int32_t kBlockN = 32;   // keys per tile
constexpr int32_t kWarps = 4;
constexpr int32_t kThreads = kWarps * 32;
//: Output columns each warp owns in the P@V stage: 256 / 4.
constexpr int32_t kColsPerWarp = kHeadDim / kWarps;
//: Score columns each warp owns in the Q@K^T stage: 32 / 4, one mma N tile.
constexpr int32_t kKeysPerWarp = kBlockN / kWarps;

// Shared row padding, and why each value. A fragment load has lane L reading
// row L/4 of a tile, so the ROW STRIDE in 4-byte banks decides the spread: an
// unpadded 256-bf16 row is 128 words and 128 % 32 == 0, putting all eight row
// groups on one bank. That is an 8-way conflict on every one of the twelve
// loads per mma, and the unpadded kernel measured 210 us against the split
// form's 25.
//   264 bf16 -> 132 words, 132 % 32 == 4  -> 8 groups x 4 = 32 distinct banks
//    36 bf16 ->  18 words,  18 % 32 == 18 -> odd multiple, spreads likewise
// The vt padding is 4 rather than 8 only to keep the total inside the 48 KB of
// static shared memory: 8448 + 16896 + 18432 + 2048 + 1280 = 47.3 KB.
// Tiles move in 16-byte units: 8 bf16 per thread per instruction. The row
// strides below are all multiples of 8 elements so every vector store to
// shared memory stays 16-byte aligned, and the global rows are 256 elements so
// their starts are too.
constexpr int32_t kVec = 8;
constexpr int32_t kVecs = kHeadDim / kVec;

constexpr int32_t kLdQ = kHeadDim + 8;
constexpr int32_t kLdK = kHeadDim + 8;
constexpr int32_t kLdV = kBlockN + 4;
constexpr int32_t kLdP = kBlockN + 8;

// out[q, :] = softmax_j(mask(q, j) ? Q[q,:].K[j,:] * scale : -inf) . V[j, :]
//
// The mask keeps key j for flat query row q when `q >= heads || j <= prefix`.
__global__ __launch_bounds__(kThreads) void attention_mma_kernel(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, __nv_bfloat16 *__restrict__ out,
    int32_t queries, int32_t keys, int32_t heads, int32_t prefix, float scale) {
  // 8 + 16 + 16 + 2 + 1 = 43 KB, inside the 48 KB static budget.
  __shared__ __nv_bfloat16 q_tile[kBlockM * kLdQ];
  __shared__ __nv_bfloat16 k_tile[kBlockN * kLdK];
  __shared__ __nv_bfloat16 vt_tile[kHeadDim * kLdV];  // V TRANSPOSED: [dim][key]
  __shared__ float s_tile[kBlockM * kBlockN];
  __shared__ __nv_bfloat16 p_tile[kBlockM * kLdP];
  __shared__ float row_max[kBlockM], row_sum[kBlockM], row_corr[kBlockM];

  const int32_t q0 = blockIdx.x * kBlockM;
  const int32_t rows = min(kBlockM, queries - q0);
  if (rows <= 0) return;
  const int32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

  for (int32_t i = threadIdx.x; i < kBlockM * kVecs; i += kThreads) {
    const int32_t r = i / kVecs, d = (i - r * kVecs) * kVec;
    int4 val = make_int4(0, 0, 0, 0);
    if (r < rows) val = *(const int4 *)(q + int64_t(q0 + r) * kHeadDim + d);
    *(int4 *)(&q_tile[r * kLdQ + d]) = val;
  }
  if (threadIdx.x < kBlockM) {
    row_max[threadIdx.x] = -FLT_MAX;
    row_sum[threadIdx.x] = 0.f;
  }

  // Each warp owns kColsPerWarp output columns; its accumulator is
  // 16 x 64 fp32 spread over 32 lanes, so 8 mma tiles of 4 registers each.
  constexpr int32_t kAccTiles = kColsPerWarp / 8;
  float acc[kAccTiles][4];
#pragma unroll
  for (int32_t t = 0; t < kAccTiles; ++t)
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) acc[t][i] = 0.f;

  // K and V are staged global -> register -> shared, one key tile ahead of the
  // tile being computed. Ablating the per-tile global load entirely cost 49 of
  // 87.65 us, so more than half the kernel was standing in front of that load:
  // this CTA has four warps and one warp per scheduler, so nothing else was
  // available to issue into the latency. Prefetching into registers puts the
  // next tile's loads in flight before this tile's mma work instead.
  constexpr int32_t kStage = kBlockN * kVecs / kThreads;
  int4 kreg[kStage], vreg[kStage];
  auto stage_tile = [&](int32_t j) {
    const int32_t m = min(kBlockN, keys - j);
#pragma unroll
    for (int32_t t = 0; t < kStage; ++t) {
      const int32_t i = threadIdx.x + t * kThreads;
      const int32_t c = i / kVecs, d = (i - c * kVecs) * kVec;
      kreg[t] = make_int4(0, 0, 0, 0);
      vreg[t] = make_int4(0, 0, 0, 0);
      if (c < m) {
        kreg[t] = *(const int4 *)(k + int64_t(j + c) * kHeadDim + d);
        vreg[t] = *(const int4 *)(v + int64_t(j + c) * kHeadDim + d);
      }
    }
  };
  stage_tile(0);
  __syncthreads();

  for (int32_t j0 = 0; j0 < keys; j0 += kBlockN) {
    const int32_t n = min(kBlockN, keys - j0);
    // Publish the staged tile. The loop's last barrier is what makes this safe:
    // every warp has finished reading the previous tile out of these buffers.
#pragma unroll
    for (int32_t t = 0; t < kStage; ++t) {
      const int32_t i = threadIdx.x + t * kThreads;
      const int32_t c = i / kVecs, d = (i - c * kVecs) * kVec;
      *(int4 *)(&k_tile[c * kLdK + d]) = kreg[t];
      // V is transposed on the way in, so the store side scatters: the P@V
      // mma needs B indexed [dim][key] (.col), and only the global side can be
      // made contiguous. Eight 2-byte shared stores, one 16-byte global load.
      const __nv_bfloat16 *e = (const __nv_bfloat16 *)&vreg[t];
#pragma unroll
      for (int32_t w = 0; w < kVec; ++w) vt_tile[(d + w) * kLdV + c] = e[w];
    }
    __syncthreads();
    if (j0 + kBlockN < keys) stage_tile(j0 + kBlockN);

    // Q @ K^T. Warp w takes score columns [w*8, w*8+8) -- one mma N tile -- and
    // walks the whole 256-deep head dimension.
    {
      // Four independent accumulators, summed at the end. One chain would be
      // 16 mma each waiting on the previous instruction's result, and with a
      // single warp per scheduler there is nothing else to issue into that
      // latency. Partial sums are exact here: each chain covers a disjoint
      // quarter of the head dimension, so this only reassociates the reduction.
      constexpr int32_t kChains = 4;
      float sp[kChains][4] = {};
      uint32_t a[kChains][4], b[kChains][2];
#pragma unroll
      for (int32_t kd = 0; kd < kHeadDim; kd += 16 * kChains) {
#pragma unroll
        for (int32_t c = 0; c < kChains; ++c) {
          load_a(a[c], q_tile, kLdQ, kd + c * 16, lane);
          load_b(b[c], k_tile, kLdK, warp * kKeysPerWarp, kd + c * 16, lane);
        }
#pragma unroll
        for (int32_t c = 0; c < kChains; ++c) mma_m16n8k16(sp[c], a[c], b[c]);
      }
      float s[4];
#pragma unroll
      for (int32_t i = 0; i < 4; ++i)
        s[i] = (sp[0][i] + sp[1][i]) + (sp[2][i] + sp[3][i]);
      const int32_t r0 = lane >> 2, c0 = (lane & 3) * 2;
      const int32_t col = warp * 8 + c0;
#pragma unroll
      for (int32_t i = 0; i < 4; ++i) {
        const int32_t r = r0 + (i >> 1) * 8, c = col + (i & 1);
        const int32_t key = j0 + c;
        const bool keep = (q0 + r < queries) && (c < n)
                       && ((q0 + r >= heads) || (key <= prefix));
        s_tile[r * kBlockN + c] = keep ? s[i] * scale : -FLT_MAX;
      }
    }
    __syncthreads();

    // Online softmax over the tile, eight threads per row. Thread (r, j) owns
    // columns j, j+8, j+16, j+24, and the eight threads of a row are an aligned
    // lane octet of one warp, so the two row reductions are shuffles inside
    // that octet rather than a serial scan. The form this replaced had 16
    // threads walk 32 columns twice while the other 112 waited at the barrier;
    // with one warp per scheduler that scan was exposed latency, not work.
    {
      constexpr int32_t kColThreads = 8;
      constexpr int32_t kTileCols = kBlockN / kColThreads;
      const int32_t r = threadIdx.x / kColThreads;
      const int32_t j = threadIdx.x % kColThreads;
      //: Rows past the query tail carry no probability mass and no correction.
      const bool live = r < rows;
      const float prev = live ? row_max[r] : -FLT_MAX;
      const float prev_sum = live ? row_sum[r] : 0.f;

      float vals[kTileCols];
      float m = prev;
#pragma unroll
      for (int32_t t = 0; t < kTileCols; ++t) {
        const int32_t c = j + t * kColThreads;
        vals[t] = (live && c < n) ? s_tile[r * kBlockN + c] : -FLT_MAX;
        m = fmaxf(m, vals[t]);
      }
      // Every lane of the warp reaches these, so the full mask is correct even
      // though each octet reduces independently.
#pragma unroll
      for (int32_t d = 1; d < kColThreads; d <<= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));

      const float corr = (prev == -FLT_MAX) ? 0.f : __expf(prev - m);
      float sum = 0.f;
#pragma unroll
      for (int32_t t = 0; t < kTileCols; ++t) {
        const float e = (vals[t] == -FLT_MAX) ? 0.f : __expf(vals[t] - m);
        p_tile[r * kLdP + j + t * kColThreads] = __float2bfloat16(e);
        sum += e;
      }
#pragma unroll
      for (int32_t d = 1; d < kColThreads; d <<= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, d);

      if (j == 0) {
        row_max[r] = m;
        row_sum[r] = prev_sum * corr + sum;
        row_corr[r] = live ? corr : 0.f;
      }
    }
    __syncthreads();

    // P @ V. Warp w takes output columns [w*64, w*64+64), which is kAccTiles
    // mma N tiles, over the kBlockN keys of this tile.
    {
      // Rescale the running accumulator by this tile's softmax correction.
      // A lane's rows are fixed by the mma D layout: L/4 and L/4+8.
      const int32_t r0 = lane >> 2;
      const float c_lo = row_corr[r0], c_hi = row_corr[r0 + 8];
#pragma unroll
      for (int32_t t = 0; t < kAccTiles; ++t) {
        acc[t][0] *= c_lo; acc[t][1] *= c_lo;
        acc[t][2] *= c_hi; acc[t][3] *= c_hi;
      }
      // The kAccTiles accumulators are already independent; hoisting their B
      // fragments above the mma block lets the shared loads pipeline instead
      // of each one standing immediately in front of the instruction that
      // consumes it.
      uint32_t a[4], b[kAccTiles][2];
#pragma unroll
      for (int32_t kd = 0; kd < kBlockN; kd += 16) {
        load_a(a, p_tile, kLdP, kd, lane);
#pragma unroll
        for (int32_t t = 0; t < kAccTiles; ++t)
          load_b(b[t], vt_tile, kLdV, warp * kColsPerWarp + t * 8, kd, lane);
#pragma unroll
        for (int32_t t = 0; t < kAccTiles; ++t) mma_m16n8k16(acc[t], a, b[t]);
      }
    }
    __syncthreads();
  }

  const int32_t r0 = lane >> 2, c0 = (lane & 3) * 2;
  const float inv_lo = (row_sum[r0] > 0.f) ? __frcp_rn(row_sum[r0]) : 0.f;
  const float inv_hi = (row_sum[r0 + 8] > 0.f) ? __frcp_rn(row_sum[r0 + 8]) : 0.f;
#pragma unroll
  for (int32_t t = 0; t < kAccTiles; ++t) {
    const int32_t col = warp * kColsPerWarp + t * 8 + c0;
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      const int32_t r = r0 + (i >> 1) * 8;
      if (q0 + r >= queries) continue;
      const float inv = (i >> 1) ? inv_hi : inv_lo;
      out[int64_t(q0 + r) * kHeadDim + col + (i & 1)] =
          __float2bfloat16(acc[t][i] * inv);
    }
  }
}

}  // namespace

extern "C" int expert_attention_mma_launch(const void *q, const void *k,
                                           const void *v, void *out,
                                           int queries, int keys, int head_dim,
                                           int heads, int prefix, float scale,
                                           void *stream) {
  if (head_dim != kHeadDim) return cudaErrorInvalidValue;
  const int32_t grid = (queries + kBlockM - 1) / kBlockM;
  attention_mma_kernel<<<grid, kThreads, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k,
      (const __nv_bfloat16 *)v, (__nv_bfloat16 *)out, queries, keys, heads,
      prefix, scale);
  return (int)cudaGetLastError();
}
