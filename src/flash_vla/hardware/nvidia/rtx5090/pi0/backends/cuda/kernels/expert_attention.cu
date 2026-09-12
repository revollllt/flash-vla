// Pi0's action-expert attention as one kernel, on sm_120.
//
// The torch route spells this as about twelve launches -- an fp32 QK matmul, two
// `arange`s and three comparisons that rebuild the mask on EVERY call, a
// `masked_fill`, a softmax, a cast and a second matmul -- and materialises the
// (408, 819) score matrix twice. The profile attributes 6.014 ms to it against a
// 0.751 ms ceiling, the worst ratio in the model at 800%.
//
// Three things are wrong with that and this kernel fixes all three:
//
//   * the mask is a pure function of the shape and is rebuilt 180 times per
//     forward. Here it is two comparisons on indices the kernel already has.
//   * the QK product runs in fp32, four times the bytes of the bf16 the inputs
//     are stored in, for a reduction that is accumulated in fp32 either way.
//   * the scores round-trip to global memory between the three stages. Here
//     they never leave shared memory, through the usual online softmax.
//
// `F.scaled_dot_product_attention` with a precomputed mask was measured first,
// as the cheap answer: 90.3 us against the torch chain's 73.0. It is SLOWER at
// this shape, so a library call was not the way out.
//
// Sizing, from the measured table. The work is 342 MFLOP per call, which the
// tensor core would do in 1.4 us and the fp32 CUDA cores in 2.7 us at full
// occupancy [mma.rate.sm.bf16] -- a factor of two apart, so this first version
// stays on CUDA cores and keeps the structure simple. What decides the time is
// parallelism against L2 traffic: BLOCK_M query rows per CTA means K and V are
// re-read ceil(408 / BLOCK_M) times, and at 838 KB the whole K/V working set
// sits in this part's 96 MB L2, so those re-reads are L2 hits rather than DRAM.
// BLOCK_M is a launch parameter so the trade can be swept rather than guessed.
#include <cstdint>
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int32_t kThreads = 256;
constexpr int32_t kHeadDim = 256; // Pi0's expert head dimension, compile-time

// Shared-memory row padding. The transposed K tile is written with consecutive
// threads holding consecutive `d`, so its row stride decides the bank spread of
// the WRITE: at a stride of kBlockN bf16 (64 B, 16 words) that is a 16-way
// conflict, and ncu measured 5.67M shared bank conflicts on the unpadded form.
// Padding by 2 bf16 makes the stride 17 words, and 17 is coprime with the 32
// banks, so 32 consecutive rows land on 32 distinct banks.
constexpr int32_t kPadBf16 = 2;
//: The score tile is read one row per thread in the softmax phase, so its
//: float stride needs the same treatment: 33 words, 33 % 32 == 1.
constexpr int32_t kPadF32 = 1;

// out[q, :] = softmax_j(mask(q, j) ? Q[q,:].K[j,:] * scale : -inf) . V[j, :]
//
// The mask keeps key j for flat query row q when `q >= heads || j <= prefix`:
// the state token's heads attend only to the prefix, the action tokens to
// everything. `q` is flat over (token, head) because the expert is multi-query.
template <int32_t BLOCK_M, int32_t kBlockN>
__global__ void expert_attention_kernel(const __nv_bfloat16 *__restrict__ q,
                                        const __nv_bfloat16 *__restrict__ k,
                                        const __nv_bfloat16 *__restrict__ v,
                                        __nv_bfloat16 *__restrict__ out,
                                        int32_t queries, int32_t keys,
                                        int32_t heads, int32_t prefix,
                                        float scale) {
  // Q stays resident; K and V stream. These are STATIC shared arrays, so the
  // budget is 48 KB, not the 99 KB a block can opt in to
  // [smem.bytes.cta.max] -- the opt-in needs dynamic shared memory and a
  // cudaFuncSetAttribute call, which is a later change if the sweep says wider
  // tiles pay. Every (BLOCK_M, kBlockN) pair below is sized to fit 48 KB:
  //   16 x 32 -> 8 + 16 + 16 + 2 = 42 KB
  //    8 x 32 -> 4 + 16 + 16 + 1 = 37 KB
  //   32 x 16 -> 16 + 8 + 8 + 2  = 34 KB
  __shared__ __nv_bfloat16 q_tile[BLOCK_M * kHeadDim];
  // K is stored TRANSPOSED, [d][key]. In the score loop a warp holds one query
  // row and 32 consecutive keys, so this makes its K read consecutive addresses
  // instead of 32 rows separated by 512 B -- that stride is 128 words, and
  // 128 % 32 == 0 puts every lane on one bank. The first version of this kernel
  // did exactly that and measured 1365 us against the torch chain's 70.
  __shared__ __nv_bfloat16 k_tile[kHeadDim * (kBlockN + kPadBf16)];
  __shared__ __nv_bfloat16 v_tile[kBlockN * kHeadDim];
  __shared__ float s_tile[BLOCK_M * (kBlockN + kPadF32)];
  __shared__ float row_max[BLOCK_M], row_sum[BLOCK_M], row_corr[BLOCK_M];

  const int32_t q0 = blockIdx.x * BLOCK_M;
  const int32_t rows = min(BLOCK_M, queries - q0);
  if (rows <= 0) return;

  for (int32_t i = threadIdx.x; i < rows * kHeadDim; i += kThreads)
    q_tile[i] = q[int64_t(q0) * kHeadDim + i];
  if (threadIdx.x < BLOCK_M) {
    row_max[threadIdx.x] = -FLT_MAX;
    row_sum[threadIdx.x] = 0.f;
  }

  // Output accumulator: thread t owns column t of every row, so the reduction
  // over keys below is thread-local and no cross-thread traffic is needed.
  float acc[BLOCK_M];
#pragma unroll
  for (int32_t r = 0; r < BLOCK_M; ++r) acc[r] = 0.f;
  __syncthreads();

  for (int32_t j0 = 0; j0 < keys; j0 += kBlockN) {
    const int32_t n = min(kBlockN, keys - j0);
    for (int32_t i = threadIdx.x; i < n * kHeadDim; i += kThreads) {
      const int32_t c = i / kHeadDim, d = i - c * kHeadDim;
      const __nv_bfloat16 kv = k[int64_t(j0) * kHeadDim + i];
      k_tile[d * (kBlockN + kPadBf16) + c] = kv;   // transposed on the way in
      v_tile[i] = v[int64_t(j0) * kHeadDim + i];
    }
    __syncthreads();

    // Scores for this tile. Thread layout is (row, key) with the key index
    // fastest, so each warp covers `kBlockN` consecutive keys of one row: the
    // q_tile read is a broadcast and the k_tile read is contiguous.
    // The bound is rows * kBlockN, not rows * n: the decode below divides by
    // kBlockN, and on the last partial tile (819 % 32 = 19 keys) a bound of
    // rows * n leaves some (row, key) pairs unvisited. That mismatch measured
    // cos 0.990 instead of 0.999995.
    for (int32_t idx = threadIdx.x; idx < rows * kBlockN; idx += kThreads) {
      const int32_t r = idx / kBlockN, c = idx - r * kBlockN;
      if (c >= n) continue;
      const int32_t key = j0 + c;
      float dot = 0.f;
#pragma unroll 8
      for (int32_t d = 0; d < kHeadDim; ++d)
        dot += __bfloat162float(q_tile[r * kHeadDim + d])
             * __bfloat162float(k_tile[d * (kBlockN + kPadBf16) + c]);
      const bool keep = (q0 + r >= heads) || (key <= prefix);
      s_tile[r * (kBlockN + kPadF32) + c] = keep ? dot * scale : -FLT_MAX;
    }
    __syncthreads();

    // Online softmax: rescale the running sum and the accumulator by the new
    // maximum, then add this tile's contribution.
    if (threadIdx.x < rows) {
      const int32_t r = threadIdx.x;
      float m = row_max[r];
      for (int32_t c = 0; c < n; ++c) m = fmaxf(m, s_tile[r * (kBlockN + kPadF32) + c]);
      const float corr = (row_max[r] == -FLT_MAX) ? 0.f : __expf(row_max[r] - m);
      float sum = row_sum[r] * corr;
      for (int32_t c = 0; c < n; ++c) {
        const float e = (s_tile[r * (kBlockN + kPadF32) + c] == -FLT_MAX)
                            ? 0.f : __expf(s_tile[r * (kBlockN + kPadF32) + c] - m);
        s_tile[r * (kBlockN + kPadF32) + c] = e;
        sum += e;
      }
      row_max[r] = m;
      row_sum[r] = sum;
      row_corr[r] = corr;
    }
    __syncthreads();

    // Accumulate P @ V. Thread t owns column t of every row. The row loop is
    // unrolled over the COMPILE-TIME BLOCK_M with a guard rather than run to
    // the runtime `rows`: a runtime index into `acc` forces it out of registers
    // and into local memory, which is what made the first version spill.
    const int32_t d = threadIdx.x;
    if (d < kHeadDim) {
#pragma unroll
      for (int32_t r = 0; r < BLOCK_M; ++r) {
        if (r >= rows) break;
        float a = acc[r] * row_corr[r];
        for (int32_t c = 0; c < n; ++c)
          a += s_tile[r * (kBlockN + kPadF32) + c]
             * __bfloat162float(v_tile[c * kHeadDim + d]);
        acc[r] = a;
      }
    }
    __syncthreads();
  }

  const int32_t d = threadIdx.x;
  if (d < kHeadDim) {
#pragma unroll
    for (int32_t r = 0; r < BLOCK_M; ++r) {
      if (r >= rows) break;
      const float inv = (row_sum[r] > 0.f) ? __frcp_rn(row_sum[r]) : 0.f;
      out[int64_t(q0 + r) * kHeadDim + d] = __float2bfloat16(acc[r] * inv);
    }
  }
}

}  // namespace

extern "C" {

// `block_m` selects the query rows per CTA. Larger blocks re-read K and V fewer
// times; smaller blocks put more CTAs on the machine. Both matter here and the
// balance is swept, not assumed.
int expert_attention_launch(const void *q, const void *k, const void *v,
                            void *out, int queries, int keys, int head_dim,
                            int heads, int prefix, float scale, int block_m,
                            void *stream) {
  if (head_dim != kHeadDim) return cudaErrorInvalidValue;
  cudaStream_t s = (cudaStream_t)stream;
  const auto *qb = (const __nv_bfloat16 *)q;
  const auto *kb = (const __nv_bfloat16 *)k;
  const auto *vb = (const __nv_bfloat16 *)v;
  auto *ob = (__nv_bfloat16 *)out;
#define LAUNCH(BM, BN)                                                         \
  case BM: {                                                                   \
    const int32_t grid = (queries + (BM) - 1) / (BM);                          \
    expert_attention_kernel<BM, BN><<<grid, kThreads, 0, s>>>(                 \
        qb, kb, vb, ob, queries, keys, heads, prefix, scale);                  \
    break;                                                                     \
  }
  switch (block_m) {
    LAUNCH(8, 32)
    LAUNCH(16, 32)
    LAUNCH(32, 16)
    default:
      return cudaErrorInvalidValue;
  }
#undef LAUNCH
  return (int)cudaGetLastError();
}

}  // extern "C"

// ---------------------------------------------------------------------------
// The masked softmax alone, which is what actually ships.
//
// The fused kernel above is correct and SLOW -- 201 us against the torch
// chain's 70 at Pi0's shape. ncu says why: 32.09M instructions for 342 MFLOP,
// because a 256-deep dot product per thread out of shared memory costs two
// loads and an FFMA per two FLOP, where one `mma.sync` does 4096 FLOP in one
// instruction [mma.rate.sm.bf16]. A CUDA-core flash kernel cannot win here; a
// tensor-core one would need ldmatrix-fed mma fragments and is a separate
// piece of work. It is kept so the negative result can be re-run.
//
// What ships instead leaves both GEMMs on cuBLAS, which already reaches the
// tensor core, and replaces only the glue between them: the two `arange`s, the
// three comparisons, the `masked_fill`, the softmax and the cast -- about seven
// launches -- with one. The mask is recomputed from indices the kernel already
// has rather than materialised.
namespace {

// probs[r, :] = softmax_j(mask(r, j) ? scores[r, j] * scale : -inf)
//
// One CTA per query row, block-wide max and sum through shuffles. `scores` is
// bf16 as cuBLAS wrote it; the reduction and the exponential are fp32.
__global__ void masked_softmax_kernel(const __nv_bfloat16 *__restrict__ scores,
                                      __nv_bfloat16 *__restrict__ probs,
                                      int32_t queries, int32_t keys,
                                      int32_t heads, int32_t prefix,
                                      float scale) {
  __shared__ float reduction[32];
  const int32_t r = blockIdx.x;
  if (r >= queries) return;
  const int64_t base = int64_t(r) * keys;
  // The state token's heads see only the prefix; the action tokens see all.
  const bool all_keys = (r >= heads);

  float m = -FLT_MAX;
  for (int32_t j = threadIdx.x; j < keys; j += blockDim.x)
    if (all_keys || j <= prefix)
      m = fmaxf(m, __bfloat162float(scores[base + j]) * scale);

  const int32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int32_t o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  if (lane == 0) reduction[warp] = m;
  __syncthreads();
  const int32_t warps = (blockDim.x + 31) >> 5;
  m = (threadIdx.x < warps) ? reduction[threadIdx.x] : -FLT_MAX;
#pragma unroll
  for (int32_t o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  __shared__ float row_max;
  if (threadIdx.x == 0) row_max = m;
  __syncthreads();
  const float mx = row_max;

  float sum = 0.f;
  for (int32_t j = threadIdx.x; j < keys; j += blockDim.x) {
    const float e = (all_keys || j <= prefix)
        ? __expf(__bfloat162float(scores[base + j]) * scale - mx) : 0.f;
    probs[base + j] = __float2bfloat16(e);
    sum += e;
  }
#pragma unroll
  for (int32_t o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  if (lane == 0) reduction[warp] = sum;
  __syncthreads();
  sum = (threadIdx.x < warps) ? reduction[threadIdx.x] : 0.f;
#pragma unroll
  for (int32_t o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  __shared__ float row_inv;
  if (threadIdx.x == 0) row_inv = (sum > 0.f) ? __frcp_rn(sum) : 0.f;
  __syncthreads();
  const float inv = row_inv;

  for (int32_t j = threadIdx.x; j < keys; j += blockDim.x)
    probs[base + j] = __float2bfloat16(__bfloat162float(probs[base + j]) * inv);
}

}  // namespace

extern "C" int expert_masked_softmax_launch(const void *scores, void *probs,
                                            int queries, int keys, int heads,
                                            int prefix, float scale,
                                            void *stream) {
  masked_softmax_kernel<<<queries, 256, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)scores, (__nv_bfloat16 *)probs, queries, keys,
      heads, prefix, scale);
  return (int)cudaGetLastError();
}
