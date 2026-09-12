// Pi0's action-expert attention as one tensor-core kernel, on sm_120.
//
// The CUDA-core attempt in `expert_attention.cu` is correct and 8x slower than
// splitting the work across cuBLAS and a softmax kernel. ncu settled why: 32.09M
// instructions for 342 MFLOP, because a dot product out of shared memory costs
// two loads and an FFMA per two FLOP. Only `mma.sync` clears that
// [mma.rate.sm.bf16], and this is the `mma.sync` form.
//
// An earlier version of this kernel measured 79.5 us and was recorded as a
// second negative. That reading did not survive: it was written before the
// fused QKV projection established what this part actually needs, and it used
// none of it. Three things it was missing, each worth its own factor there:
//
//   - fragments built by `ldmatrix` rather than twelve scalar shared loads
//   - one job per warp, so a scheduler has more than one warp to issue from
//   - operands staged in their natural order, because the transposed form's
//     scatter store is an 8-way bank conflict that no padding removes
//
// The last one is free here. `mma`'s B operand is `.col`, indexed [n][k]:
// Q@K^T wants [key][dim], which is K's own layout, and P@V wants [dim][key],
// which `ldmatrix.trans` produces FROM [key][dim]. So both stage naturally and
// neither pays a transpose.
//
// Shape: 51 tokens x 8 query heads = 408 flat queries, 768 prefix + 51 suffix =
// 819 keys, head dimension 256, multi-query so all eight heads share one KV.
//
// Splitting. 408 queries is 26 mma M tiles, and 26 CTAs on a 170-SM part is
// 15% of the machine. The key axis is the only other place parallelism can
// come from, so the kernel takes a split count: CTA (tile, s) walks its own
// slice of the keys and writes an UNNORMALIZED partial plus that slice's
// running max and sum, and a second kernel merges the slices by the usual
// log-sum-exp. The cost is that round trip -- S x 408 x 256 fp32 -- and at
// 2.5 MB for S=6 it stays in L2, which this machine serves at a measured
// 4879 GB/s rather than [ld.bw.dev.dram]'s 1524.
//
// Tiling. BLOCK_M is 16 because that is one mma M tile and 408 queries is all
// there is. BLOCK_N is 64, which is eight mma N tiles and therefore eight
// warps with one tile each in the Q@K^T stage, and which halves the key-tile
// count to 13. That needs 82.6 KB of shared memory, over the 48 KB static
// limit, so the tiles are carved out of a dynamic allocation
// [smem.bytes.cta.max].
#include <algorithm>
#include <cstdint>
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "mma_bf16.cuh"
#include "pdl.cuh"

//: Owned by expert_pointwise.cu, which holds this library's PDL switch.
bool flash_vla_pdl_enabled();

namespace {

using flash_vla::rtx5090::ldmatrix_a;
using flash_vla::rtx5090::pdl_trigger;
using flash_vla::rtx5090::pdl_trigger_at;
using flash_vla::rtx5090::pdl_wait;
using flash_vla::rtx5090::ldmatrix_b;
using flash_vla::rtx5090::ldmatrix_b_trans;
using flash_vla::rtx5090::mma_m16n8k16;

constexpr int32_t kHeadDim = 256;
constexpr int32_t kBlockM = 16;   // one mma M tile
constexpr int32_t kBlockN = 64;   // keys per tile, eight mma N tiles
constexpr int32_t kWarps = 8;
constexpr int32_t kThreads = kWarps * 32;
constexpr int32_t kVec = 8;                       // bf16 per 16-byte access
constexpr int32_t kVecs = kHeadDim / kVec;
//: Output columns each warp owns in the P@V stage: 256 / 8, four mma N tiles.
constexpr int32_t kColsPerWarp = kHeadDim / kWarps;
constexpr int32_t kAccTiles = kColsPerWarp / 8;
//: Threads sharing one score row in the softmax, an aligned lane group.
constexpr int32_t kRowThreads = kThreads / kBlockM;
constexpr int32_t kTileCols = kBlockN / kRowThreads;

// Row strides. Every one is a multiple of 8 bf16 so that an `ldmatrix` row
// address is 16-byte aligned, and every one is 4 (mod 32) in 4-byte banks so
// the eight rows an `ldmatrix` gathers land on eight different bank groups and
// cover all 32.
constexpr int32_t kLdQ = kHeadDim + 8;   // 264 elements, 132 words, 132 % 32 == 4
constexpr int32_t kLdK = kHeadDim + 8;
constexpr int32_t kLdV = kHeadDim + 8;
constexpr int32_t kLdP = kBlockN + 8;    // 72 elements, 36 words, 36 % 32 == 4

constexpr int32_t kQBytes = kBlockM * kLdQ * 2;
constexpr int32_t kKBytes = kBlockN * kLdK * 2;
constexpr int32_t kVBytes = kBlockN * kLdV * 2;
constexpr int32_t kSBytes = kBlockM * kBlockN * 4;
constexpr int32_t kPBytes = kBlockM * kLdP * 2;
constexpr int32_t kRowBytes = kBlockM * 4 * 3;
constexpr int32_t kSmemBytes =
    kQBytes + kKBytes + kVBytes + kSBytes + kPBytes + kRowBytes;

// out[q, :] = softmax_j(mask(q, j) ? Q[q,:].K[j,:] * scale : -inf) . V[j, :]
//
// The mask keeps key j for flat query row q when `q >= heads || j <= prefix`.
__global__ __launch_bounds__(kThreads) void attention_mma_kernel(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, __nv_bfloat16 *__restrict__ out,
    int32_t queries, int32_t keys, int32_t heads, int32_t prefix, float scale,
    float *__restrict__ part_o, float *__restrict__ part_m,
    float *__restrict__ part_l, int32_t splits, int32_t chunk) {
  extern __shared__ __align__(16) char smem[];
  __nv_bfloat16 *q_tile = reinterpret_cast<__nv_bfloat16 *>(smem);
  __nv_bfloat16 *k_tile = reinterpret_cast<__nv_bfloat16 *>(smem + kQBytes);
  __nv_bfloat16 *v_tile =
      reinterpret_cast<__nv_bfloat16 *>(smem + kQBytes + kKBytes);
  float *s_tile =
      reinterpret_cast<float *>(smem + kQBytes + kKBytes + kVBytes);
  __nv_bfloat16 *p_tile = reinterpret_cast<__nv_bfloat16 *>(
      smem + kQBytes + kKBytes + kVBytes + kSBytes);
  float *row_max = reinterpret_cast<float *>(
      smem + kQBytes + kKBytes + kVBytes + kSBytes + kPBytes);
  float *row_sum = row_max + kBlockM;
  float *row_corr = row_sum + kBlockM;

  //: Derived: Q, K and V are all the QKV projection's output, so there is no
  //: producer-independent work to hoist and the wait is the first statement.
  pdl_wait();

  const int32_t q0 = blockIdx.x * kBlockM;
  const int32_t rows = min(kBlockM, queries - q0);
  if (rows <= 0) return;
  const int32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  //: This CTA's slice of the key axis. `chunk` is a multiple of kBlockN, so a
  //: slice boundary never falls inside a tile.
  const int32_t split = blockIdx.y;
  const int32_t j_beg = split * chunk;
  const int32_t j_end = min(keys, j_beg + chunk);

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

  //: This warp's slice of the output: kColsPerWarp columns, kAccTiles mma
  //: tiles of 4 registers each.
  float acc[kAccTiles][4];
#pragma unroll
  for (int32_t t = 0; t < kAccTiles; ++t)
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) acc[t][i] = 0.f;

  // K and V are staged global -> register -> shared one key tile ahead, both
  // in their natural [key][dim] order: one 16-byte load and one 16-byte store
  // each, no transpose on either side.
  constexpr int32_t kStage = kBlockN * kVecs / kThreads;
  int4 kreg[kStage], vreg[kStage];
  auto stage_tile = [&](int32_t j) {
    const int32_t m = min(kBlockN, j_end - j);
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
  stage_tile(j_beg);
  __syncthreads();

  for (int32_t j0 = j_beg; j0 < j_end; j0 += kBlockN) {
    const int32_t n = min(kBlockN, j_end - j0);
#pragma unroll
    for (int32_t t = 0; t < kStage; ++t) {
      const int32_t i = threadIdx.x + t * kThreads;
      const int32_t c = i / kVecs, d = (i - c * kVecs) * kVec;
      *(int4 *)(&k_tile[c * kLdK + d]) = kreg[t];
      *(int4 *)(&v_tile[c * kLdV + d]) = vreg[t];
    }
    __syncthreads();
    if (j0 + kBlockN < j_end) stage_tile(j0 + kBlockN);

    // Q @ K^T. Warp w takes score columns [w*8, w*8+8) -- one mma N tile --
    // and walks the whole 256-deep head dimension. Four independent chains,
    // summed at the end: one would be sixteen mma each waiting on the last.
    {
      constexpr int32_t kChains = 4;
      float sp[kChains][4] = {};
      uint32_t a[kChains][4], b[kChains][2];
#pragma unroll
      for (int32_t kd = 0; kd < kHeadDim; kd += 16 * kChains) {
#pragma unroll
        for (int32_t c = 0; c < kChains; ++c) {
          ldmatrix_a(a[c], q_tile, kLdQ, kd + c * 16, lane);
          ldmatrix_b(b[c], k_tile, kLdK, warp * 8, kd + c * 16, lane);
        }
#pragma unroll
        for (int32_t c = 0; c < kChains; ++c) mma_m16n8k16(sp[c], a[c], b[c]);
      }
      const int32_t r0 = lane >> 2, c0 = (lane & 3) * 2;
      const int32_t col = warp * 8 + c0;
#pragma unroll
      for (int32_t i = 0; i < 4; ++i) {
        const int32_t r = r0 + (i >> 1) * 8, c = col + (i & 1);
        const int32_t key = j0 + c;
        const bool keep = (q0 + r < queries) && (c < n)
                       && ((q0 + r >= heads) || (key <= prefix));
        const float s = (sp[0][i] + sp[1][i]) + (sp[2][i] + sp[3][i]);
        s_tile[r * kBlockN + c] = keep ? s * scale : -FLT_MAX;
      }
    }
    __syncthreads();

    // Online softmax, kRowThreads threads per row, reduced by shuffles inside
    // an aligned lane group rather than by one thread walking the row.
    {
      const int32_t r = threadIdx.x / kRowThreads;
      const int32_t j = threadIdx.x % kRowThreads;
      const bool live = r < rows;
      const float prev = live ? row_max[r] : -FLT_MAX;
      const float prev_sum = live ? row_sum[r] : 0.f;

      float vals[kTileCols];
      float m = prev;
#pragma unroll
      for (int32_t t = 0; t < kTileCols; ++t) {
        const int32_t c = j + t * kRowThreads;
        vals[t] = (live && c < n) ? s_tile[r * kBlockN + c] : -FLT_MAX;
        m = fmaxf(m, vals[t]);
      }
#pragma unroll
      for (int32_t d = 1; d < kRowThreads; d <<= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));

      const float corr = (prev == -FLT_MAX) ? 0.f : __expf(prev - m);
      float sum = 0.f;
#pragma unroll
      for (int32_t t = 0; t < kTileCols; ++t) {
        const float e = (vals[t] == -FLT_MAX) ? 0.f : __expf(vals[t] - m);
        p_tile[r * kLdP + j + t * kRowThreads] = __float2bfloat16(e);
        sum += e;
      }
#pragma unroll
      for (int32_t d = 1; d < kRowThreads; d <<= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, d);

      if (j == 0) {
        row_max[r] = m;
        row_sum[r] = prev_sum * corr + sum;
        row_corr[r] = live ? corr : 0.f;
      }
    }
    __syncthreads();

    // P @ V. Warp w takes output columns [w*32, w*32+32), which is kAccTiles
    // mma N tiles, over the kBlockN keys of this tile. V is read straight out
    // of its natural [key][dim] tile by `ldmatrix.trans`.
    {
      const int32_t r0 = lane >> 2;
      const float c_lo = row_corr[r0], c_hi = row_corr[r0 + 8];
#pragma unroll
      for (int32_t t = 0; t < kAccTiles; ++t) {
        acc[t][0] *= c_lo; acc[t][1] *= c_lo;
        acc[t][2] *= c_hi; acc[t][3] *= c_hi;
      }
      uint32_t a[4], b[kAccTiles][2];
#pragma unroll
      for (int32_t kk = 0; kk < kBlockN; kk += 16) {
        ldmatrix_a(a, p_tile, kLdP, kk, lane);
#pragma unroll
        for (int32_t t = 0; t < kAccTiles; ++t)
          ldmatrix_b_trans(b[t], v_tile, kLdV, warp * kColsPerWarp + t * 8, kk,
                           lane);
#pragma unroll
        for (int32_t t = 0; t < kAccTiles; ++t) mma_m16n8k16(acc[t], a, b[t]);
      }
    }
    __syncthreads();
  }

  //: The key loop is done and only the epilogue remains.
  pdl_trigger_at<FLASH_VLA_PDL_TRIGGER_EARLY>();

  const int32_t r0 = lane >> 2, c0 = (lane & 3) * 2;

  if (splits > 1) {
    // Hand the slice on unnormalized, with the numbers the merge needs. A
    // slice that starts past the last key contributes nothing and says so
    // with sum 0, which the merge skips.
    const int64_t base = int64_t(split) * queries * kHeadDim;
#pragma unroll
    for (int32_t t = 0; t < kAccTiles; ++t) {
      const int32_t col = warp * kColsPerWarp + t * 8 + c0;
#pragma unroll
      for (int32_t i = 0; i < 4; ++i) {
        const int32_t r = r0 + (i >> 1) * 8;
        if (q0 + r >= queries) continue;
        part_o[base + int64_t(q0 + r) * kHeadDim + col + (i & 1)] = acc[t][i];
      }
    }
    if (threadIdx.x < kBlockM && q0 + threadIdx.x < queries) {
      const int32_t row = int32_t(split) * queries + q0 + threadIdx.x;
      part_m[row] = row_max[threadIdx.x];
      part_l[row] = row_sum[threadIdx.x];
    }
    pdl_trigger_at<FLASH_VLA_PDL_TRIGGER_LAST>();
    return;
  }

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

// Merge the splits: standard log-sum-exp over slices, one CTA per query row.
//
//   m = max_s m_s;  w_s = exp(m_s - m);  out = sum_s w_s o_s / sum_s w_s l_s
__global__ __launch_bounds__(kHeadDim) void attention_merge_kernel(
    const float *__restrict__ part_o, const float *__restrict__ part_m,
    const float *__restrict__ part_l, __nv_bfloat16 *__restrict__ out,
    int32_t queries, int32_t splits) {
  //: Derived: the partials are the split kernel's output.
  pdl_wait();
  const int32_t row = blockIdx.x;
  if (row >= queries) return;
  const int32_t d = threadIdx.x;

  float m = -FLT_MAX;
  for (int32_t s = 0; s < splits; ++s)
    if (part_l[int64_t(s) * queries + row] > 0.f)
      m = fmaxf(m, part_m[int64_t(s) * queries + row]);

  float num = 0.f, den = 0.f;
  for (int32_t s = 0; s < splits; ++s) {
    const float l = part_l[int64_t(s) * queries + row];
    if (l <= 0.f) continue;
    const float w = __expf(part_m[int64_t(s) * queries + row] - m);
    num += w * part_o[(int64_t(s) * queries + row) * kHeadDim + d];
    den += w * l;
  }
  out[int64_t(row) * kHeadDim + d] =
      __float2bfloat16(den > 0.f ? num / den : 0.f);
  pdl_trigger();
}

}  // namespace

extern "C" int expert_attention_mma_workspace(int queries, int splits,
                                              long long *floats) {
  //: partial outputs, then the per-slice max and sum.
  *floats = int64_t(splits) * queries * (kHeadDim + 2);
  return cudaSuccess;
}

extern "C" int expert_attention_mma_launch(const void *q, const void *k,
                                           const void *v, void *out,
                                           int queries, int keys, int head_dim,
                                           int heads, int prefix, float scale,
                                           void *ws, int splits, void *stream) {
  if (head_dim != kHeadDim) return cudaErrorInvalidValue;
  if (splits < 1) splits = 1;
  if (splits > 1 && ws == nullptr) return cudaErrorInvalidValue;
  // 82.6 KB is over the 48 KB a kernel gets without asking, and inside the
  // 99 KB this part allows [smem.bytes.cta.max].
  static bool opted = false;
  if (!opted) {
    cudaError_t e = cudaFuncSetAttribute(
        attention_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSmemBytes);
    if (e != cudaSuccess) return (int)e;
    opted = true;
  }
  const int32_t tiles = (queries + kBlockM - 1) / kBlockM;
  //: Rounded up to a whole number of key tiles so a slice boundary never
  //: falls inside one.
  const int32_t chunk =
      ((keys + splits - 1) / splits + kBlockN - 1) / kBlockN * kBlockN;
  float *part_o = (float *)ws;
  float *part_m = part_o + (splits > 1 ? int64_t(splits) * queries * kHeadDim : 0);
  float *part_l = part_m + (splits > 1 ? int64_t(splits) * queries : 0);

  const bool pdl = flash_vla_pdl_enabled();
  cudaError_t e = pdl_launch(
      pdl, attention_mma_kernel, dim3(tiles, splits), dim3(kThreads),
      kSmemBytes, (cudaStream_t)stream, (const __nv_bfloat16 *)q,
      (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v, (__nv_bfloat16 *)out,
      queries, keys, heads, prefix, scale, part_o, part_m, part_l, splits,
      chunk);
  if (e != cudaSuccess) return (int)e;
  if (splits > 1) {
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    e = pdl_launch(pdl, attention_merge_kernel, dim3(queries), dim3(kHeadDim), 0,
                   (cudaStream_t)stream, (const float *)part_o,
                   (const float *)part_m, (const float *)part_l,
                   (__nv_bfloat16 *)out, queries, splits);
    if (e != cudaSuccess) return (int)e;
  }
  return (int)cudaGetLastError();
}
