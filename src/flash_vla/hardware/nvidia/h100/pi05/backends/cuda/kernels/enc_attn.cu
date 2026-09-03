// enc_attn.cu -- fused Pi0.5 encoder (prefix) attention, one launch.
//
//   out[m, :] = softmax(Q[m, :] K^T * scale + key_mask) V,   m < M
//
// Multi-query with ONE K/V head, so the op is single-head attention over M
// flattened query rows (token * heads + head) against KEYS keys at head_dim
// DH.  One CTA owns 64 query rows; Q stays resident and the K/V stream is
// re-read by every CTA (0.5 MB each, L2-resident).
//
// Structure: two TMA producer warps fill independent K and V frame rings,
// and NWG math warpgroups take the key blocks round-robin (warpgroup w takes
// blocks w, w + NWG, ...) with private online-softmax state, folding their
// partials through shared memory at the end.
//
// NWG = 1 is the shipped configuration.  The FlashAttention-3 ping-pong that
// NWG = 2 implements was the design the rejected TileLang lane proposed, but
// it measures slower here: shared memory caps the ring at six 32 KB frames,
// so a second warpgroup halves the lookahead each one gets, and this kernel
// is bound by producer depth rather than by softmax overlap (Agent Note
// 2026-09-03-encoder-mqa-cuda-attention).
//
// The mainloop, the log2-domain online softmax and the fragment reuse that
// feeds P straight into the P.V wgmma are ported from the decoder's
// kAttention path (attn_taskloop.cu); it is the same 64-row tile shape.
//
// Build: nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -arch=sm_90a
//        --expt-relaxed-constexpr -I$CUTLASS_DIR/include
//        -I<repo>/src/flash_vla/hardware/nvidia/cuda enc_attn.cu -lcuda

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math_constants.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include "tile/sm90/sm90.cuh"
#include "tile/sm90/tma_host.cuh"

namespace enc_attn {

using namespace cute;
using BF = cutlass::bfloat16_t;
namespace tile = flash_vla::sm90;

// -------------------------------------------------------------- geometry
// Fixed by the production shape (models/pi05/spec.py, buffers.py at
// num_views = 3 and the prompt padded to 200): 968 prefix rows x 8 heads.
constexpr int DH = 256;             // head dim
constexpr int BM = 64;              // query rows per CTA
constexpr int BKK = 64;             // keys per block; the MmaS atom N must match

constexpr float LOG2E = 1.4426950408889634f;
constexpr float MASK_FLOOR = -1.0e30f;   // finite stand-in for MASK_NEG * log2e

#ifndef ENC_NWG
#define ENC_NWG 1
#endif
// K and V get independent rings and independent producer warps.  The math
// warpgroup blocks on K at the top of a block and on V at the end, so the
// critical path wants K lookahead; V only has to arrive before the P.V gemm.
// Shared memory is the budget (jobs 589060-589067 showed the kernel is
// producer-depth bound: 3 -> 2 slots costs 5.4 us/layer), so spend it on K.
#ifndef ENC_KDEPTH
#define ENC_KDEPTH 4
#endif
#ifndef ENC_VDEPTH
#define ENC_VDEPTH 2
#endif
// 0 none; 1 one-time offset so two warpgroups run half a block apart.
#ifndef ENC_PINGPONG
#define ENC_PINGPONG 0
#endif
constexpr int NWG = ENC_NWG;
constexpr int KDEPTH = ENC_KDEPTH;
constexpr int VDEPTH = ENC_VDEPTH;

constexpr int kMathThreads = 128;
// The producer is a whole warpgroup so `setmaxnreg` can hand its registers to
// the math warpgroups: setmaxnreg is warpgroup-wide, and a lone producer warp
// left ptxas at 168 registers per thread, spilling 192 B and serializing the
// wgmma pipeline (ptxas C7511).  Only warp 0 of the group issues TMAs; the
// other three release their registers and exit.
constexpr int THREADS = (NWG + 1) * kMathThreads;
constexpr int kProdWarp = NWG * 4;
constexpr int kProdRegs = 32;
constexpr int kMathRegs = NWG == 1 ? 240 : 232;
// named barriers: 0 is __syncthreads, 1..2 per warpgroup, 3 the ping-pong
// offset, 4 the math-only rendezvous the producer must not join.
constexpr int kBarPingPong = 3;
constexpr int kBarMath = 4;

// ------------------------------------------------------------ shared pool
constexpr int Q_FRAME_B = BM * DH * 2;            // 32768
constexpr int KV_FRAME_B = BKK * DH * 2;          // 32768 per K or V frame
constexpr int MASK_MAX_B = 2048;                  // padded key mask
constexpr int OFF_Q = 0;
constexpr int OFF_K = OFF_Q + Q_FRAME_B;
constexpr int OFF_V = OFF_K + KDEPTH * KV_FRAME_B;
constexpr int OFF_MASK = OFF_V + VDEPTH * KV_FRAME_B;
constexpr int OFF_BARS = OFF_MASK + MASK_MAX_B;
// [0, KDEPTH) fullK, then emptyK, then fullV, then emptyV, then Q
constexpr int BAR_FULL_K = 0;
constexpr int BAR_EMPTY_K = BAR_FULL_K + KDEPTH;
constexpr int BAR_FULL_V = BAR_EMPTY_K + KDEPTH;
constexpr int BAR_EMPTY_V = BAR_FULL_V + VDEPTH;
constexpr int BAR_Q = BAR_EMPTY_V + VDEPTH;
constexpr int N_BARS = BAR_Q + 1;
constexpr int SMEM_B = OFF_BARS + N_BARS * 8 + 8;
// The cross-warpgroup fold and the output staging re-image the ring, which
// every math warpgroup has finished with by then.
constexpr int OFF_XCHG = OFF_K;
constexpr int RING_B = (KDEPTH + VDEPTH) * KV_FRAME_B;
constexpr int XCHG_ACC_B = kMathThreads * 128 * 4;      // 64 KB
constexpr int XCHG_ML_B = BM * 2 * 4;
static_assert(NWG == 1 || RING_B >= XCHG_ACC_B + XCHG_ML_B,
              "the ring must hold one warpgroup's staged accumulator");
static_assert(RING_B >= BM * DH * 2, "output staging overlays the ring");
static_assert(SMEM_B <= 227 * 1024, "shared memory per block");

// --------------------------------------------------------- smem layouts
// Q: (64 queries, 256 dh) K-major SW128; image [chunk][row][64].
using SmemLayoutQ = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<BF>{}, Shape<Int<BM>, Int<DH>>{}));
// K frame: B of S = Q K^T is (N = keys, K = dh), dh contiguous -> K-major.
using SmemLayoutK = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<BF>{}, Shape<Int<BKK>, Int<DH>>{}));
// V frame: B of O = P V is (N = dh, K = keys) -> MN-major.  Step<_2,_1> makes
// the frame image identical to K's, so one 3-D TMA box fills either.
using SmemLayoutV = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<BF>{}, Shape<Int<DH>, Int<BKK>>{}, Step<_2, _1>{}));

using FullBar = tile::FullBarrier;
using EmptyBar = tile::EmptyBarrier;

using MmaS = decltype(make_tiled_mma(
    SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
using MmaO = decltype(make_tiled_mma(
    SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// wgmma C fragment (2,2,N/8) reinterpreted as the RS A fragment ((2,2,2),N/16):
// same register order, so P feeds the P.V wgmma without a shuffle.  Ported
// from FlashAttention-3 hopper/utils.h (convert_layout_acc_Aregs).
template <class Layout>
__device__ __forceinline__ auto acc_to_aregs(Layout acc) {
  auto l = logical_divide(get<0>(acc), Shape<Underscore, Underscore, _2>{});
  return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)),
                     get<1>(acc),
                     coalesce(make_layout(get<2, 1>(l), get<2>(acc))));
}

// Rendezvous of the math warpgroups only: the producer warp has returned by
// then, so __syncthreads would hang.
__device__ __forceinline__ void math_sync() {
  if constexpr (NWG == 1) asm volatile("bar.sync 1, 128;" ::: "memory");
  else asm volatile("bar.sync %0, %1;" ::"r"(kBarMath), "r"(NWG * kMathThreads) : "memory");
}

__device__ __forceinline__ void wg0_sync() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

struct Params {
  const CUtensorMap* tm_q;
  const CUtensorMap* tm_k;
  const CUtensorMap* tm_v;
  const __nv_bfloat16* __restrict__ key_mask;
  __nv_bfloat16* __restrict__ out;
  int keys;          // real key count; the mask covers [0, keys)
  int key_blocks;    // ceil(keys / BKK)
  float scale_log2;  // DH^-0.5 * log2(e)
};

// ------------------------------------------------------------- producer
// One frame per key block per stream.  K and V have independent rings, depths
// and issuing warps: a TMA costs [tma.issue.warp] 248 ns in the warp that
// issues it, so splitting the two streams across two warps halves the issue
// serialization, and independent depths let the deeper K ring cover the wait
// that is actually on the critical path.
template <int Depth, int OffFrame, int BarFull, int BarEmpty>
__device__ void frame_producer(const CUtensorMap* map, uint64_t* bars, uint8_t* pool,
                               int key_blocks, int lane) {
  for (int g = 0; g < key_blocks; ++g) {
    const int s = g % Depth;
    if (g >= Depth) {
      const uint32_t ph = ((g / Depth) - 1) & 1;
      while (!tile::mbarrier_try_wait_parity(bars + BarEmpty + s, ph)) {}
    }
    __syncwarp();
    if (lane == 0) {
      reinterpret_cast<FullBar*>(bars + BarFull + s)->arrive_and_expect_tx(KV_FRAME_B);
      tile::tma_load_3d(map, pool + OffFrame + s * KV_FRAME_B, 0, g * BKK, 0,
                        bars + BarFull + s);
    }
  }
}

// --------------------------------------------------------------- math
// Warpgroup `wg` owns key blocks wg, wg + NWG, ...  It keeps the running max
// and sum in the log2 domain and an unnormalised O accumulator; the caller
// folds the warpgroups' partials.
template <class AccO>
__device__ void math_loop(const Params& p, uint64_t* bars, uint8_t* pool,
                          int tid, int wg, AccO& acc_o, float* m_run, float* l_run) {
  MmaS mma_s;
  MmaO mma_o;
  auto thr_s = mma_s.get_thread_slice(tid);
  auto thr_o = mma_o.get_thread_slice(tid);
  Tensor acc_s = partition_fragment_C(mma_s, Shape<Int<BM>, Int<BKK>>{});
  Tensor cS = thr_s.partition_C(make_identity_tensor(Shape<Int<BM>, Int<BKK>>{}));
  constexpr int S_ELEMS = BKK / 2;                  // (2,2,N/8) fragment per thread
  static_assert(size(acc_s) == S_ELEMS);

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(pool + OFF_Q)), SmemLayoutQ{});
  Tensor tSrQ = thr_s.make_fragment_A(thr_s.partition_A(sQ));

  __align__(16) BF p_regs[S_ELEMS];
  Tensor tOrP = make_tensor(make_rmem_ptr(p_regs), acc_to_aregs(acc_s.layout()));

  while (!tile::mbarrier_try_wait_parity(bars + BAR_Q, 0)) {}

#if ENC_PINGPONG == 1
  // One-time offset: warpgroup 1 enters its first S gemm only after
  // warpgroup 0 has retired one, so from then on one warpgroup sits in the
  // tensor-core section while the other is in the softmax section.
  if (NWG == 2 && wg == 1)
    asm volatile("bar.sync %0, %1;" ::"r"(kBarPingPong), "r"(2 * kMathThreads) : "memory");
#endif

  int prev = -1;                                    // block whose V is still in flight
  for (int g = wg; g < p.key_blocks; g += NWG) {
    const int ks = g % KDEPTH;
    while (!tile::mbarrier_try_wait_parity(bars + BAR_FULL_K + ks, (g / KDEPTH) & 1)) {}

    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(pool + OFF_K + ks * KV_FRAME_B)),
                            SmemLayoutK{});
    Tensor tSrK = thr_s.make_fragment_B(thr_s.partition_B(sK));
    // wait 0 also retires the previous P.V batch, which is what makes both
    // the acc_o rescale below and the previous V frame's release safe.
    tile::gemm<true, 0, true, true>(mma_s, tSrQ, tSrK, acc_s);
    if (tid == 0) {
      // K(g) is consumed the moment that gemm retires; V(g-NWG) is retired by
      // the same wait, because its P.V batch was issued before this S batch.
      reinterpret_cast<EmptyBar*>(bars + BAR_EMPTY_K + ks)->arrive();
      if (prev >= 0) reinterpret_cast<EmptyBar*>(bars + BAR_EMPTY_V + prev % VDEPTH)->arrive();
    }
    prev = g;

#if ENC_PINGPONG == 1
    if (NWG == 2 && wg == 0 && g == 0)
      asm volatile("bar.arrive %0, %1;" ::"r"(kBarPingPong), "r"(2 * kMathThreads) : "memory");
#endif

    // Masked, scaled logits in the log2 domain, in place.  The mask slice is
    // padded to key_blocks * BKK with MASK_NEG, so keys past `keys` (whose K
    // rows the TMA zero-filled) contribute nothing and the hot loop needs no
    // column predicate.
    float rmax[2] = {-CUDART_INF_F, -CUDART_INF_F};
    CUTE_UNROLL
    for (int e = 0; e < S_ELEMS; e += 2) {
      const int col = get<1>(cS(e)), r = (e >> 1) & 1;
      const __nv_bfloat162 mk = *reinterpret_cast<const __nv_bfloat162*>(
          pool + OFF_MASK + (g * BKK + col) * 2);
      acc_s(e) = acc_s(e) * p.scale_log2 + fmaxf(bf2f(mk.x) * LOG2E, MASK_FLOOR);
      acc_s(e + 1) = acc_s(e + 1) * p.scale_log2 + fmaxf(bf2f(mk.y) * LOG2E, MASK_FLOOR);
      rmax[r] = fmaxf(rmax[r], fmaxf(acc_s(e), acc_s(e + 1)));
    }
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) {
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 1));
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 2));
      const float m_new = fmaxf(m_run[r], rmax[r]);
      const float alpha = fast_exp2(m_run[r] - m_new);     // 0 on the first block
      m_run[r] = m_new;
      l_run[r] *= alpha;
      CUTE_UNROLL
      for (int e = 0; e < 128; ++e)
        if (((e >> 1) & 1) == r) acc_o(e) *= alpha;
    }
    CUTE_UNROLL
    for (int e = 0; e < S_ELEMS; ++e) {
      const int r = (e >> 1) & 1;
      const float pe = fast_exp2(acc_s(e) - m_run[r]);
      l_run[r] += pe;
      p_regs[e] = BF(pe);
    }

    const int vs = g % VDEPTH;
    while (!tile::mbarrier_try_wait_parity(bars + BAR_FULL_V + vs, (g / VDEPTH) & 1)) {}
    Tensor sV = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(pool + OFF_V + vs * KV_FRAME_B)),
                            SmemLayoutV{});
    Tensor tOrV = thr_o.make_fragment_B(thr_o.partition_B(sV));
    tile::gemm<false, -1, true, true>(mma_o, tOrP, tOrV, acc_o);
  }
  warpgroup_wait<0>();
  if (prev >= 0 && tid == 0)
    reinterpret_cast<EmptyBar*>(bars + BAR_EMPTY_V + prev % VDEPTH)->arrive();

  // row sums across the quad
  CUTE_UNROLL
  for (int r = 0; r < 2; ++r) {
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 1);
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 2);
  }
}

__global__ void __launch_bounds__(THREADS, 1)
enc_attn_kernel(const __grid_constant__ CUtensorMap tm_q,
                const __grid_constant__ CUtensorMap tm_k,
                const __grid_constant__ CUtensorMap tm_v,
                const __nv_bfloat16* __restrict__ key_mask,
                __nv_bfloat16* __restrict__ out,
                int keys, int key_blocks, float scale_log2) {
  extern __shared__ __align__(1024) uint8_t pool[];
  const Params p{&tm_q, &tm_k, &tm_v, key_mask, out, keys, key_blocks, scale_log2};
  uint64_t* bars = reinterpret_cast<uint64_t*>(pool + OFF_BARS);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int cta = blockIdx.x;

  // One arrival each: a producer's arrive_and_expect_tx completes a full
  // barrier, and exactly one thread of the owning warpgroup releases an empty
  // one (a key block belongs to one warpgroup, so a slot never takes two).
  if (tid < N_BARS) {
    const bool is_empty = (tid >= BAR_EMPTY_K && tid < BAR_FULL_V)
                          || (tid >= BAR_EMPTY_V && tid < BAR_Q);
    if (is_empty) reinterpret_cast<EmptyBar*>(bars + tid)->init(1);
    else reinterpret_cast<FullBar*>(bars + tid)->init(1);
  }
  // The mask has no dependency and is padded to the block grid so the hot
  // loop needs no predicate; one warp covers it as 16 B stores.
  if (warp == kProdWarp) {
    const int lane = tid & 31;
    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(pool + OFF_MASK);
    for (int i = lane * 8; i < key_blocks * BKK; i += 32 * 8) {
      __nv_bfloat16 v[8];
      CUTE_UNROLL
      for (int j = 0; j < 8; ++j) {
        const int k = i + j;
        v[j] = k < keys ? key_mask[k] : __float2bfloat16(-3.0e38f);
      }
      *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(v);
    }
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  if (warp >= kProdWarp) {
    cutlass::arch::warpgroup_reg_dealloc<kProdRegs>();
    const int lane = tid & 31;
    if (warp == kProdWarp) {
      // Q first: every math warpgroup blocks on it before its first block.
      if (lane == 0) {
        reinterpret_cast<FullBar*>(bars + BAR_Q)->arrive_and_expect_tx(Q_FRAME_B);
        tile::tma_load_3d(&tm_q, pool + OFF_Q, 0, cta * BM, 0, bars + BAR_Q);
      }
      frame_producer<KDEPTH, OFF_K, BAR_FULL_K, BAR_EMPTY_K>(&tm_k, bars, pool, key_blocks, lane);
    } else if (warp == kProdWarp + 1) {
      frame_producer<VDEPTH, OFF_V, BAR_FULL_V, BAR_EMPTY_V>(&tm_v, bars, pool, key_blocks, lane);
    }
    return;
  }
  cutlass::arch::warpgroup_reg_alloc<kMathRegs>();

  const int wg = warp >> 2;
  const int wtid = tid - wg * kMathThreads;
  MmaO mma_o;
  auto thr_o = mma_o.get_thread_slice(wtid);
  Tensor acc_o = partition_fragment_C(mma_o, Shape<Int<BM>, Int<DH>>{});
  Tensor cO = thr_o.partition_C(make_identity_tensor(Shape<Int<BM>, Int<DH>>{}));
  static_assert(size(acc_o) == 128);
  clear(acc_o);
  float m_run[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float l_run[2] = {0.f, 0.f};

  math_loop(p, bars, pool, wtid, wg, acc_o, m_run, l_run);

  if constexpr (NWG == 2) {
    // Fold warpgroup 1's partial into warpgroup 0's.  Both warpgroups use the
    // same MmaO thread slice, so thread i of either covers the same (row, col)
    // set and the exchange is a plain per-thread copy through the ring.  The
    // first rendezvous is what makes overwriting the ring safe: warpgroup 0
    // may still have been reading a K/V frame from it.
    float* xa = reinterpret_cast<float*>(pool + OFF_XCHG);
    float* xml = reinterpret_cast<float*>(pool + OFF_XCHG + XCHG_ACC_B);
    math_sync();
    if (wg == 1) {
      CUTE_UNROLL
      for (int e = 0; e < 128; e += 4)
        *reinterpret_cast<float4*>(xa + (e >> 2) * (kMathThreads * 4) + wtid * 4) =
            make_float4(acc_o(e), acc_o(e + 1), acc_o(e + 2), acc_o(e + 3));
      if ((wtid & 3) == 0) {
        CUTE_UNROLL
        for (int r = 0; r < 2; ++r) {
          const int row = get<0>(cO(2 * r));
          xml[row * 2] = m_run[r];
          xml[row * 2 + 1] = l_run[r];
        }
      }
    }
    math_sync();
    if (wg == 1) return;

    float w0[2], w1[2];
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) {
      const int row = get<0>(cO(2 * r));
      const float m1 = xml[row * 2], l1 = xml[row * 2 + 1];
      const float m = fmaxf(m_run[r], m1);
      w0[r] = fast_exp2(m_run[r] - m);
      w1[r] = fast_exp2(m1 - m);
      l_run[r] = l_run[r] * w0[r] + l1 * w1[r];
    }
    CUTE_UNROLL
    for (int e = 0; e < 128; e += 4) {
      const float4 v = *reinterpret_cast<const float4*>(
          xa + (e >> 2) * (kMathThreads * 4) + wtid * 4);
      const int r0 = (e >> 1) & 1, r1 = ((e + 2) >> 1) & 1;
      acc_o(e) = acc_o(e) * w0[r0] + v.x * w1[r0];
      acc_o(e + 1) = acc_o(e + 1) * w0[r0] + v.y * w1[r0];
      acc_o(e + 2) = acc_o(e + 2) * w0[r1] + v.z * w1[r1];
      acc_o(e + 3) = acc_o(e + 3) * w0[r1] + v.w * w1[r1];
    }
  }

  // Normalise and store.  The CTA's 64 rows are contiguous in `out`, so the
  // tile is staged row-major in shared memory and written with one bulk store
  // instead of 4 B fragment stores.
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(pool + OFF_XCHG);
  const float inv[2] = {1.f / l_run[0], 1.f / l_run[1]};
  if constexpr (NWG == 1) math_sync();   // the ring is free only after the mainloop
  CUTE_UNROLL
  for (int e = 0; e < 128; e += 2) {
    const int row = get<0>(cO(e)), col = get<1>(cO(e)), r = (e >> 1) & 1;
    *reinterpret_cast<__nv_bfloat162*>(sO + row * DH + col) =
        __nv_bfloat162{__float2bfloat16(acc_o(e) * inv[r]),
                       __float2bfloat16(acc_o(e + 1) * inv[r])};
  }
  tile::fence_proxy_async_shared();
  wg0_sync();
  if (wtid == 0) {
    tile::bulk_store_1d(out + (size_t)cta * BM * DH, sO, BM * DH * 2);
    tile::store_commit_group();
    tile::store_wait_group_read<0>();
  }
}

// ------------------------------------------------------------------ host
// (rows, cols) row-major viewed as {64 elems, rows, cols/64}: one box of
// {64, box_rows, cols/64} lands as [chunk][row][64], the CuTe SW128 K-major
// image of a (box_rows x cols) tile.  Rows past `rows` are zero-filled by the
// TMA, which is what makes the ragged last key block free.
static CUtensorMap enc3d(const void* ptr, uint64_t cols, uint64_t rows, uint32_t box_rows,
                         CUresult* rc) {
  CUtensorMap m{};
  *rc = tile::encode_tensor_map_3d<BF>(&m, ptr, 64, rows, cols / 64, cols * sizeof(BF),
                                       64 * sizeof(BF), 64, box_rows,
                                       (uint32_t)(cols / 64), 128);
  return m;
}

struct Maps {
  CUtensorMap q, k, v;
};

}  // namespace enc_attn

// Encode the three tensor maps once and hand the caller an opaque blob: the
// launch path is graph-captured and cuTensorMapEncodeTiled is not capture safe.
extern "C" int enc_attn_make_maps(const void* q, const void* k, const void* v,
                                  int m_rows, int keys, void* out_maps) {
  using namespace enc_attn;
  CUresult rc = CUDA_SUCCESS;
  Maps& maps = *reinterpret_cast<Maps*>(out_maps);
  maps = Maps{};
  maps.q = enc3d(q, DH, m_rows, BM, &rc);
  if (rc) return 1000 + (int)rc;
  maps.k = enc3d(k, DH, keys, BKK, &rc);
  if (rc) return 1100 + (int)rc;
  maps.v = enc3d(v, DH, keys, BKK, &rc);
  if (rc) return 1200 + (int)rc;
  return 0;
}

extern "C" int enc_attn_launch(const void* maps, const void* key_mask, void* out,
                               int m_rows, int keys, float scale, void* stream) {
  using namespace enc_attn;
  if (m_rows % BM != 0) return 1300;
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(enc_attn_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_B);
    if (e != cudaSuccess) return 1400 + (int)e;
    attr_set = true;
  }
  const Maps& mm = *reinterpret_cast<const Maps*>(maps);
  const int key_blocks = (keys + BKK - 1) / BKK;
  if (key_blocks * BKK * 2 > MASK_MAX_B) return 1500;
  enc_attn_kernel<<<m_rows / BM, THREADS, SMEM_B, (cudaStream_t)stream>>>(
      mm.q, mm.k, mm.v, (const __nv_bfloat16*)key_mask, (__nv_bfloat16*)out,
      keys, key_blocks, scale * LOG2E);
  return (int)cudaGetLastError();
}

extern "C" int enc_attn_geometry(int* threads, int* smem_b, int* nwg, int* kdepth, int* vdepth,
                                 int* map_bytes) {
  using namespace enc_attn;
  *threads = THREADS;
  *smem_b = SMEM_B;
  *nwg = NWG;
  *kdepth = KDEPTH;
  *vdepth = VDEPTH;
  *map_bytes = (int)sizeof(Maps);
  return 0;
}
