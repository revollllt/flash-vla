// attn_taskloop.cu -- Pi0.5 action-expert attention block as ONE persistent
// task-loop launch (Agent Note 2026-08-27-attention-block-taskloop):
//
//   kQkvProj   (n-tile, k-split) : q_buf[h] / cache suffix = RoPE(F * ((x*s) @ W_qkv) + b)
//   kAttention (head, kv-split)  : o_buf[h]  = softmax(q_buf[h] K^T * scale + mask) V
//   kOutProj   (n-tile, head)    : out      += (o_buf @ W_o) * g
//
// Contract: specs/tile/attention_block_contract.md.  ABI, geometry and the
// counter map: sm90_attn_task_desc.cuh.  Torch mirrors: attn_reference.py
// (task by task) and attn_block_reference.py (block).
//
// Structure follows ffn_taskloop.cu: 132 workers, blockIdx.x indexes a private
// (TASK_SLOTS, 4) descriptor row, one task per slot, gmem counters order the
// tasks, no runtime scheduler.  192 threads: warps 0..3 are the math
// warpgroup, warp 4 the "weight" TMA producer (W_qkv / K cache / W_o), warp 5
// the "activation" TMA producer (x + scale slice / Q + V cache / o_buf).  One
// 160 KB shared pool serves every task kind; rings are reinitialised at each
// slot boundary, so a slot never inherits phase state from the previous kind.
//
// Build: nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -arch=sm_90a
//        --expt-relaxed-constexpr -I$CUTLASS_DIR/include -I$FLASHMLA_CSRC
//        attn_taskloop.cu -lcuda

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <math_constants.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

#include "sm90_attn_task_desc.cuh"
#include "sm90/helpers.h"

namespace attn {

using namespace cute;
using BF = cutlass::bfloat16_t;
namespace hdr = flash_vla::pi05::sm90::attn;
using TaskDesc = hdr::TaskDescriptor;
using TaskKind = hdr::TaskKind;
using CM = hdr::CounterMap;

using hdr::M; using hdr::M_PAD; using hdr::D; using hdr::H; using hdr::DH;
using hdr::QKV_W; using hdr::PREFIX_LEN; using hdr::KEYS; using hdr::KEYS_PAD;
using hdr::QKV_BN; using hdr::QKV_BK; using hdr::QKV_SPLIT; using hdr::QKV_TILES;
using hdr::QKV_TASKS; using hdr::QKV_TRIP;
using hdr::ATTN_BKK; using hdr::ATTN_DEPTH; using hdr::ATTN_SPLIT; using hdr::ATTN_TRIP;
using hdr::OUT_BN; using hdr::OUT_BK; using hdr::OUT_SPLIT; using hdr::OUT_TILES;
using hdr::OUT_TRIP; using hdr::COMBINE_ROWS; using hdr::COMBINE_TASKS;
using hdr::ATTN_TASKS; using hdr::OUT_TASKS;
using hdr::N_CTAS; using hdr::TASK_SLOTS; using hdr::THREADS;

// ------------------------------------------------------------------ roles
constexpr int kMathThreads = 128;
constexpr int kWeightWarp = 4;      // W_qkv ring / K ring / W_o ring
constexpr int kActWarp = 5;         // x ring + scale slice / Q + V ring / o_buf ring
static_assert(THREADS == 6 * 32, "192-thread profile: 1 math WG + 2 producer warps");

// -------------------------------------------------------------- geometry
constexpr int ROPE_COLS = (H + 1) * DH;            // Q and K rotate, V does not
constexpr float MASK_FLOOR = -1.0e30f;             // finite stand-in for MASK_NEG*log2e
constexpr float SCALE_LOG2 = 0.0625f * 1.4426950408889634f;   // DH^-0.5 * log2(e)
// Ring depths.  qkv: 16 KB frames (BK=128), depth 4, 8 stages -- half the
// task in flight, one refill per frame.  o_proj: 32 KB frames (BK=256),
// depth 2, 1 stage.  Attention: 2 x 32 KB per ring.  Every ring's frames
// occupy [0, 64 KB) (A) and [64 KB, 128 KB) (W); only the stride differs.
constexpr int QKV_DEPTH = 4;
constexpr int OUT_DEPTH = 2;
constexpr int DEPTH_MAX = 4;
static_assert(DEPTH_MAX >= ATTN_DEPTH && DEPTH_MAX >= QKV_DEPTH && DEPTH_MAX >= OUT_DEPTH);
static_assert(ATTN_DEPTH >= ATTN_TRIP && OUT_DEPTH >= OUT_TRIP && QKV_DEPTH * 2 == QKV_TRIP);
static_assert(QKV_BN == 64 && QKV_BK == 128 && OUT_BN == 64 && OUT_BK == 256,
              "qkv frames are 64x128 (one {64,64,2} box + one {64,128} box); o_proj 64x256");
static_assert(DH % 64 == 0 && M_PAD == 64);

// spec: pipeline.staged_buffers -- one pool, three views.  All offsets are
// multiples of 1024 so every SW128 tile base is swizzle-aligned.
constexpr int F256_B     = 64 * 256 * 2;           // 32768: o_buf / W_o frame
constexpr int QKV_F_B    = 64 * QKV_BK * 2;        // 16384: x / W_qkv frame
constexpr int S_FRAME_B  = QKV_BK * 2;             // 256: ada_scale slice
constexpr int KV_FRAME_B = ATTN_BKK * DH * 2;      // 32768: 64 keys x 256
constexpr int Q_FRAME_B  = M_PAD * DH * 2;         // 32768: one head's queries
// projection view (qkv, o_proj)
constexpr int OFF_A = 0;                                // 4 x 16 KB (qkv) | 2 x 32 KB (o_proj)
constexpr int OFF_W = OFF_A + 65536;                    // same
constexpr int OFF_S = OFF_W + 65536;                    // 131072: QKV_DEPTH x 256
static_assert(QKV_DEPTH * QKV_F_B == 65536 && OUT_DEPTH * F256_B == 65536);
// attention view
constexpr int OFF_Q = 0;                                // 32768
constexpr int OFF_K = OFF_Q + Q_FRAME_B;                // 32768: ATTN_DEPTH x 32768
constexpr int OFF_V = OFF_K + ATTN_DEPTH * KV_FRAME_B;  // 98304: ATTN_DEPTH x 32768
static_assert(OFF_V + ATTN_DEPTH * KV_FRAME_B == hdr::SMEM_POOL_B, "attention body sizes the pool");
static_assert(OFF_S + QKV_DEPTH * S_FRAME_B <= hdr::SMEM_POOL_B);
constexpr int OFF_BARS = hdr::SMEM_POOL_B;
// barrier pool (8 B each): [0,8) fullA [8,16) emptyA [16,24) fullW [24,32) emptyW [32] fullQ
constexpr int BAR_FULL_A = 0, BAR_EMPTY_A = DEPTH_MAX, BAR_FULL_W = 2 * DEPTH_MAX,
              BAR_EMPTY_W = 3 * DEPTH_MAX, BAR_FULL_Q = 4 * DEPTH_MAX;
// [4*DEPTH_MAX+1] drain: the four math warps arrive once their mainloop has
// retired every frame, so the producer may re-image the pool for the fold.
constexpr int BAR_DRAIN = 4 * DEPTH_MAX + 1;
constexpr int N_BARS = 4 * DEPTH_MAX + 2;
// the attention key-mask slice (128 keys x bf16) rides in the barrier page
constexpr int OFF_MASK = OFF_BARS + 512;
constexpr int MASK_SLICE_B = (KEYS_PAD / ATTN_SPLIT) * 2;
static_assert(N_BARS * 8 <= 512 && OFF_MASK + MASK_SLICE_B <= hdr::SMEM_B);
// Split-0 fold staging (contract 3.3, implementation-private): after the
// mainloop the pool is free, so the act producer bulk-copies the sibling
// partials into it and the math warps fold from shared memory.  Folding from
// global directly (job 555106) cost up to 4.7 / 9.8 / 3.5 us per join: under
// 255 registers the compiler could only keep a few loads in flight.
constexpr int QKV_PARTIAL_B = 64 * 64 * 4;                          // 16 KB f32
constexpr int OUT_PARTIAL_B = 64 * 64 * 4;                          // 16 KB f32
constexpr int OFF_FOLD = 0;
static_assert((OUT_SPLIT - 1) * OUT_PARTIAL_B <= hdr::SMEM_POOL_B);
static_assert((QKV_SPLIT - 1) * QKV_PARTIAL_B <= hdr::SMEM_POOL_B);

// ------------------------------------------------------- smem layouts (CuTe)
// 64x256 K-major SW128 tile: x frame, o_buf frame (A operands); image
// [chunk][row][64], filled by one 3-D box {64, 64, 4}.
using SmemLayoutA256 = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<BF>{}, Shape<Int<64>, Int<256>>{}));
// (K=256 rows, N=64) weight tile as stored, N contiguous -> MN-major B; one
// {64, 256} box.
using SmemLayoutW256 = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<BF>{}, Shape<Int<64>, Int<256>>{}));
// qkv frames: 64x128 K-major x tile (one 3-D box {64, 64, 2}) and the
// (K=128 rows, N=64) W_qkv tile as stored.
using SmemLayoutA_QKV = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<BF>{}, Shape<Int<64>, Int<QKV_BK>>{}));
#ifdef ATTN_QKV_WT
using SmemLayoutW_QKV = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<BF>{}, Shape<Int<64>, Int<QKV_BK>>{}));      // (N=64, K=128) K-major
#else
using SmemLayoutW_QKV = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<BF>{}, Shape<Int<64>, Int<QKV_BK>>{}));
#endif
// Q: (64 queries, 256 dh) K-major; image [dh chunk][row][64].
using SmemLayoutQ = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<BF>{}, Shape<Int<M_PAD>, Int<DH>>{}));
// K frame: B of S = Q K^T is (N=keys, K=dh), dh contiguous -> K-major.
// Image [dh chunk][32 keys][64], the same bytes the V frame holds.
using SmemLayoutK = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<BF>{}, Shape<Int<ATTN_BKK>, Int<DH>>{}));
// V frame: B of O = P V is (N=dh, K=keys), dh contiguous -> MN-major.  The
// Step<_2,_1> order puts the 8-key groups of one dh chunk back to back, so
// the frame image is identical to K's and one 3-D TMA box fills either.
using SmemLayoutV = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<BF>{}, Shape<Int<DH>, Int<ATTN_BKK>>{}, Step<_2, _1>{}));

using FullBar  = cutlass::arch::ClusterTransactionBarrier;
using EmptyBar = cutlass::arch::ClusterBarrier;

// spec math: wgmma atoms.  A second math warpgroup buys nothing [wgmma.ratio.sm.wg2].
using MmaProj = decltype(make_tiled_mma(
    SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{}));
// qkv: A from REGISTERS.  The AdaRMS scale s (indexed by K) is applied to the
// ldmatrix'd fragment, so the frame is never rewritten in shared memory: no
// RMW pass, no proxy fence, no warpgroup barrier, and the wgmma reads only B
// from smem.  Ablation job 556195 priced the in-smem scale at 0.33 us/stage
// and the SS wgmma at 0.46 us/stage (3.3x [wgmma.issue.wg.ss]): both smem-bandwidth
// bound once the RMW, the TMA landing and the 4 KB/wgmma operand reads share
// the 128 B/clk port.
// ATTN_QKV_WT (experiment): W_qkv arrives pre-transposed, (QKV_W, D) row-major,
// so B is K-major and one 3-D box {64, 64, 2} loads the (64 n x 128 k) tile.
// Prices the contract's "weights as stored" clause against wgmma.issue.wg.ss, whose
// probe used K-major operands on both sides.
#ifdef ATTN_QKV_WT
using MmaQkv = decltype(make_tiled_mma(
    SM90_64x64x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}));
#else
using MmaQkv = decltype(make_tiled_mma(
    SM90_64x64x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));
#endif
using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, BF>;
template <class TiledCopy>
__device__ __forceinline__ auto smem_thr_copy_A_slice(const TiledCopy& c, int tid) { return c.get_thread_slice(tid); }
using MmaS = decltype(make_tiled_mma(
    SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
static_assert(ATTN_BKK == 64, "MmaS atom N must equal the key block");
using MmaO = decltype(make_tiled_mma(
    SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

// --------------------------------------------------------------- utilities
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void issue_tma_2d(
    const CUtensorMap* map, void* dst, int32_t c0, int32_t c1, uint64_t* bar) {
  uint32_t d = smem_u32(dst), b = smem_u32(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      :: "r"(d), "l"(map), "r"(c0), "r"(c1), "r"(b) : "memory");
}

__device__ __forceinline__ void issue_tma_3d(
    const CUtensorMap* map, void* dst, int32_t c0, int32_t c1, int32_t c2, uint64_t* bar) {
  uint32_t d = smem_u32(dst), b = smem_u32(bar);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];"
      :: "r"(d), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b) : "memory");
}

__device__ __forceinline__ void issue_bulk_1d(
    void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  uint32_t d = smem_u32(dst), b = smem_u32(bar);
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];"
      :: "r"(d), "l"(src), "r"(bytes), "r"(b) : "memory");
}

// Bulk store of a contiguous smem image to global, issued by one thread.
// Publishing a partial as thousands of 4-16 B generic stores made the release
// fence wait for every one of them (job 555286: 3.9 us per attention task);
// one bulk group is a single completion.
__device__ __forceinline__ void issue_bulk_store_1d(void* gdst, const void* ssrc, uint32_t bytes) {
  uint32_t src = smem_u32(ssrc);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :: "l"(gdst), "r"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_store_commit_and_wait() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
// only the smem source must stay valid: enough when nothing in this kernel
// consumes the destination (a later kernel does)
__device__ __forceinline__ void bulk_store_commit_and_wait_read() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// Generic-proxy stores by another CTA, published through a release/acquire
// counter, are visible to THIS thread's generic loads after the acquire, but
// not yet to its async-proxy (TMA) reads: the proxy fence closes that gap.
// Executed by the issuing lane, after its acquire and before its first TMA.
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

// release: every prior generic store of this CTA precedes the count.  A
// release reduction carries the fence itself; [atom.ratio.ret] prices red at
// 1.3x below a returning atom, and a separate membar.gl is not needed.
__device__ __forceinline__ void counter_release(uint32_t* c) {
  asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(c) : "memory");
}
// arrive-and-count: release this CTA's publish, acquire the others'
__device__ __forceinline__ uint32_t counter_arrive_acq_rel(uint32_t* c) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(c) : "memory");
  return old;
}

// Watchdog: a persistent-kernel bug hangs rather than fails, so every wait
// carries a deadline and records {site, g, tid} before trapping.
constexpr long long WATCHDOG_CYCLES = 1ll << 31;   // ~1.2 s at 1.8 GHz

__device__ __forceinline__ void wd_fire(long long* dbg, int site, int g) {
  if (dbg) {
    long long* d = dbg + blockIdx.x * 4;
    d[0] = site; d[1] = g; d[2] = threadIdx.x; d[3] = 1;
    __threadfence_system();
  }
  __trap();
}

__device__ __forceinline__ void wait_bar_wd(uint64_t* bar, uint32_t phase,
                                            int site, int g, long long* dbg) {
  uint32_t addr = smem_u32(bar);
  uint32_t done = 0;
  long long t0 = clock64();
  while (!done) {
    asm volatile(
        "{\n .reg .pred p;\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        " selp.u32 %0, 1, 0, p;\n}"
        : "=r"(done) : "r"(addr), "r"(phase));
    if (!done && clock64() - t0 > WATCHDOG_CYCLES) wd_fire(dbg, site, g);
  }
}

// acquire poll on a gmem counter; the caller decides which lane polls
__device__ __forceinline__ void counter_wait(const uint32_t* c, uint32_t need,
                                             int site, int g, long long* dbg) {
  uint32_t v;
  long long t0 = clock64();
  while (true) {
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(c));
    if (v >= need) break;
    if (clock64() - t0 > WATCHDOG_CYCLES) wd_fire(dbg, site, g);
    __nanosleep(64);
  }
}

// math-WG barrier (warps 0..3; producer warps never arrive)
__device__ __forceinline__ void mathwg_sync() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

// Physical SW128 element offset of (row m, k kk) inside one 64x256 K-major
// frame [chunk][row][64]: the 16 B chunk index within the 128 B row XORs with
// (row & 7).  Shared by the TMA image and the in-place scale.
__device__ __forceinline__ int a_phys_elem(int m, int kk) {
  const int chunk = kk >> 6, in = kk & 63, c = in >> 3;
  return chunk * (64 * 64) + m * 64 + ((c ^ (m & 7)) << 3) + (in & 7);
}

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Split partials live in an implementation-private, THREAD-MAJOR interleaved
// layout (contract 3.3): element e of thread t sits at (e/V)*(128*V) + t*V +
// e%V, V = 16 B worth of elements, so one warp instruction moves 512 B
// contiguous and split 0's fold is 8 (f32) or 16 (bf16) vector loads per
// partial instead of 32-128 scattered 4 B loads.  The (row, col) form the
// timeline job 555066 measured cost 5-17 us per join under 255 registers.
__device__ __forceinline__ float* partial_f32_ptr(float* base, int tid, int e) {
  return base + (e >> 2) * (kMathThreads * 4) + tid * 4;
}

template <class Acc>
__device__ __forceinline__ void stage_partial_f32_smem(uint8_t* base, int tid, const Acc& acc) {
  CUTE_UNROLL
  for (int e = 0; e < size(acc); e += 4)
    *reinterpret_cast<float4*>(base + ((e >> 2) * (kMathThreads * 4) + tid * 4) * 4) =
        make_float4(acc(e), acc(e + 1), acc(e + 2), acc(e + 3));
}

template <class Acc>
__device__ __forceinline__ void store_partial_f32(float* base, int tid, const Acc& acc) {
  CUTE_UNROLL
  for (int e = 0; e < size(acc); e += 4)
    *reinterpret_cast<float4*>(partial_f32_ptr(base, tid, e)) =
        make_float4(acc(e), acc(e + 1), acc(e + 2), acc(e + 3));
}

// same thread-major image, read back from the staging pool
template <class Acc>
__device__ __forceinline__ void fold_partial_f32_smem(const uint8_t* base, int tid, Acc& acc) {
  CUTE_UNROLL
  for (int e = 0; e < size(acc); e += 4) {
    const float4 v = *reinterpret_cast<const float4*>(base + ((e >> 2) * (kMathThreads * 4) + tid * 4) * 4);
    acc(e) += v.x; acc(e + 1) += v.y; acc(e + 2) += v.z; acc(e + 3) += v.w;
  }
}

struct Params;
// the math warps have retired every frame: the pool may be re-imaged
__device__ __forceinline__ void arrive_drain(const Params& p, int tid);

template <class Acc>
__device__ __forceinline__ void fold_partial_f32(const float* base, int tid, Acc& acc) {
  CUTE_UNROLL
  for (int e = 0; e < size(acc); e += 4) {
    const float4 v = __ldcg(reinterpret_cast<const float4*>(
        partial_f32_ptr(const_cast<float*>(base), tid, e)));
    acc(e) += v.x; acc(e + 1) += v.y; acc(e + 2) += v.z; acc(e + 3) += v.w;
  }
}

// wgmma C fragment (2,2,N/8) reinterpreted as the RS A fragment ((2,2,2), N/16):
// same register order, so P feeds the P.V wgmma without a shuffle.  Ported
// from FlashAttention-3 hopper/utils.h (convert_layout_acc_Aregs).
template <class Layout>
__device__ __forceinline__ auto acc_to_aregs(Layout acc) {
  auto l = logical_divide(get<0>(acc), Shape<Underscore, Underscore, _2>{});
  return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)),
                     get<1>(acc),
                     coalesce(make_layout(get<2, 1>(l), get<2>(acc))));
}

struct Bars {
  uint64_t* base;
  __device__ FullBar*  full_a(int s) const { return reinterpret_cast<FullBar*>(base + BAR_FULL_A + s); }
  __device__ EmptyBar* empty_a(int s) const { return reinterpret_cast<EmptyBar*>(base + BAR_EMPTY_A + s); }
  __device__ FullBar*  full_w(int s) const { return reinterpret_cast<FullBar*>(base + BAR_FULL_W + s); }
  __device__ EmptyBar* empty_w(int s) const { return reinterpret_cast<EmptyBar*>(base + BAR_EMPTY_W + s); }
  __device__ FullBar*  full_q() const { return reinterpret_cast<FullBar*>(base + BAR_FULL_Q); }
  __device__ EmptyBar* drain() const  { return reinterpret_cast<EmptyBar*>(base + BAR_DRAIN); }
  __device__ uint64_t* raw(void* b) const { return reinterpret_cast<uint64_t*>(b); }
};

struct Params {
  const CUtensorMap* tm_x;     // (M_PAD, D) 3-D      box 64x64x4 (256 k)
  const CUtensorMap* tm_wqkv;  // (D, QKV_W)          box 64x256
  const CUtensorMap* tm_q;     // (H*M_PAD, DH) 3-D   box 64x64x4
  const CUtensorMap* tm_k;     // (KEYS_PAD, DH) 3-D  box 64x32x4
  const CUtensorMap* tm_v;
  const CUtensorMap* tm_o;     // (H*M_PAD, DH) 3-D   box 64x64x4
  const CUtensorMap* tm_wo;    // (H*DH, D)           box 64x256
  const __nv_bfloat16* rms_factor;
  const __nv_bfloat16* ada_scale;
  const __nv_bfloat16* qkv_bias;
  const __nv_bfloat16* rope;
  const __nv_bfloat16* key_mask;
  const __nv_bfloat16* ada_gate;
  __nv_bfloat16* k_cache;
  __nv_bfloat16* v_cache;
  __nv_bfloat16* out;
  __nv_bfloat16* q_buf;
  __nv_bfloat16* o_buf;
  float* qkv_partial;          // (QKV_SPLIT-1, QKV_TILES, 64x64) f32
  __nv_bfloat16* attn_partial; // (ATTN_SPLIT-1, H, M_PAD, DH) bf16, normalised
  float* attn_lse;             // (ATTN_SPLIT-1, H, M_PAD, 2) f32: m, l
  float* out_partial;          // (OUT_SPLIT-1, OUT_TILES, 64x64) f32
  uint32_t* counters;
  long long* dbg;
  long long* timeline;  // optional (N_CTAS, TASK_SLOTS, 5) ns stamps
  // Standalone mode: one task per CTA of an ordinary grid kernel, no
  // counters -- dependencies are kernel order, every split publishes its
  // partial and a reduce/combine kernel folds them.  Same bodies, so a
  // per-op number and the task-loop number measure the same code.
  bool standalone;
  // PDL standalone: this grid carries the programmatic stream-serialization
  // attribute, so it may start before its stream predecessor retires.
  // Kernel-order dependencies then hold only behind an explicit
  // cudaGridDependencySynchronize at every read of predecessor-written state.
  bool pdl;
  uint8_t* pool;
  Bars bars;
};

// %globaltimer stamps for the critical-path timeline: 0 slot start, 1 first
// frame landed on the math side, 2 mainloop retired, 3 split join satisfied
// (== 2 for a split that only publishes), 4 task end.
__device__ __forceinline__ void stamp(const Params& p, int slot, int idx) {
  if (p.timeline == nullptr || threadIdx.x != 0) return;
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  p.timeline[(blockIdx.x * TASK_SLOTS + slot) * 5 + idx] = t;
}

// Publish `bytes` of a partial staged at pool+0 (and optionally a second
// region) to global with one bulk group, then release the counter.
__device__ __forceinline__ void publish_staged(const Params& p, int tid, void* gdst0, uint32_t bytes0,
                                               void* gdst1, uint32_t bytes1, uint32_t* counter) {
  cutlass::arch::fence_view_async_shared();   // generic smem writes -> async proxy
  mathwg_sync();
  if (tid == 0) {
    issue_bulk_store_1d(gdst0, p.pool, bytes0);
    if (bytes1) issue_bulk_store_1d(gdst1, p.pool + bytes0, bytes1);
    if (counter) {
      bulk_store_commit_and_wait();
      fence_proxy_async_global();             // async-proxy writes -> generic readers
      counter_release(counter);
    } else {
      bulk_store_commit_and_wait_read();      // kernel completion publishes
    }
  }
}

__device__ __forceinline__ void arrive_drain(const Params& p, int tid) {
  __syncwarp();
  if ((tid & 31) == 0) p.bars.drain()->arrive();
}

// Watchdog sites (mirrored by WATCHDOG_SITES in attn_taskloop.py)
enum Site : int {
  kSiteQkvMathFull = 1, kSiteQkvProdEmpty = 2, kSiteQkvJoin = 3,
  kSiteAttnMathFull = 4, kSiteAttnProdEmpty = 5, kSiteAttnDep = 6, kSiteAttnJoin = 7,
  kSiteOutMathFull = 8, kSiteOutProdEmpty = 9, kSiteOutDep = 10, kSiteOutJoin = 11,
  kSiteOutQkvDone = 12,
};

// qkv epilogue (contract 4.2): rstd, then bias, THEN RoPE on adjacent pairs,
// bf16 stores head-major into q_buf or into the cache suffix.  Fragment order
// is e = c + 2r + 4i: c the adjacent column, r the row half (row0, row0 + 8),
// i the 8-column group, so operands load once per group, not per element.
// Shared by the task-loop split-0 path and the standalone reduce kernel.
struct QkvEpilogueOperands {
  float f_row[2];
  __nv_bfloat162 bias2[8];
  __nv_bfloat162 rope2[16];
};

template <class CC>
__device__ __forceinline__ QkvEpilogueOperands qkv_epilogue_load(const Params& p, int n0, const CC& cC) {
  QkvEpilogueOperands op;
  const bool rotate = n0 < ROPE_COLS;
  const int d0 = n0 % DH, row0 = get<0>(cC(0));
  op.f_row[0] = bf2f(p.rms_factor[row0]);
  op.f_row[1] = bf2f(p.rms_factor[row0 + 8]);
  CUTE_UNROLL
  for (int i = 0; i < 8; ++i) {
    const int col = get<1>(cC(4 * i));
    op.bias2[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv_bias + n0 + col);
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r)
      op.rope2[2 * i + r] = rotate
          ? *reinterpret_cast<const __nv_bfloat162*>(p.rope + (row0 + 8 * r) * DH + d0 + col)
          : __nv_bfloat162{__float2bfloat16(1.f), __float2bfloat16(0.f)};
  }
  return op;
}

// One 8-column group i of the fragment: values v[4] = acc(4i .. 4i+3).
template <class CC>
__device__ __forceinline__ void qkv_epilogue_group(const Params& p, int n0, int i, const float* v, const CC& cC,
                                                   const QkvEpilogueOperands& op) {
  const bool rotate = n0 < ROPE_COLS;
  const int d0 = n0 % DH, row0 = get<0>(cC(0)), col = get<1>(cC(4 * i));
  __nv_bfloat16* dst_base = (n0 < H * DH)
      ? p.q_buf + ((size_t)((n0 / DH) * M_PAD)) * DH + d0                                   // Q: head-major slab
      : ((n0 < H * DH + DH) ? p.k_cache : p.v_cache) + ((size_t)PREFIX_LEN) * DH + d0;   // K/V: cache suffix
  const bool store_pad_rows = n0 < H * DH;
  const __nv_bfloat162 b2 = op.bias2[i];
  CUTE_UNROLL
  for (int r = 0; r < 2; ++r) {
    const int row = row0 + 8 * r;
    float v0 = v[2 * r] * op.f_row[r] + bf2f(b2.x);
    float v1 = v[2 * r + 1] * op.f_row[r] + bf2f(b2.y);
    if (rotate) {
      const __nv_bfloat162 cs = op.rope2[2 * i + r];
      const float c = bf2f(cs.x), sn = bf2f(cs.y);
      const float r0 = v0 * c - v1 * sn, r1 = v1 * c + v0 * sn;
      v0 = r0; v1 = r1;
    }
    if (store_pad_rows || row < M)
      *reinterpret_cast<__nv_bfloat162*>(dst_base + (size_t)row * DH + col) =
          __nv_bfloat162{__float2bfloat16(v0), __float2bfloat16(v1)};
  }
}

template <class Acc, class CC>
__device__ __forceinline__ void qkv_epilogue_store(const Params& p, int n0, const Acc& acc, const CC& cC,
                                                   const QkvEpilogueOperands& op) {
  static_assert(size(Acc{}) == 32, "64x64 fragment");
  CUTE_UNROLL
  for (int i = 0; i < 8; ++i) {
    const float v[4] = {acc(4 * i), acc(4 * i + 1), acc(4 * i + 2), acc(4 * i + 3)};
    qkv_epilogue_group(p, n0, i, v, cC, op);
  }
}

// o_proj epilogue (contract 4.4): out = out + acc * g in fp32, bf16 store;
// each thread reads its own element before writing it (out may alias x).
template <class CC>
__device__ __forceinline__ void out_epilogue_group(const Params& p, int n0, int i, const float* v, const CC& cC,
                                                   __nv_bfloat162 g2) {
  const int row0 = get<0>(cC(0)), col = get<1>(cC(4 * i));
  CUTE_UNROLL
  for (int r = 0; r < 2; ++r) {
    const int row = row0 + 8 * r;
    __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)row * D + n0 + col);
    const __nv_bfloat162 res = *dst;
    const float o0 = bf2f(res.x) + v[2 * r] * bf2f(g2.x);
    const float o1 = bf2f(res.y) + v[2 * r + 1] * bf2f(g2.y);
    *dst = __nv_bfloat162{__float2bfloat16(o0), __float2bfloat16(o1)};
  }
}

template <class Acc, class CC>
__device__ __forceinline__ void out_epilogue_store(const Params& p, int n0, const Acc& acc, const CC& cC,
                                                   const __nv_bfloat162* gate2) {
  CUTE_UNROLL
  for (int i = 0; i < 8; ++i) {
    const float v[4] = {acc(4 * i), acc(4 * i + 1), acc(4 * i + 2), acc(4 * i + 3)};
    out_epilogue_group(p, n0, i, v, cC, gate2[i]);
  }
}

// ================================================================ kQkvProj
// task (column = n-tile, split = k-half): 8 stages of (x[:, k:k+64], W[k:k+64, n:n+64])

__device__ void qkv_weight_producer(const Params& p, const TaskDesc& t, int lane) {
  const int n0 = t.column * QKV_BN;
  for (int g = 0; g < QKV_TRIP; ++g) {
    const int s = g % QKV_DEPTH, k = t.split * (D / QKV_SPLIT) + g * QKV_BK;
    if (g >= QKV_DEPTH)
      wait_bar_wd(p.bars.raw(p.bars.empty_w(s)), ((g / QKV_DEPTH) - 1) & 1, kSiteQkvProdEmpty, g, p.dbg);
    __syncwarp();
    if (lane == 0) {
#ifdef ATTN_ABL_QKV_NO_W
      p.bars.full_w(s)->arrive_and_expect_tx(0);          // ablation: no weight traffic
#else
      p.bars.full_w(s)->arrive_and_expect_tx(QKV_F_B);
#ifdef ATTN_QKV_WT
      issue_tma_3d(p.tm_wqkv, p.pool + OFF_W + s * QKV_F_B, 0, n0, k / 64, p.bars.raw(p.bars.full_w(s)));
#else
      issue_tma_2d(p.tm_wqkv, p.pool + OFF_W + s * QKV_F_B, n0, k, p.bars.raw(p.bars.full_w(s)));
#endif
#endif
    }
  }
}

__device__ void qkv_act_producer(const Params& p, const TaskDesc& t, int lane) {
  for (int g = 0; g < QKV_TRIP; ++g) {
    const int s = g % QKV_DEPTH, k = t.split * (D / QKV_SPLIT) + g * QKV_BK;
    if (g >= QKV_DEPTH)
      wait_bar_wd(p.bars.raw(p.bars.empty_a(s)), ((g / QKV_DEPTH) - 1) & 1, kSiteQkvProdEmpty, g, p.dbg);
    __syncwarp();
    if (lane == 0) {
#ifdef ATTN_ABL_QKV_NO_X
      p.bars.full_a(s)->arrive_and_expect_tx(S_FRAME_B);  // ablation: no activation traffic
#else
      p.bars.full_a(s)->arrive_and_expect_tx(QKV_F_B + S_FRAME_B);
      issue_tma_3d(p.tm_x, p.pool + OFF_A + s * QKV_F_B, 0, 0, k / 64, p.bars.raw(p.bars.full_a(s)));
#endif
      issue_bulk_1d(p.pool + OFF_S + s * S_FRAME_B, p.ada_scale + k, S_FRAME_B,
                    p.bars.raw(p.bars.full_a(s)));
    }
  }
  if (p.standalone || QKV_SPLIT == 1 || t.split != 0 || lane != 0) return;
  // split 0: once the mainloop has retired the pool and the sibling has
  // published, stage its partial into the pool for the fold
  const int tile = t.column;
  wait_bar_wd(p.bars.raw(p.bars.drain()), 0, kSiteQkvProdEmpty, -1, p.dbg);
  counter_wait(&p.counters[CM::kQkvJoinBegin + tile], CM::kQkvJoinArrive, kSiteQkvJoin, tile, p.dbg);
  fence_proxy_async_global();
  p.bars.full_q()->arrive_and_expect_tx((QKV_SPLIT - 1) * QKV_PARTIAL_B);
  for (int sp = 1; sp < QKV_SPLIT; ++sp)
    issue_bulk_1d(p.pool + OFF_FOLD + (sp - 1) * QKV_PARTIAL_B,
                  p.qkv_partial + ((size_t)(sp - 1) * QKV_TILES + tile) * (64 * 64),
                  QKV_PARTIAL_B, p.bars.raw(p.bars.full_q()));
}

__device__ void qkv_math(const Params& p, const TaskDesc& t, int tid, int slot) {
  MmaQkv mma;
  auto thr = mma.get_thread_slice(tid);
  Tensor acc = partition_fragment_C(mma, Shape<Int<64>, Int<64>>{});
  Tensor cC = thr.partition_C(make_identity_tensor(Shape<Int<64>, Int<64>>{}));
  clear(acc);
  auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, mma);
  auto smem_thr_copy_A = smem_thr_copy_A_slice(smem_tiled_copy_A, tid);
  // (m, k) of every A-fragment element: the per-K scale needs k
  Tensor tCcA = thr.partition_A(make_identity_tensor(Shape<Int<64>, Int<QKV_BK>>{}));
  // TWO register A fragments, alternated by stage parity.  A wgmma reads its
  // register operands asynchronously; refilling the same registers for the
  // next stage before the group retired makes ptxas insert a WG.DP wait and
  // serialise every wgmma in the function (C7518).  wait_group<1> after
  // stage g retires g-1, so g+1 may overwrite g-1's fragment.
  Tensor sA_shape = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(p.pool + OFF_A)), SmemLayoutA_QKV{});
  Tensor tCrA0 = thr.partition_fragment_A(sA_shape);
  Tensor tCrA1 = thr.partition_fragment_A(sA_shape);

  auto stage = [&](int g, auto& tCrA) {
    const int s = g % QKV_DEPTH;
    const uint32_t ph = (g / QKV_DEPTH) & 1;
    wait_bar_wd(p.bars.raw(p.bars.full_a(s)), ph, kSiteQkvMathFull, g, p.dbg);
    wait_bar_wd(p.bars.raw(p.bars.full_w(s)), ph, kSiteQkvMathFull, g, p.dbg);
    if (g == 0) stamp(p, slot, 1);

    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(p.pool + OFF_A + s * QKV_F_B)), SmemLayoutA_QKV{});
    Tensor sW = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(p.pool + OFF_W + s * QKV_F_B)), SmemLayoutW_QKV{});
    Tensor tCsA = smem_thr_copy_A.partition_S(sA);
    Tensor tCrA_view = smem_thr_copy_A.retile_D(tCrA);
    cute::copy(smem_tiled_copy_A, tCsA, tCrA_view);                 // ldmatrix through the swizzle

    // contract 4.1: a = bf16(x * s), formed in registers.  Adjacent fragment
    // slots (i, i+1) hold columns (k, k+1), one bf16x2 of s.
#ifndef ATTN_ABL_QKV_NO_SCALE
    const __nv_bfloat162* S2 = reinterpret_cast<const __nv_bfloat162*>(p.pool + OFF_S + s * S_FRAME_B);
    CUTE_UNROLL
    for (int i = 0; i < size(tCrA); i += 2) {
      const int k = get<1>(tCcA(i));
      __nv_bfloat162 v{__ushort_as_bfloat16(tCrA(i).raw()), __ushort_as_bfloat16(tCrA(i + 1).raw())};
      v = __hmul2(v, S2[k >> 1]);
      tCrA(i) = BF::bitcast(__bfloat16_as_ushort(v.x));
      tCrA(i + 1) = BF::bitcast(__bfloat16_as_ushort(v.y));
    }
#endif
    Tensor tCrB = thr.make_fragment_B(thr.partition_B(sW));
#ifndef ATTN_ABL_QKV_NO_MMA
    ::sm90::gemm<false, -1, true, true>(mma, tCrA, tCrB, acc);
#endif
    // Release on retirement, one group outstanding: frame g-1 is free once
    // its batch retired (A was already consumed by ldmatrix; W by the wgmma).
    if (g >= 1) {
      warpgroup_wait<1>();
      __syncwarp();
      if ((tid & 31) == 0) {
        p.bars.empty_a((g - 1) % QKV_DEPTH)->arrive();
        p.bars.empty_w((g - 1) % QKV_DEPTH)->arrive();
      }
    }
  };
  static_assert(QKV_TRIP % 2 == 0, "stages alternate two register A fragments");
  // Fully unrolled so ptxas sees straight-line wgmma / wait_group order and
  // does not have to guard the register operands across a runtime loop.
  CUTE_UNROLL
  for (int g = 0; g < QKV_TRIP; g += 2) {
    stage(g, tCrA0);
    stage(g + 1, tCrA1);
  }
  warpgroup_wait<0>();
  __syncwarp();
  if ((tid & 31) == 0) {
    p.bars.empty_a((QKV_TRIP - 1) % QKV_DEPTH)->arrive();
    p.bars.empty_w((QKV_TRIP - 1) % QKV_DEPTH)->arrive();
  }
  stamp(p, slot, 2);
  // Every x read of this task has landed: o_proj may overwrite `out` (= x)
  // once all QKV_TASKS have passed this point.
  mathwg_sync();
  if (tid == 0 && !p.standalone) counter_release(&p.counters[CM::kQkvDone]);

  const int tile = t.column, n0 = tile * QKV_BN;
  if (QKV_SPLIT > 1 && (p.standalone || t.split != 0)) {
    // the pool is free once every math warp has retired its wgmma reads.
    // Task loop: siblings publish into slot (split-1) and split 0 folds them.
    // Standalone: every split publishes into slot `split`; qkv_reduce folds.
    mathwg_sync();
    stage_partial_f32_smem(p.pool, tid, acc);
    const int pslot = p.standalone ? t.split : t.split - 1;
    publish_staged(p, tid, p.qkv_partial + ((size_t)pslot * QKV_TILES + tile) * (64 * 64),
                   QKV_PARTIAL_B, nullptr, 0,
                   p.standalone ? nullptr : &p.counters[CM::kQkvJoinBegin + tile]);
    stamp(p, slot, 3); stamp(p, slot, 4);
    return;
  }
  // split 0: the act producer stages the sibling partial into the pool
  // (drain -> join counter -> bulk copy); fold it from shared memory in fixed
  // order, then own the epilogue.  Per-row / per-column epilogue operands are
  // loaded before the wait so their latency overlaps it.
  if (QKV_SPLIT > 1) arrive_drain(p, tid);
  // PDL: rms_factor is the ONE operand the prerequisite grid writes
  // (tl_rms_factor reads x and writes only the factor), so the whole
  // mainloop above -- weights, x, ada_scale -- legally ran pre-wait and this
  // is the first dependent read. Valid only while the immediate stream
  // predecessor does not write x; revisit if the chain is ever deepened.
  if (p.standalone && p.pdl) cudaGridDependencySynchronize();
  const QkvEpilogueOperands ops = qkv_epilogue_load(p, n0, cC);   // overlaps the wait
  if (QKV_SPLIT > 1) wait_bar_wd(p.bars.raw(p.bars.full_q()), 0, kSiteQkvJoin, tile, p.dbg);
  stamp(p, slot, 3);
  CUTE_UNROLL
  for (int sp = 1; sp < QKV_SPLIT; ++sp)
    fold_partial_f32_smem(p.pool + OFF_FOLD + (sp - 1) * QKV_PARTIAL_B, tid, acc);
  qkv_epilogue_store(p, n0, acc, cC, ops);
  mathwg_sync();
  if (tid == 0 && !p.standalone) counter_release(&p.counters[t.dependency]);
  stamp(p, slot, 4);
}

// ============================================================== kAttention
// task (column = head, split = kv-split): Q resident, 4 stages of 32 keys

// Cache rows [0, PREFIX_LEN) are read-only for the whole layer-step, so a
// frame that lies inside the prefix has no dependency and is issued at task
// start -- with attention dealt to CTAs that are idle during qkv, that is
// t = 0.  Only a frame overlapping the suffix rows qkv writes waits on kKv.
__device__ __forceinline__ bool frame_needs_kv(int key0, int g) {
  return key0 + (g + 1) * ATTN_BKK > PREFIX_LEN;
}

__device__ void attn_weight_producer(const Params& p, const TaskDesc& t, int lane) {
  const int key0 = t.split * (KEYS_PAD / ATTN_SPLIT);
  bool kv_ready = false;
  for (int g = 0; g < ATTN_TRIP; ++g) {
    const int s = g % ATTN_DEPTH;
    if (g >= ATTN_DEPTH)
      wait_bar_wd(p.bars.raw(p.bars.empty_w(s)), ((g / ATTN_DEPTH) - 1) & 1, kSiteAttnProdEmpty, g, p.dbg);
    __syncwarp();
    if (lane == 0) {
      // A frame overlapping the suffix rows qkv writes must wait: on the kKv
      // counter in the task loop, on the prerequisite grid under PDL. Pure
      // prefix frames issue pre-wait in both modes. Either way qkv's
      // generic-proxy cache stores need the proxy fence before this warp's
      // async-proxy TMA reads them.
      if (!kv_ready && frame_needs_kv(key0, g) && (!p.standalone || p.pdl)) {
        if (p.standalone) cudaGridDependencySynchronize();
        else counter_wait(&p.counters[CM::kKv], CM::kKvArrive, kSiteAttnDep, 0, p.dbg);
        fence_proxy_async_global();
        kv_ready = true;
      }
      p.bars.full_w(s)->arrive_and_expect_tx(KV_FRAME_B);
      issue_tma_3d(p.tm_k, p.pool + OFF_K + s * KV_FRAME_B, 0, key0 + g * ATTN_BKK, 0,
                   p.bars.raw(p.bars.full_w(s)));
    }
  }
}

__device__ void attn_act_producer(const Params& p, const TaskDesc& t, int lane) {
  const int head = t.column, key0 = t.split * (KEYS_PAD / ATTN_SPLIT);
  bool kv_ready = false;
  // V frames first, so the prefix frames are in flight before this warp
  // blocks on the Q counter; the mask slice (no dependency) is posted with
  // them and the Q load joins the same barrier once its head has published.
  if (lane == 0) {
    p.bars.full_q()->arrive_and_expect_tx(Q_FRAME_B + MASK_SLICE_B);
    issue_bulk_1d(p.pool + OFF_MASK, p.key_mask + key0, MASK_SLICE_B, p.bars.raw(p.bars.full_q()));
  }
  __syncwarp();
  for (int g = 0; g < ATTN_TRIP; ++g) {
    const int s = g % ATTN_DEPTH;
    if (g >= ATTN_DEPTH)
      wait_bar_wd(p.bars.raw(p.bars.empty_a(s)), ((g / ATTN_DEPTH) - 1) & 1, kSiteAttnProdEmpty, g, p.dbg);
    __syncwarp();
    if (lane == 0) {
      // Same suffix-frame dependency as the K producer above.
      if (!kv_ready && frame_needs_kv(key0, g) && (!p.standalone || p.pdl)) {
        if (p.standalone) cudaGridDependencySynchronize();
        else counter_wait(&p.counters[CM::kKv], CM::kKvArrive, kSiteAttnDep, 2, p.dbg);
        fence_proxy_async_global();
        kv_ready = true;
      }
      p.bars.full_a(s)->arrive_and_expect_tx(KV_FRAME_B);
      issue_tma_3d(p.tm_v, p.pool + OFF_V + s * KV_FRAME_B, 0, key0 + g * ATTN_BKK, 0,
                   p.bars.raw(p.bars.full_a(s)));
    }
  }
  if (lane == 0) {
    if (!p.standalone) {
      counter_wait(&p.counters[t.dependency], CM::kQArrive, kSiteAttnDep, 1, p.dbg);
      fence_proxy_async_global();
    } else if (p.pdl) {
      // q_buf is published by the prerequisite qkv grid via generic stores;
      // wait, then fence the generic->async proxy hop before the Q TMA.
      cudaGridDependencySynchronize();
      fence_proxy_async_global();
    }
    issue_tma_3d(p.tm_q, p.pool + OFF_Q, 0, head * M_PAD, 0, p.bars.raw(p.bars.full_q()));
  }
}

__device__ void attn_math(const Params& p, const TaskDesc& t, int tid, int slot) {
  const int head = t.column, key0 = t.split * (KEYS_PAD / ATTN_SPLIT);
  const int lane = tid & 31;

  MmaS mma_s;
  MmaO mma_o;
  auto thr_s = mma_s.get_thread_slice(tid);
  auto thr_o = mma_o.get_thread_slice(tid);
  Tensor acc_s = partition_fragment_C(mma_s, Shape<Int<64>, Int<ATTN_BKK>>{});
  Tensor acc_o = partition_fragment_C(mma_o, Shape<Int<64>, Int<DH>>{});
  Tensor cS = thr_s.partition_C(make_identity_tensor(Shape<Int<64>, Int<ATTN_BKK>>{}));
  Tensor cO = thr_o.partition_C(make_identity_tensor(Shape<Int<64>, Int<DH>>{}));
  constexpr int S_ELEMS = ATTN_BKK / 2;   // (2,2,N/8) fragment per thread
  static_assert(size(acc_s) == S_ELEMS && size(acc_o) == 128);
  clear(acc_o);

  // Online-softmax state for this thread's two rows (r0 and r0 + 8), log2 domain.
  float m_run[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float l_run[2] = {0.f, 0.f};                 // per-thread partial; quad-reduced at the end

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(p.pool + OFF_Q)), SmemLayoutQ{});
  Tensor tSrQ = thr_s.make_fragment_A(thr_s.partition_A(sQ));
  wait_bar_wd(p.bars.raw(p.bars.full_q()), 0, kSiteAttnMathFull, -1, p.dbg);
  stamp(p, slot, 1);

  __align__(16) BF p_regs[S_ELEMS];
  Tensor tOrP = make_tensor(make_rmem_ptr(p_regs), acc_to_aregs(acc_s.layout()));

  for (int g = 0; g < ATTN_TRIP; ++g) {
    const int s = g % ATTN_DEPTH;
    const uint32_t ph = (g / ATTN_DEPTH) & 1;

    // S = Q K^T over this stage's 32 keys; wait 0 also retires the previous
    // P.V batch, which is what makes the rescale below safe.
    wait_bar_wd(p.bars.raw(p.bars.full_w(s)), ph, kSiteAttnMathFull, g, p.dbg);
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(p.pool + OFF_K + s * KV_FRAME_B)), SmemLayoutK{});
    Tensor tSrK = thr_s.make_fragment_B(thr_s.partition_B(sK));
#ifdef ATTN_ABL_NO_S
    clear(acc_s); warpgroup_wait<0>();
#else
    ::sm90::gemm<true, 0, true, true>(mma_s, tSrQ, tSrK, acc_s);
#endif
    __syncwarp();
    if (lane == 0) {
      p.bars.empty_w(s)->arrive();                          // K frame g consumed
      if (g >= 1) p.bars.empty_a((g - 1) % ATTN_DEPTH)->arrive();  // V frame g-1 retired
    }

    // masked, scaled logits in the log2 domain, in place; the mask stays
    // finite so a fully masked tile yields a uniform row, never NaN.
    float rmax[2] = {-CUDART_INF_F, -CUDART_INF_F};
    CUTE_UNROLL
    for (int e = 0; e < S_ELEMS; e += 2) {
      const int col = get<1>(cS(e)), r = (e >> 1) & 1;
      const __nv_bfloat162 mk = *reinterpret_cast<const __nv_bfloat162*>(p.pool + OFF_MASK + (g * ATTN_BKK + col) * 2);
      acc_s(e)     = acc_s(e)     * SCALE_LOG2 + fmaxf(bf2f(mk.x) * 1.4426950408889634f, MASK_FLOOR);
      acc_s(e + 1) = acc_s(e + 1) * SCALE_LOG2 + fmaxf(bf2f(mk.y) * 1.4426950408889634f, MASK_FLOOR);
      rmax[r] = fmaxf(rmax[r], fmaxf(acc_s(e), acc_s(e + 1)));
    }
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) {
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 1));
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 2));
      const float m_new = fmaxf(m_run[r], rmax[r]);
      const float alpha = fast_exp2(m_run[r] - m_new);      // 0 on the first stage
      m_run[r] = m_new;
      l_run[r] *= alpha;
      CUTE_UNROLL
      for (int e = 0; e < 128; ++e)
        if (((e >> 1) & 1) == r) acc_o(e) *= alpha;
    }
#ifdef ATTN_ABL_NO_SOFTMAX
    CUTE_UNROLL
    for (int e = 0; e < S_ELEMS; ++e) { p_regs[e] = BF(acc_s(e)); }
    l_run[0] += 1.f; l_run[1] += 1.f;
#else
    CUTE_UNROLL
    for (int e = 0; e < S_ELEMS; ++e) {
      const int r = (e >> 1) & 1;
      const float pe = fast_exp2(acc_s(e) - m_run[r]);
      l_run[r] += pe;
      p_regs[e] = BF(pe);
    }
#endif

    // O += P V
    wait_bar_wd(p.bars.raw(p.bars.full_a(s)), ph, kSiteAttnMathFull, g, p.dbg);
    Tensor sV = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(p.pool + OFF_V + s * KV_FRAME_B)), SmemLayoutV{});
    Tensor tOrV = thr_o.make_fragment_B(thr_o.partition_B(sV));
#ifndef ATTN_ABL_NO_PV
    ::sm90::gemm<false, -1, true, true>(mma_o, tOrP, tOrV, acc_o);
#else
    acc_o(0) += float(p_regs[0]);
#endif
  }
  warpgroup_wait<0>();
  __syncwarp();
  if (lane == 0) p.bars.empty_a((ATTN_TRIP - 1) % ATTN_DEPTH)->arrive();
  stamp(p, slot, 2);

  // row sums across the quad
  CUTE_UNROLL
  for (int r = 0; r < 2; ++r) {
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 1);
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 2);
  }

  // every split publishes its normalised O and (m, l), row-major; the
  // kCombine tasks fold the eight of them fd_combine-style
  // Staged row-major in the freed pool (32 KB O + 512 B (m, l)) and written
  // with one bulk group.  The 512 B row pitch costs an 8-way bank conflict on
  // the staging stores, ~0.3 us, against the 3.9 us the scattered publish took.
  mathwg_sync();                                // every warp's wgmma reads retired
  constexpr int PARTIAL_B = M_PAD * DH * 2, LSE_B = M_PAD * 2 * 4;
  __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(p.pool);
  float* sL = reinterpret_cast<float*>(p.pool + PARTIAL_B);
  const float inv[2] = {1.f / l_run[0], 1.f / l_run[1]};
  CUTE_UNROLL
  for (int e = 0; e < 128; e += 2) {
    const int row = get<0>(cO(e)), col = get<1>(cO(e)), r = (e >> 1) & 1;
    *reinterpret_cast<__nv_bfloat162*>(sP + row * DH + col) =
        __nv_bfloat162{__float2bfloat16(acc_o(e) * inv[r]), __float2bfloat16(acc_o(e + 1) * inv[r])};
  }
  if ((lane & 3) == 0) {
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) {
      const int row = get<0>(cO(2 * r));
      sL[row * 2] = m_run[r];
      sL[row * 2 + 1] = l_run[r];
    }
  }
  publish_staged(p, tid,
                 p.attn_partial + ((size_t)t.split * H + head) * (M_PAD * DH), PARTIAL_B,
                 p.attn_lse + ((size_t)t.split * H + head) * (M_PAD * 2), LSE_B,
                 p.standalone ? nullptr : &p.counters[CM::kAttnBegin + head]);
  stamp(p, slot, 3); stamp(p, slot, 4);
}

// ================================================================= kCombine
// task (column = head, split = 8-row group): o_buf[h][rows] = sum_s w_s O_s / sum_s w_s,
// w_s = exp2(m_s - max_s m_s) * l_s.  128 math threads, one 16-column strip
// of one row each; every load is independent and 16 B, 8 x 4 KB contiguous.
__device__ void combine_math(const Params& p, const TaskDesc& t, int tid, int slot) {
  const int head = t.column;
  const int row = t.split * COMBINE_ROWS + (tid >> 4), c0 = (tid & 15) * 16;
  static_assert(COMBINE_ROWS * (DH / 16) == kMathThreads, "one strip per math thread");
  if (tid == 0 && !p.standalone)
    counter_wait(&p.counters[t.dependency], CM::kAttnArrive, kSiteAttnJoin, head, p.dbg);
  mathwg_sync();
  stamp(p, slot, 1);
  float m_s[ATTN_SPLIT], l_s[ATTN_SPLIT];
  float m_max = -CUDART_INF_F;
  CUTE_UNROLL
  for (int sp = 0; sp < ATTN_SPLIT; ++sp) {
    const float2 ml = __ldcg(reinterpret_cast<const float2*>(
        p.attn_lse + ((size_t)sp * H + head) * (M_PAD * 2) + row * 2));
    m_s[sp] = ml.x; l_s[sp] = ml.y;
    m_max = fmaxf(m_max, ml.x);
  }
  uint4 raw[ATTN_SPLIT][2];
  CUTE_UNROLL
  for (int sp = 0; sp < ATTN_SPLIT; ++sp) {
    const uint4* src = reinterpret_cast<const uint4*>(
        p.attn_partial + ((size_t)sp * H + head) * (M_PAD * DH) + (size_t)row * DH + c0);
    raw[sp][0] = __ldcg(src);
    raw[sp][1] = __ldcg(src + 1);
  }
  float acc[16] = {};
  float l_tot = 0.f;
  CUTE_UNROLL
  for (int sp = 0; sp < ATTN_SPLIT; ++sp) {
    const float w = fast_exp2(m_s[sp] - m_max) * l_s[sp];
    l_tot += w;
    CUTE_UNROLL
    for (int q = 0; q < 2; ++q) {
      const __nv_bfloat162* pr = reinterpret_cast<const __nv_bfloat162*>(&raw[sp][q]);
      CUTE_UNROLL
      for (int k = 0; k < 4; ++k) {
        acc[q * 8 + 2 * k]     += w * bf2f(pr[k].x);
        acc[q * 8 + 2 * k + 1] += w * bf2f(pr[k].y);
      }
    }
  }
  stamp(p, slot, 2); stamp(p, slot, 3);
  const float inv = 1.f / l_tot;
  __nv_bfloat16* dst = p.o_buf + ((size_t)(head * M_PAD + row)) * DH + c0;
  CUTE_UNROLL
  for (int q = 0; q < 2; ++q) {
    uint4 v;
    __nv_bfloat162* pr = reinterpret_cast<__nv_bfloat162*>(&v);
    CUTE_UNROLL
    for (int k = 0; k < 4; ++k)
      pr[k] = __nv_bfloat162{__float2bfloat16(acc[q * 8 + 2 * k] * inv),
                             __float2bfloat16(acc[q * 8 + 2 * k + 1] * inv)};
    *reinterpret_cast<uint4*>(dst + q * 8) = v;
  }
  mathwg_sync();
  if (tid == 0 && !p.standalone) counter_release(&p.counters[CM::kOBegin + head]);
  stamp(p, slot, 4);
}

// ================================================================ kOutProj
// task (column = n-tile, split = head): 4 stages of (o_buf[h][:, k:k+64], W_o[h*256+k : +64, n:n+64])

__device__ void out_weight_producer(const Params& p, const TaskDesc& t, int lane) {
  // dependency-free: the weight ring runs ahead of the head-combined wait
  const int n0 = t.column * OUT_BN;
  for (int g = 0; g < OUT_TRIP; ++g) {
    const int s = g % OUT_DEPTH, k = t.split * DH + g * OUT_BK;
    if (g >= OUT_DEPTH)
      wait_bar_wd(p.bars.raw(p.bars.empty_w(s)), ((g / OUT_DEPTH) - 1) & 1, kSiteOutProdEmpty, g, p.dbg);
    __syncwarp();
    if (lane == 0) {
      p.bars.full_w(s)->arrive_and_expect_tx(F256_B);
      issue_tma_2d(p.tm_wo, p.pool + OFF_W + s * F256_B, n0, k, p.bars.raw(p.bars.full_w(s)));
    }
  }
}

__device__ void out_act_producer(const Params& p, const TaskDesc& t, int lane) {
  const int head = t.split;
  if (lane == 0 && !p.standalone) {
    counter_wait(&p.counters[CM::kOBegin + head], CM::kOArrive, kSiteOutDep, head, p.dbg);
    fence_proxy_async_global();
  }
  __syncwarp();
  for (int g = 0; g < OUT_TRIP; ++g) {
    const int s = g % OUT_DEPTH;
    if (g >= OUT_DEPTH)
      wait_bar_wd(p.bars.raw(p.bars.empty_a(s)), ((g / OUT_DEPTH) - 1) & 1, kSiteOutProdEmpty, g, p.dbg);
    __syncwarp();
    if (lane == 0) {
      p.bars.full_a(s)->arrive_and_expect_tx(F256_B);
      issue_tma_3d(p.tm_o, p.pool + OFF_A + s * F256_B, 0, head * M_PAD, g * (OUT_BK / 64),
                   p.bars.raw(p.bars.full_a(s)));
    }
  }
  if (p.standalone || t.split != 0 || lane != 0) return;
  // split 0: stage the 7 head partials once drained, joined, and every qkv
  // task has finished reading x (`out` may alias it)
  const int tile = t.column;
  wait_bar_wd(p.bars.raw(p.bars.drain()), 0, kSiteOutProdEmpty, -1, p.dbg);
  counter_wait(&p.counters[CM::kOutBegin + tile], CM::kOutArrive, kSiteOutJoin, tile, p.dbg);
  counter_wait(&p.counters[CM::kQkvDone], CM::kQkvDoneArrive, kSiteOutQkvDone, tile, p.dbg);
  fence_proxy_async_global();
  p.bars.full_q()->arrive_and_expect_tx((OUT_SPLIT - 1) * OUT_PARTIAL_B);
  for (int sp = 1; sp < OUT_SPLIT; ++sp)
    issue_bulk_1d(p.pool + OFF_FOLD + (sp - 1) * OUT_PARTIAL_B,
                  p.out_partial + ((size_t)(sp - 1) * OUT_TILES + tile) * (64 * 64),
                  OUT_PARTIAL_B, p.bars.raw(p.bars.full_q()));
}

__device__ void out_math(const Params& p, const TaskDesc& t, int tid, int slot) {
  MmaProj mma;
  auto thr = mma.get_thread_slice(tid);
  Tensor acc = partition_fragment_C(mma, Shape<Int<64>, Int<64>>{});
  Tensor cC = thr.partition_C(make_identity_tensor(Shape<Int<64>, Int<64>>{}));
  clear(acc);

  for (int g = 0; g < OUT_TRIP; ++g) {
    const int s = g % OUT_DEPTH;
    const uint32_t ph = (g / OUT_DEPTH) & 1;
    wait_bar_wd(p.bars.raw(p.bars.full_a(s)), ph, kSiteOutMathFull, g, p.dbg);
    wait_bar_wd(p.bars.raw(p.bars.full_w(s)), ph, kSiteOutMathFull, g, p.dbg);
    if (g == 0) stamp(p, slot, 1);
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(p.pool + OFF_A + s * F256_B)), SmemLayoutA256{});
    Tensor sW = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(p.pool + OFF_W + s * F256_B)), SmemLayoutW256{});
    Tensor tCrA = thr.make_fragment_A(thr.partition_A(sA));
    Tensor tCrB = thr.make_fragment_B(thr.partition_B(sW));
    ::sm90::gemm<false, -1, true, true>(mma, tCrA, tCrB, acc);
    if (g >= 1) {
      warpgroup_wait<1>();
      __syncwarp();
      if ((tid & 31) == 0) {
        p.bars.empty_a((g - 1) % OUT_DEPTH)->arrive();
        p.bars.empty_w((g - 1) % OUT_DEPTH)->arrive();
      }
    }
  }
  warpgroup_wait<0>();
  __syncwarp();
  if ((tid & 31) == 0) {
    p.bars.empty_a((OUT_TRIP - 1) % OUT_DEPTH)->arrive();
    p.bars.empty_w((OUT_TRIP - 1) % OUT_DEPTH)->arrive();
  }
  stamp(p, slot, 2);

  const int tile = t.column, n0 = tile * OUT_BN, head = t.split;
  // Last-arriver single-launch o_proj measured 8.7 us against 8.1 for split +
  // reduce (job 556329): the last CTA folds 128 KB alone.  Kept for the
  // record, off by default.
  constexpr bool kOutLastArriver = false;
  if (p.standalone && kOutLastArriver) {
    // Single-launch split-K: every head publishes its partial, and the LAST
    // one to arrive on the tile's counter folds the other seven and runs the
    // epilogue.  Nobody waits, so there is no residency requirement, and the
    // last arriver resets the counter so a replay needs no host memset.
    mathwg_sync();
    stage_partial_f32_smem(p.pool, tid, acc);
    cutlass::arch::fence_view_async_shared();
    mathwg_sync();
    uint32_t* flag = reinterpret_cast<uint32_t*>(p.pool + OFF_MASK);   // unused by o_proj
    if (tid == 0) {
      issue_bulk_store_1d(p.out_partial + ((size_t)head * OUT_TILES + tile) * (64 * 64), p.pool, OUT_PARTIAL_B);
      bulk_store_commit_and_wait();
      fence_proxy_async_global();
      uint32_t* c = &p.counters[CM::kSaOutBegin + tile];
      const uint32_t old = counter_arrive_acq_rel(c);
      const bool last = old == OUT_SPLIT - 1;
      if (last) *c = 0u;
      *flag = last ? 1u : 0u;
    }
    mathwg_sync();                                  // thread 0's acquire covers the CTA
    const bool last = *flag != 0u;
    stamp(p, slot, 3);
    if (!last) { stamp(p, slot, 4); return; }
    __nv_bfloat162 gate2[8];
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i)
      gate2[i] = *reinterpret_cast<const __nv_bfloat162*>(p.ada_gate + n0 + get<1>(cC(4 * i)));
    // Fold ALL eight partials in head order, own included (its bulk store
    // is complete), so the sum is bit-identical whichever head arrives last.
    clear(acc);
    CUTE_UNROLL
    for (int sp = 0; sp < OUT_SPLIT; ++sp)
      fold_partial_f32(p.out_partial + ((size_t)sp * OUT_TILES + tile) * (64 * 64), tid, acc);
    out_epilogue_store(p, n0, acc, cC, gate2);
    stamp(p, slot, 4);
    return;
  }
  if (p.standalone || head != 0) {
    mathwg_sync();
    stage_partial_f32_smem(p.pool, tid, acc);
    const int pslot = p.standalone ? head : head - 1;
    publish_staged(p, tid, p.out_partial + ((size_t)pslot * OUT_TILES + tile) * (64 * 64),
                   OUT_PARTIAL_B, nullptr, 0,
                   p.standalone ? nullptr : &p.counters[CM::kOutBegin + tile]);
    stamp(p, slot, 3); stamp(p, slot, 4);
    return;
  }
  // split 0: the act producer stages the 7 partials (after the join and the
  // qkv-done wait that guards the x alias); fold from shared memory
  arrive_drain(p, tid);
  __nv_bfloat162 gate2[8];
  CUTE_UNROLL
  for (int i = 0; i < 8; ++i)
    gate2[i] = *reinterpret_cast<const __nv_bfloat162*>(p.ada_gate + n0 + get<1>(cC(4 * i)));
  wait_bar_wd(p.bars.raw(p.bars.full_q()), 0, kSiteOutJoin, tile, p.dbg);
  stamp(p, slot, 3);
  CUTE_UNROLL
  for (int sp = 1; sp < OUT_SPLIT; ++sp)
    fold_partial_f32_smem(p.pool + OFF_FOLD + (sp - 1) * OUT_PARTIAL_B, tid, acc);
  out_epilogue_store(p, n0, acc, cC, gate2);
  stamp(p, slot, 4);
}

// ================================================================== kernels
#define ATTN_KERNEL_PARAMS                                                              \
    const __grid_constant__ CUtensorMap tm_x, const __grid_constant__ CUtensorMap tm_wqkv, \
    const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k,    \
    const __grid_constant__ CUtensorMap tm_v, const __grid_constant__ CUtensorMap tm_o,    \
    const __grid_constant__ CUtensorMap tm_wo,                                             \
    const __nv_bfloat16* __restrict__ rms_factor, const __nv_bfloat16* __restrict__ ada_scale, \
    const __nv_bfloat16* __restrict__ qkv_bias, const __nv_bfloat16* __restrict__ rope,    \
    const __nv_bfloat16* __restrict__ key_mask, const __nv_bfloat16* __restrict__ ada_gate, \
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, __nv_bfloat16* out,                    \
    __nv_bfloat16* q_buf, __nv_bfloat16* o_buf,                                            \
    float* qkv_partial, __nv_bfloat16* attn_partial, float* attn_lse, float* out_partial,  \
    uint32_t* counters, long long* dbg, long long* timeline, int pdl_flag
#define ATTN_MAKE_PARAMS(pool_, standalone_)                                            \
    Params{&tm_x, &tm_wqkv, &tm_q, &tm_k, &tm_v, &tm_o, &tm_wo,                         \
           rms_factor, ada_scale, qkv_bias, rope, key_mask, ada_gate,                    \
           k_cache, v_cache, out, q_buf, o_buf,                                          \
           qkv_partial, attn_partial, attn_lse, out_partial, counters, dbg, timeline,    \
           standalone_, pdl_flag != 0, pool_,                                            \
           Bars{(pool_) ? reinterpret_cast<uint64_t*>((pool_) + OFF_BARS) : nullptr}}

// One task: fresh rings (one producer arrival + tx per full barrier, four
// math-warp arrivals per empty barrier, one-shot Q barrier), dispatch by
// kind and warp role, then a CTA barrier so the pool may be re-imaged.
__device__ __forceinline__ void run_task(const Params& p, const TaskDesc& t, int tid, int slot) {
  const int warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < DEPTH_MAX; ++i) {
      p.bars.full_a(i)->init(1);
      p.bars.empty_a(i)->init(4);
      p.bars.full_w(i)->init(1);
      p.bars.empty_w(i)->init(4);
    }
    p.bars.full_q()->init(1);
    p.bars.drain()->init(4);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  stamp(p, slot, 0);

  if (hdr::is_qkv(t.kind)) {
    if (warp < 4)                 qkv_math(p, t, tid, slot);
    else if (warp == kWeightWarp) qkv_weight_producer(p, t, lane);
    else if (warp == kActWarp)    qkv_act_producer(p, t, lane);
  } else if (hdr::is_attention(t.kind)) {
    if (warp < 4)                 attn_math(p, t, tid, slot);
    else if (warp == kWeightWarp) attn_weight_producer(p, t, lane);
    else if (warp == kActWarp)    attn_act_producer(p, t, lane);
  } else if (hdr::is_combine(t.kind)) {
    if (warp < 4)                 combine_math(p, t, tid, slot);   // producers idle
  } else {
    if (warp < 4)                 out_math(p, t, tid, slot);
    else if (warp == kWeightWarp) out_weight_producer(p, t, lane);
    else if (warp == kActWarp)    out_act_producer(p, t, lane);
  }
  __syncthreads();
}

__global__ void __launch_bounds__(THREADS, 1)
attn_taskloop_kernel(const TaskDesc* __restrict__ table, ATTN_KERNEL_PARAMS) {
  extern __shared__ __align__(1024) uint8_t pool[];
  const Params p = ATTN_MAKE_PARAMS(pool, false);
  const TaskDesc* my = table + blockIdx.x * TASK_SLOTS;
  for (int slot = 0; slot < TASK_SLOTS; ++slot) {
    const TaskDesc t = my[slot];
    if (hdr::is_sentinel(t.kind)) continue;    // idle slot; later slots may still hold work
    run_task(p, t, threadIdx.x, slot);
  }
}

// Standalone kernels: the same task bodies, one task per CTA, dependencies by
// launch order.  KIND: 0 qkv (grid QKV_TASKS), 1 attention (ATTN_TASKS),
// 2 o_proj (OUT_TASKS), 3 combine (COMBINE_TASKS).
template <int KIND>
__global__ void __launch_bounds__(THREADS, 1)
attn_standalone_kernel(ATTN_KERNEL_PARAMS) {
  extern __shared__ __align__(1024) uint8_t pool[];
  const Params p = ATTN_MAKE_PARAMS(pool, true);
  // Fire the programmatic-launch trigger at entry: the dependent grid's
  // griddepcontrol.wait carries correctness (it spans full completion and
  // visibility), the trigger only schedules, and with no attribute-carrying
  // successor it is a no-op. Entry placement maximizes the overlap window;
  // the qkv->attention pair fits co-resident (40 + 64 CTAs < 132 SMs at
  // 1 CTA/SM).
  if (threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
  const int c = static_cast<int>(blockIdx.x);
  TaskDesc t;
  if constexpr (KIND == 0)      t = {TaskKind::kQkvProj,   c / QKV_SPLIT,     0, c % QKV_SPLIT};
  else if constexpr (KIND == 1) t = {TaskKind::kAttention, c / ATTN_SPLIT,    0, c % ATTN_SPLIT};
  else if constexpr (KIND == 2) t = {TaskKind::kOutProj,   c / OUT_SPLIT,     0, c % OUT_SPLIT};
  else                          t = {TaskKind::kCombine,   c / (M_PAD / COMBINE_ROWS), 0, c % (M_PAD / COMBINE_ROWS)};
  run_task(p, t, threadIdx.x, 0);
}

// Reduce kernels for the projections: grid = tiles x 8 fragment groups.  A
// CTA folds one 16 B group per thread from every split (16 KB per CTA, one
// float4 per split per thread) and runs that group's epilogue.  The
// per-tile form (16 CTAs x 128 KB) ran at one SM's bandwidth: 6.1 us
// (job 556125); this spreads the same bytes over 8x the SMs.
constexpr int REDUCE_GROUPS = 8;

__global__ void __launch_bounds__(kMathThreads)
qkv_reduce_kernel(ATTN_KERNEL_PARAMS) {
  const Params p = ATTN_MAKE_PARAMS(static_cast<uint8_t*>(nullptr), true);
  const int tid = threadIdx.x, tile = blockIdx.x / REDUCE_GROUPS, i = blockIdx.x % REDUCE_GROUPS;
  const int n0 = tile * QKV_BN;
  MmaProj mma;
  auto thr = mma.get_thread_slice(tid);
  Tensor cC = thr.partition_C(make_identity_tensor(Shape<Int<64>, Int<64>>{}));
  const QkvEpilogueOperands ops = qkv_epilogue_load(p, n0, cC);
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  CUTE_UNROLL
  for (int sp = 0; sp < QKV_SPLIT; ++sp) {
    const float4 q = __ldcg(reinterpret_cast<const float4*>(
        p.qkv_partial + ((size_t)sp * QKV_TILES + tile) * (64 * 64) + i * (kMathThreads * 4) + tid * 4));
    v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
  }
  qkv_epilogue_group(p, n0, i, v, cC, ops);
}

__global__ void __launch_bounds__(kMathThreads)
out_reduce_kernel(ATTN_KERNEL_PARAMS) {
  const Params p = ATTN_MAKE_PARAMS(static_cast<uint8_t*>(nullptr), true);
  const int tid = threadIdx.x, tile = blockIdx.x / REDUCE_GROUPS, i = blockIdx.x % REDUCE_GROUPS;
  const int n0 = tile * OUT_BN;
  MmaProj mma;
  auto thr = mma.get_thread_slice(tid);
  Tensor cC = thr.partition_C(make_identity_tensor(Shape<Int<64>, Int<64>>{}));
  const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(p.ada_gate + n0 + get<1>(cC(4 * i)));
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  CUTE_UNROLL
  for (int sp = 0; sp < OUT_SPLIT; ++sp) {
    const float4 q = __ldcg(reinterpret_cast<const float4*>(
        p.out_partial + ((size_t)sp * OUT_TILES + tile) * (64 * 64) + i * (kMathThreads * 4) + tid * 4));
    v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
  }
  out_epilogue_group(p, n0, i, v, cC, g2);
}

// Standalone combine at fd_combine granularity: 2 rows per CTA, 256 CTAs;
// each of 128 threads owns a 4-column strip of one row (8 B per split).
constexpr int SA_COMBINE_ROWS = 2;
// kTokenMajor: the combined rows go to p.out read as (M, H * DH) token-major
// -- the layout the TileLang o_proj consumes -- and only the M real rows are
// written.  Otherwise o_buf, head-major, all M_PAD rows (contract 3.3).
template <bool kTokenMajor>
__global__ void __launch_bounds__(kMathThreads)
combine_rows_kernel(ATTN_KERNEL_PARAMS) {
  const Params p = ATTN_MAKE_PARAMS(static_cast<uint8_t*>(nullptr), true);
  const int tid = threadIdx.x, head = blockIdx.x / (M_PAD / SA_COMBINE_ROWS);
  const int row = (blockIdx.x % (M_PAD / SA_COMBINE_ROWS)) * SA_COMBINE_ROWS + tid / (DH / 4);
  const int c0 = (tid % (DH / 4)) * 4;
  if (kTokenMajor && row >= M) return;
  // PDL: the very next loads read attn_lse/attn_partial from the prerequisite
  // attention grid. Rows past M exited above without waiting -- legal, they
  // read nothing. The prerequisite's bulk-store flush guarantee carries over:
  // griddepcontrol.wait spans full grid completion.
  if (p.pdl) cudaGridDependencySynchronize();
  float m_s[ATTN_SPLIT], l_s[ATTN_SPLIT];
  float m_max = -CUDART_INF_F;
  CUTE_UNROLL
  for (int sp = 0; sp < ATTN_SPLIT; ++sp) {
    const float2 ml = __ldcg(reinterpret_cast<const float2*>(
        p.attn_lse + ((size_t)sp * H + head) * (M_PAD * 2) + row * 2));
    m_s[sp] = ml.x; l_s[sp] = ml.y;
    m_max = fmaxf(m_max, ml.x);
  }
  uint2 raw[ATTN_SPLIT];
  CUTE_UNROLL
  for (int sp = 0; sp < ATTN_SPLIT; ++sp)
    raw[sp] = __ldcg(reinterpret_cast<const uint2*>(
        p.attn_partial + ((size_t)sp * H + head) * (M_PAD * DH) + (size_t)row * DH + c0));
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  float l_tot = 0.f;
  CUTE_UNROLL
  for (int sp = 0; sp < ATTN_SPLIT; ++sp) {
    const float w = fast_exp2(m_s[sp] - m_max) * l_s[sp];
    l_tot += w;
    const __nv_bfloat162* pr = reinterpret_cast<const __nv_bfloat162*>(&raw[sp]);
    acc[0] += w * bf2f(pr[0].x); acc[1] += w * bf2f(pr[0].y);
    acc[2] += w * bf2f(pr[1].x); acc[3] += w * bf2f(pr[1].y);
  }
  const float inv = 1.f / l_tot;
  uint2 v;
  __nv_bfloat162* pr = reinterpret_cast<__nv_bfloat162*>(&v);
  pr[0] = __nv_bfloat162{__float2bfloat16(acc[0] * inv), __float2bfloat16(acc[1] * inv)};
  pr[1] = __nv_bfloat162{__float2bfloat16(acc[2] * inv), __float2bfloat16(acc[3] * inv)};
  __nv_bfloat16* dst = kTokenMajor ? p.out + ((size_t)row * H + head) * DH + c0
                                   : p.o_buf + ((size_t)(head * M_PAD + row)) * DH + c0;
  *reinterpret_cast<uint2*>(dst) = v;
}

// ------------------------------------------------------------------ host
static CUtensorMap enc2d(const void* ptr, uint64_t inner, uint64_t outer,
                         uint32_t box_inner, uint32_t box_outer, CUresult* rc) {
  CUtensorMap m{};
  uint64_t dims[2] = {inner, outer};
  uint64_t strides[1] = {inner * 2};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t es[2] = {1, 1};
  *rc = cuTensorMapEncodeTiled(
      &m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides,
      box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return m;
}

// (rows, cols) row-major viewed as {64 elems, rows, cols/64 chunks}: one box of
// {64, box_rows, box_chunks} lands as [chunk][row][64], the CuTe SW128 K-major
// image of a (box_rows x 64*box_chunks) tile, whatever the row pitch.
static CUtensorMap enc3d_chunks(const void* ptr, uint64_t cols, uint64_t rows, uint32_t box_rows,
                                uint32_t box_chunks, CUresult* rc) {
  CUtensorMap m{};
  uint64_t dims[3] = {64, rows, cols / 64};
  uint64_t strides[2] = {cols * 2, 64 * 2};
  uint32_t box[3] = {64, box_rows, box_chunks};
  uint32_t es[3] = {1, 1, 1};
  *rc = cuTensorMapEncodeTiled(
      &m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides,
      box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return m;
}

struct Maps {
  CUtensorMap x, wqkv, q, k, v, o, wo;
};

// A null operand pointer leaves its map zeroed: a standalone op passes only
// the operands it reads (the qkv op has no w_o, the attention ops no x).  The
// task loop passes all of them.
static int encode_maps(Maps& m, const void* x, const void* w_qkv, const void* q_buf,
                       const void* k_cache, const void* v_cache, const void* o_buf, const void* w_o) {
  CUresult rc = CUDA_SUCCESS;
  m = Maps{};
  if (x) { m.x = enc3d_chunks(x, D, M_PAD, M_PAD, QKV_BK / 64, &rc);               if (rc) return 1000 + (int)rc; }
#ifdef ATTN_QKV_WT
  if (w_qkv) { m.wqkv = enc3d_chunks(w_qkv, D, QKV_W, 64, QKV_BK / 64, &rc);  if (rc) return 1000 + (int)rc; }
#else
  if (w_qkv) { m.wqkv = enc2d(w_qkv, QKV_W, D, 64, QKV_BK, &rc);              if (rc) return 1000 + (int)rc; }
#endif
  if (q_buf) { m.q = enc3d_chunks(q_buf, DH, (uint64_t)H * M_PAD, M_PAD, DH / 64, &rc);  if (rc) return 1000 + (int)rc; }
  if (k_cache) { m.k = enc3d_chunks(k_cache, DH, KEYS_PAD, ATTN_BKK, DH / 64, &rc);      if (rc) return 1000 + (int)rc; }
  if (v_cache) { m.v = enc3d_chunks(v_cache, DH, KEYS_PAD, ATTN_BKK, DH / 64, &rc);      if (rc) return 1000 + (int)rc; }
  if (o_buf) { m.o = enc3d_chunks(o_buf, DH, (uint64_t)H * M_PAD, M_PAD, DH / 64, &rc);  if (rc) return 1000 + (int)rc; }
  if (w_o) { m.wo = enc2d(w_o, D, (uint64_t)H * DH, 64, OUT_BK, &rc);          if (rc) return 1000 + (int)rc; }
  return 0;
}

template <class Kernel>
static int set_smem_once(Kernel kernel, bool& done) {
  if (done) return 0;
  cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, hdr::SMEM_B);
  if (e != cudaSuccess) return 1200 + (int)e;
  done = true;
  return 0;
}

}  // namespace attn

#define ATTN_HOST_ARGS(m)                                                                 \
    m.x, m.wqkv, m.q, m.k, m.v, m.o, m.wo,                                                 \
    (const __nv_bfloat16*)rms_factor, (const __nv_bfloat16*)ada_scale,                     \
    (const __nv_bfloat16*)qkv_bias, (const __nv_bfloat16*)rope,                            \
    (const __nv_bfloat16*)key_mask, (const __nv_bfloat16*)ada_gate,                        \
    (__nv_bfloat16*)k_cache, (__nv_bfloat16*)v_cache, (__nv_bfloat16*)out,                 \
    (__nv_bfloat16*)q_buf, (__nv_bfloat16*)o_buf,                                          \
    (float*)qkv_partial, (__nv_bfloat16*)attn_partial, (float*)attn_lse, (float*)out_partial, \
    (uint32_t*)counters, (long long*)dbg, (long long*)timeline

extern "C" int attn_taskloop_launch(
    const void* table, int n_ctas, int prefix_len,
    const void* x, const void* rms_factor, const void* ada_scale,
    const void* w_qkv, const void* qkv_bias, const void* rope,
    const void* key_mask, const void* w_o, const void* ada_gate,
    void* k_cache, void* v_cache, void* out,
    void* q_buf, void* o_buf,
    void* qkv_partial, void* attn_partial, void* attn_lse, void* out_partial,
    void* counters, void* dbg, void* timeline, void* stream) {
  using namespace attn;
  if (n_ctas != N_CTAS) return 1101;
  if (prefix_len != PREFIX_LEN) return 1102;   // wrong cache rows would be a silent wrong action
  static bool attr_set = false;
  if (int rc = set_smem_once(attn_taskloop_kernel, attr_set)) return rc;
  Maps m;
  if (int rc = encode_maps(m, x, w_qkv, q_buf, k_cache, v_cache, o_buf, w_o)) return rc;
  attn_taskloop_kernel<<<N_CTAS, THREADS, hdr::SMEM_B, (cudaStream_t)stream>>>(
      (const TaskDesc*)table, ATTN_HOST_ARGS(m), 0);
  return (int)cudaGetLastError();
}

// Standalone ops, launched in this order by the caller:
//   0 qkv split (QKV_TASKS CTAs)     1 qkv reduce (QKV_TILES CTAs)
//   2 attention split (ATTN_TASKS)   3 combine (COMBINE_TASKS)
//   4 o_proj split (OUT_TASKS)       5 o_proj reduce (OUT_TILES)
extern "C" int attn_standalone_launch(
    int op, int prefix_len, int use_programmatic_dependency,
    const void* x, const void* rms_factor, const void* ada_scale,
    const void* w_qkv, const void* qkv_bias, const void* rope,
    const void* key_mask, const void* w_o, const void* ada_gate,
    void* k_cache, void* v_cache, void* out,
    void* q_buf, void* o_buf,
    void* qkv_partial, void* attn_partial, void* attn_lse, void* out_partial,
    void* counters, void* dbg, void* timeline, void* stream) {
  using namespace attn;
  if (prefix_len != PREFIX_LEN) return 1102;
  if (use_programmatic_dependency < 0 || use_programmatic_dependency > 1) return 1104;
  static bool attr[4] = {false, false, false, false};
  Maps m;
  if (int rc = encode_maps(m, x, w_qkv, q_buf, k_cache, v_cache, o_buf, w_o)) return rc;
  cudaStream_t st = (cudaStream_t)stream;
  // PDL applies to the production chain only (ops 0, 2, 6): the attribute
  // lets the grid begin under its stream predecessor, and the kernel-side
  // `pdl_flag` arms the griddepcontrol waits at every dependent read. Ops
  // 1/3/4/5 always launch plainly -- their dependency structure is not
  // audited for early launch.
  const bool pdl = use_programmatic_dependency != 0;
  const int pdl_flag = pdl ? 1 : 0;
  cudaLaunchAttribute dependency_attribute{};
  dependency_attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  dependency_attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t launch_config{};
  launch_config.stream = st;
  launch_config.attrs = &dependency_attribute;
  launch_config.numAttrs = 1;
  switch (op) {
    case 0:
      if (int rc = set_smem_once(attn_standalone_kernel<0>, attr[0])) return rc;
      if (pdl) {
        launch_config.gridDim = dim3(QKV_TASKS, 1, 1);
        launch_config.blockDim = dim3(THREADS, 1, 1);
        launch_config.dynamicSmemBytes = hdr::SMEM_B;
        cudaError_t e = cudaLaunchKernelEx(
            &launch_config, attn_standalone_kernel<0>, ATTN_HOST_ARGS(m), pdl_flag);
        if (e != cudaSuccess) return (int)e;
      } else {
        attn_standalone_kernel<0><<<QKV_TASKS, THREADS, hdr::SMEM_B, st>>>(ATTN_HOST_ARGS(m), 0);
      }
      break;
    case 1:
      qkv_reduce_kernel<<<QKV_TILES * REDUCE_GROUPS, kMathThreads, 0, st>>>(ATTN_HOST_ARGS(m), 0); break;
    case 2:
      if (int rc = set_smem_once(attn_standalone_kernel<1>, attr[1])) return rc;
      if (pdl) {
        launch_config.gridDim = dim3(ATTN_TASKS, 1, 1);
        launch_config.blockDim = dim3(THREADS, 1, 1);
        launch_config.dynamicSmemBytes = hdr::SMEM_B;
        cudaError_t e = cudaLaunchKernelEx(
            &launch_config, attn_standalone_kernel<1>, ATTN_HOST_ARGS(m), pdl_flag);
        if (e != cudaSuccess) return (int)e;
      } else {
        attn_standalone_kernel<1><<<ATTN_TASKS, THREADS, hdr::SMEM_B, st>>>(ATTN_HOST_ARGS(m), 0);
      }
      break;
    case 3:
      combine_rows_kernel<false><<<H * (M_PAD / SA_COMBINE_ROWS), kMathThreads, 0, st>>>(ATTN_HOST_ARGS(m), 0); break;
    case 6:
      if (pdl) {
        launch_config.gridDim = dim3(H * (M_PAD / SA_COMBINE_ROWS), 1, 1);
        launch_config.blockDim = dim3(kMathThreads, 1, 1);
        launch_config.dynamicSmemBytes = 0;
        cudaError_t e = cudaLaunchKernelEx(
            &launch_config, combine_rows_kernel<true>, ATTN_HOST_ARGS(m), pdl_flag);
        if (e != cudaSuccess) return (int)e;
      } else {
        combine_rows_kernel<true><<<H * (M_PAD / SA_COMBINE_ROWS), kMathThreads, 0, st>>>(ATTN_HOST_ARGS(m), 0);
      }
      break;
    case 4:
      if (int rc = set_smem_once(attn_standalone_kernel<2>, attr[2])) return rc;
      attn_standalone_kernel<2><<<OUT_TASKS, THREADS, hdr::SMEM_B, st>>>(ATTN_HOST_ARGS(m), 0); break;
    case 5:
      out_reduce_kernel<<<OUT_TILES * REDUCE_GROUPS, kMathThreads, 0, st>>>(ATTN_HOST_ARGS(m), 0); break;
    default: return 1103;
  }
  return (int)cudaGetLastError();
}

extern "C" int attn_taskloop_smem_bytes() { return attn::hdr::SMEM_B; }
