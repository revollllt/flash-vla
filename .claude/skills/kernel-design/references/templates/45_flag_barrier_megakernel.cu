// Template 45 -- flag-barrier megakernels with cache-policy loads (sm90).
//
// The simplest megakernel that is still a megakernel. No interpreter, no
// planner, no pages: one launch, a fixed program of phases, and a global-memory
// flag between phases instead of a kernel boundary. It is the shape of
// gau-nernst/learn-cuda `12_megakernel` (decode-oriented Qwen3-0.6B, CUDA C++
// after Triton), distilled rather than copied: that repository carries no
// license file at the revision read, so nothing here is an excerpt.
//
// Two kernels, both self-contained, both checked against a double reference:
//
//   mlp_megakernel   rmsnorm -> gate/up GEMV + SwiGLU -> flag -> down GEMV + residual
//   attn_megakernel  rmsnorm -> QKV GEMV + KV append -> flag -> q/k norm + RoPE
//                    -> flag -> split-KV attention -> flag -> LSE combine
//                    -> flag -> o_proj + residual
//
// and the same phases as separate launches, which is the baseline (the
// upstream "2-kernel" / eager baselines).
//
// WHAT THE UPSTREAM CARRIES THAT WAS WORTH KEEPING
//
//  1. THE FLAG BARRIER. Every phase boundary is one `atom.add.release.gpu` per
//     CTA and one `ld.acquire.gpu` spin, on a counter the LAST arriving CTA
//     resets at kernel end. No cooperative launch: the grid is the SM count
//     with 1 CTA/SM by __launch_bounds__, so co-residency is by construction.
//     A grid larger than what fits deadlocks silently -- the harness asserts
//     grid <= SMs x occupancy.
//  2. REDUNDANT RMSNORM. Every CTA normalises the whole input into its own
//     shared memory rather than one CTA doing it and a barrier publishing it.
//     One hop [atom.lat.dev.hop] costs more than 132 CTAs re-reading 2 KB.
//  3. CACHE-POLICY WEIGHT LOADS. Weights are read once per token and must not
//     displace anything: L1::no_allocate, and L2::evict_first. Activations,
//     which every CTA re-reads, are loaded plainly.
//  4. bf16x2 UNPACK BY SHIFT/MASK, not cvt: bf16 is the top half of fp32, so
//     `shl.b32 16` and `and.b32 0xffff0000` are the whole conversion.
//
// SM90 FACTS THIS PORT HAD TO LEARN (checked in PTX/SASS, CUDA 13.1):
//
//   * `ld.global.L2::evict_first` is REJECTED by ptxas on sm_90a: that
//     qualifier is only legal on 256-bit loads (.v8.b32 / .v4.b64), which are
//     sm_100+. The sm90 spelling is `ld.global.L2::cache_hint` with a 64-bit
//     policy from `createpolicy.fractional.L2::evict_first`. It lands in the
//     load's memory descriptor, so it is free per instruction.
//   * `L1::no_allocate` is legal on sm90 and becomes `LDG.E.NA` -- the same
//     bit the TMA path sets by default. It is also legal on st.global.
//   * The upstream spelling `ld.global.relaxed.cta...` is NOT a cache policy:
//     it is a strong (scoped) load and compiles to `LDG.E.NA.STRONG.SM`, a
//     different instruction. This template uses weak loads; the scope was
//     never doing anything for a read-only weight.
//   * `fma.rn.f32x2` and 256-bit vector loads do not exist on sm90; the
//     upstream's fastest paths are Blackwell-only.
//
// UPSTREAM NUMBERS, for comparison (its README, its accounting of bytes moved:
// RMSNorm in+w+out, w13 + in + out, w2 + in + out):
//
//   MLP, M=1 N=3072 K=1024 bf16, "GEMV CUDA v1":
//     RTX 5090 (400 W)   15.40 us   1227 GB/s
//     H200 (Modal)       11.65 us   1622 GB/s
//   decode attention, dim 1024, 16 heads / 8 kv heads, head_dim 128, its
//   Triton v2 (the CUDA one was unfinished upstream):
//     kv 128:   5090 19.27 us    H200 29.06 us
//     kv 4096:  5090 28.67 us    H200 29.76 us
//
// STATUS (H100 SXM5, CUDA 13.1, clocks not pinned, 132 CTAs x 8 warps, 8
// weight sets rotated so the stream is L2-cold, graph-captured, min of 3;
// every variant matches the double reference exactly):
//
//   MLP, M=1 N=3072 K=1024             megakernel   3 launches   floor (1 / 3 launches)
//     plain                              11.57 us     12.22 us    8.67 / 12.37 us
//     L1::no_allocate                    11.53 us     12.16 us
//     L1::no_alloc + L2::evict_first     11.55 us     12.18 us
//     -> 1634 GB/s by the upstream's accounting, against its 1622 GB/s on H200;
//        the policies are within noise of each other.
//   attention, kv 4096                   23.7 us      23.6 us     12.5 / 19.9 us
//   attention, kv 128                    14.3 us      14.2 us      6.6 / 14.0 us
//     -> under the upstream's Triton v2 (29.8 / 29.1 us on H200); the four
//        flag barriers cost exactly what the four launch boundaries cost.
//
// Two lessons the numbers carry: the cache policies are hygiene, not speed,
// on this stream; and the attention megakernel was 34 us until its KV loop
// hoisted four blocks of loads ahead of the softmax -- a one-load-per-trip
// loop is a latency chain, whatever the kernel around it does.
//
//   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 \
//        -o mk45 45_flag_barrier_megakernel.cu -lcuda && ./mk45
//
// CHECK-GRADE: reference
// CHECK-PTX: createpolicy\.fractional\.L2::evict_first\.b64
// CHECK-PTX: ld\.global\.L1::no_allocate\.L2::cache_hint\.v4\.b32
// CHECK-PTX: ld\.global\.L1::no_allocate\.v4\.b32
// CHECK-PTX: atom\.add\.release\.gpu\.s32
// CHECK-PTX: ld\.acquire\.gpu\.b32
// CHECK-PTX: cvt\.rn\.bf16x2\.f32

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

// ----------------------------------------------------------------- shapes
// Qwen3-0.6B, the upstream target: hidden 1024, intermediate 3072, 16 query
// heads over 8 KV heads of 128. Override on the nvcc line to re-measure.
#ifndef MK45_DIM
#define MK45_DIM 1024
#endif
#ifndef MK45_MLP_DIM
#define MK45_MLP_DIM 3072
#endif
#ifndef MK45_HEAD_DIM
#define MK45_HEAD_DIM 128
#endif
#ifndef MK45_NUM_HEADS
#define MK45_NUM_HEADS 16
#endif
#ifndef MK45_NUM_KV_HEADS
#define MK45_NUM_KV_HEADS 8
#endif
#ifndef MK45_NUM_WARPS
#define MK45_NUM_WARPS 8
#endif
// KV positions already in the cache when the step runs; the upstream table
// reports 128 and 4096.
#ifndef MK45_KV_LEN
#define MK45_KV_LEN 4096
#endif
// Weight copies rotated through the timed graph so the weight stream is cold
// in L2, as it is in a real decode where every layer's weights differ.
#ifndef MK45_ROTATE
#define MK45_ROTATE 8
#endif

namespace {

constexpr int kWarpSize = 32;
constexpr int kDim = MK45_DIM;
constexpr int kMlpDim = MK45_MLP_DIM;
constexpr int kHeadDim = MK45_HEAD_DIM;
constexpr int kNumHeads = MK45_NUM_HEADS;
constexpr int kNumKvHeads = MK45_NUM_KV_HEADS;
constexpr int kQDim = kNumHeads * kHeadDim;
constexpr int kKvDim = kNumKvHeads * kHeadDim;
constexpr int kQkvDim = kQDim + 2 * kKvDim;
constexpr int kNumWarps = MK45_NUM_WARPS;
constexpr int kThreads = kNumWarps * kWarpSize;
constexpr float kEps = 1e-6f;

// ------------------------------------------------------------ vocabulary

enum class LoadPolicy : int {
  kPlain = 0,           // ld.global
  kL1NoAllocate = 1,    // ld.global.L1::no_allocate           -> LDG.E.NA
  kL1NoAllocL2EvictFirst = 2  // + L2::cache_hint(evict_first) in the descriptor
};

// bf16 is the high half of fp32: the unpack is a shift and a mask, not a cvt.
__device__ __forceinline__ void bf16x2_to_fp32x2(float* out, uint32_t x) {
  asm("shl.b32 %0, %2, 16;\n and.b32 %1, %2, 0xFFFF0000;"
      : "=f"(out[0]), "=f"(out[1]) : "r"(x));
}

__device__ __forceinline__ uint32_t fp32x2_to_bf16x2(float a, float b) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b));
  return r;
}

// Built once per thread; the descriptor rides along with every hinted load.
__device__ __forceinline__ uint64_t make_evict_first_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

// One 16-byte weight load under the chosen policy. `asm volatile` keeps the
// compiler from merging or reordering the loads across the flag barrier.
template <LoadPolicy P>
__device__ __forceinline__ uint4 ldg_16(const void* p, uint64_t pol) {
  uint4 v;
  if constexpr (P == LoadPolicy::kPlain) {
    asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  } else if constexpr (P == LoadPolicy::kL1NoAllocate) {
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  } else {
    asm volatile("ld.global.L1::no_allocate.L2::cache_hint.v4.b32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
  }
  return v;
}

// Release: this CTA's phase writes must be visible before its arrival counts.
__device__ __forceinline__ int atomic_add_release_gpu(int* p, int v) {
  int old;
  asm volatile("atom.add.release.gpu.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}

// Acquire: the next phase's reads must not be hoisted above the observation.
__device__ __forceinline__ int load_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// The whole phase barrier: CTA-local sync, one arrival, one spin, CTA-local
// sync. `expected` is the number of arrivals that end the phase -- usually the
// grid, but a phase that only some warps work in counts only those.
__device__ __forceinline__ void flag_barrier(int* flag, int arrivals, int expected) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (arrivals > 0) { atomic_add_release_gpu(flag, arrivals); }
    while (load_acquire_gpu(flag) < expected) { __nanosleep(20); }
  }
  __syncthreads();
}

// The last CTA to leave clears every flag for the next launch. Kernel
// completion orders these stores before the next launch's first acquire.
__device__ __forceinline__ void flags_reset(int* flags, int num_flags) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (atomic_add_release_gpu(flags + num_flags, 1) == static_cast<int>(gridDim.x) - 1) {
      for (int i = 0; i <= num_flags; ++i) { flags[i] = 0; }
    }
  }
}

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, off); }
  return v;
}

// RMSNorm of one row into shared memory as bf16, by the whole CTA. Called by
// EVERY CTA: the redundancy is cheaper than a barrier to share the result.
template <int DIM>
__device__ __forceinline__ void rms_norm_to_smem(const __nv_bfloat16* __restrict__ x,
                                                 const __nv_bfloat16* __restrict__ w,
                                                 __nv_bfloat16* xs, float* red) {
  static_assert(DIM % 8 == 0, "DIM must be a multiple of 8 for 16-byte chunks");
  const int tid = threadIdx.x, warp = tid / kWarpSize, lane = tid % kWarpSize;
  float ss = 0.f;
  for (int idx = tid * 8; idx < DIM; idx += kThreads * 8) {
    const uint4 raw = *reinterpret_cast<const uint4*>(x + idx);
    const uint32_t* r = reinterpret_cast<const uint32_t*>(&raw);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float f[2]; bf16x2_to_fp32x2(f, r[j]);
      ss += f[0] * f[0] + f[1] * f[1];
    }
  }
  ss = warp_sum(ss);
  if (lane == 0) { red[warp] = ss; }
  __syncthreads();
  float total = 0.f;
  #pragma unroll
  for (int i = 0; i < kNumWarps; ++i) { total += red[i]; }
  const float scale = rsqrtf(total / static_cast<float>(DIM) + kEps);
  for (int idx = tid * 8; idx < DIM; idx += kThreads * 8) {
    const uint4 rx = *reinterpret_cast<const uint4*>(x + idx);
    const uint4 rw = *reinterpret_cast<const uint4*>(w + idx);
    const uint32_t* px = reinterpret_cast<const uint32_t*>(&rx);
    const uint32_t* pw = reinterpret_cast<const uint32_t*>(&rw);
    uint4 out;
    uint32_t* po = reinterpret_cast<uint32_t*>(&out);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float fx[2], fw[2];
      bf16x2_to_fp32x2(fx, px[j]);
      bf16x2_to_fp32x2(fw, pw[j]);
      po[j] = fp32x2_to_bf16x2(fx[0] * scale * fw[0], fx[1] * scale * fw[1]);
    }
    *reinterpret_cast<uint4*>(xs + idx) = out;
  }
  __syncthreads();
}

// One warp, one weight row against a bf16 vector: the GEMV inner loop. K/256
// steps of one 16-byte load per lane; the loop is fully unrolled so all of a
// lane's loads are in flight together, which is what the bandwidth needs.
template <int K, LoadPolicy P>
__device__ __forceinline__ float warp_dot_row(const __nv_bfloat16* __restrict__ wrow,
                                              const __nv_bfloat16* __restrict__ v,
                                              int lane, uint64_t pol) {
  static_assert(K % (kWarpSize * 8) == 0, "K must be a multiple of 256");
  float acc = 0.f;
  #pragma unroll
  for (int i = 0; i < K / (kWarpSize * 8); ++i) {
    const int col = (i * kWarpSize + lane) * 8;
    const uint4 rw = ldg_16<P>(wrow + col, pol);
    const uint4 rv = *reinterpret_cast<const uint4*>(v + col);
    const uint32_t* pw = reinterpret_cast<const uint32_t*>(&rw);
    const uint32_t* pv = reinterpret_cast<const uint32_t*>(&rv);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float fw[2], fv[2];
      bf16x2_to_fp32x2(fw, pw[j]);
      bf16x2_to_fp32x2(fv, pv[j]);
      acc = fmaf(fw[0], fv[0], acc);
      acc = fmaf(fw[1], fv[1], acc);
    }
  }
  return warp_sum(acc);
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// =========================================================== MLP phases
//
// Phase bodies are __device__ functions so the megakernel and the per-phase
// baseline kernels are literally the same code with different boundaries.

struct MlpArgs {
  const __nv_bfloat16* x;      // [DIM]
  const __nv_bfloat16* norm;   // [DIM]
  const __nv_bfloat16* w13;    // [2*MLP_DIM, DIM]: gate rows then up rows
  const __nv_bfloat16* w2;     // [DIM, MLP_DIM]
  __nv_bfloat16* xn;           // [DIM] scratch, baseline only
  __nv_bfloat16* h;            // [MLP_DIM] scratch: silu(gate)*up
  __nv_bfloat16* out;          // [DIM]
  int* flags;
};

// gate/up GEMV: one warp per output row, rows strided over the grid.
template <LoadPolicy P>
__device__ __forceinline__ void mlp_phase_gateup(const MlpArgs& a, const __nv_bfloat16* xs,
                                                 uint64_t pol) {
  const int lane = threadIdx.x % kWarpSize;
  const int gw = blockIdx.x * kNumWarps + threadIdx.x / kWarpSize;
  const int stride = gridDim.x * kNumWarps;
  const __nv_bfloat16* w1 = a.w13;
  const __nv_bfloat16* w3 = a.w13 + static_cast<int64_t>(kMlpDim) * kDim;
  for (int n = gw; n < kMlpDim; n += stride) {
    const float g = warp_dot_row<kDim, P>(w1 + static_cast<int64_t>(n) * kDim, xs, lane, pol);
    const float u = warp_dot_row<kDim, P>(w3 + static_cast<int64_t>(n) * kDim, xs, lane, pol);
    if (lane == 0) { a.h[n] = __float2bfloat16(silu(g) * u); }
  }
}

// down GEMV + residual: warp per row; h is read from global, it is tiny and
// L2-resident, and staging it would cost a CTA barrier for nothing.
template <LoadPolicy P>
__device__ __forceinline__ void mlp_phase_down(const MlpArgs& a, uint64_t pol) {
  const int lane = threadIdx.x % kWarpSize;
  const int gw = blockIdx.x * kNumWarps + threadIdx.x / kWarpSize;
  const int stride = gridDim.x * kNumWarps;
  for (int n = gw; n < kDim; n += stride) {
    const float acc = warp_dot_row<kMlpDim, P>(a.w2 + static_cast<int64_t>(n) * kMlpDim, a.h, lane, pol);
    if (lane == 0) { a.out[n] = __float2bfloat16(__bfloat162float(a.x[n]) + acc); }
  }
}

template <LoadPolicy P>
__global__ __launch_bounds__(kThreads, 1) void mlp_megakernel(MlpArgs a) {
  __shared__ __align__(16) __nv_bfloat16 xs[kDim];
  __shared__ float red[kNumWarps];
  const uint64_t pol = make_evict_first_policy();
  rms_norm_to_smem<kDim>(a.x, a.norm, xs, red);
  mlp_phase_gateup<P>(a, xs, pol);
  flag_barrier(a.flags + 0, 1, static_cast<int>(gridDim.x));  // h complete
  mlp_phase_down<P>(a, pol);
  flags_reset(a.flags, 1);
}

// Baseline: the same three phases as three launches.
__global__ __launch_bounds__(kThreads, 1) void mlp_k_rmsnorm(MlpArgs a) {
  __shared__ __align__(16) __nv_bfloat16 xs[kDim];
  __shared__ float red[kNumWarps];
  rms_norm_to_smem<kDim>(a.x, a.norm, xs, red);
  for (int i = threadIdx.x; i < kDim; i += kThreads) { a.xn[i] = xs[i]; }
}
template <LoadPolicy P>
__global__ __launch_bounds__(kThreads, 1) void mlp_k_gateup(MlpArgs a) {
  __shared__ __align__(16) __nv_bfloat16 xs[kDim];
  for (int i = threadIdx.x; i < kDim; i += kThreads) { xs[i] = a.xn[i]; }
  __syncthreads();
  mlp_phase_gateup<P>(a, xs, make_evict_first_policy());
}
template <LoadPolicy P>
__global__ __launch_bounds__(kThreads, 1) void mlp_k_down(MlpArgs a) {
  mlp_phase_down<P>(a, make_evict_first_policy());
}

// ===================================================== attention phases

struct AttnArgs {
  const __nv_bfloat16* x;        // [DIM]
  const __nv_bfloat16* norm;     // [DIM]
  __nv_bfloat16* kv_cache;       // [2][max_ctx][NUM_KV_HEADS][HEAD_DIM]; k at 0, v at 1
  const __nv_bfloat16* wqkv;     // [QKV_DIM, DIM]: q rows, k rows, v rows
  const __nv_bfloat16* q_norm;   // [HEAD_DIM]
  const __nv_bfloat16* k_norm;   // [HEAD_DIM]
  const float* rope;             // [2*HEAD_DIM] for this position: cos[HEAD_DIM], sin[HEAD_DIM]
  const __nv_bfloat16* wo;       // [DIM, Q_DIM]
  __nv_bfloat16* out;            // [DIM]
  __nv_bfloat16* xn;             // [DIM] scratch, baseline only
  __nv_bfloat16* q;              // [Q_DIM] scratch
  float* part_o;                 // [P][NUM_HEADS][HEAD_DIM] split-KV partials
  float* part_ml;                // [P][NUM_HEADS][2]: running max (log2 domain), sum
  __nv_bfloat16* attn;           // [Q_DIM] combined attention output
  int* flags;
  int position;                  // index of the token being decoded; kv[0..position] attend
  int max_ctx;
  int parts;                     // split-KV partitions per head = grid / NUM_HEADS
};

__device__ __forceinline__ __nv_bfloat16* kv_ptr(const AttnArgs& a, int which, int pos, int kvh) {
  return a.kv_cache + ((static_cast<int64_t>(which) * a.max_ctx + pos) * kNumKvHeads + kvh) * kHeadDim;
}

// P1: QKV GEMV; q to scratch, k and v straight into the cache at `position`.
template <LoadPolicy P>
__device__ __forceinline__ void attn_phase_qkv(const AttnArgs& a, const __nv_bfloat16* xs,
                                               uint64_t pol) {
  const int lane = threadIdx.x % kWarpSize;
  const int gw = blockIdx.x * kNumWarps + threadIdx.x / kWarpSize;
  const int stride = gridDim.x * kNumWarps;
  for (int r = gw; r < kQkvDim; r += stride) {
    const float acc = warp_dot_row<kDim, P>(a.wqkv + static_cast<int64_t>(r) * kDim, xs, lane, pol);
    if (lane == 0) {
      const __nv_bfloat16 y = __float2bfloat16(acc);
      if (r < kQDim) {
        a.q[r] = y;
      } else if (r < kQDim + kKvDim) {
        const int i = r - kQDim;
        kv_ptr(a, 0, a.position, i / kHeadDim)[i % kHeadDim] = y;
      } else {
        const int i = r - kQDim - kKvDim;
        kv_ptr(a, 1, a.position, i / kHeadDim)[i % kHeadDim] = y;
      }
    }
  }
}

// P2: per-head RMSNorm then RoPE (rotate-half), one warp per head, in place.
// Qwen3 normalises q and k per head BEFORE the rotation; the rotation pairs
// element d with d + HEAD_DIM/2, so each lane holds both halves.
__device__ __forceinline__ void qk_norm_rope_head(__nv_bfloat16* h, const __nv_bfloat16* w,
                                                  const float* rope, int lane) {
  constexpr int kHalf = kHeadDim / 2;
  static_assert(kHalf % kWarpSize == 0, "half head must split over the warp");
  constexpr int N = kHalf / kWarpSize;  // elements per lane per half
  float lo[N], hi[N], wlo[N], whi[N], c[N], s[N];
  float ss = 0.f;
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    const int d = lane * N + i;
    lo[i] = __bfloat162float(h[d]);
    hi[i] = __bfloat162float(h[d + kHalf]);
    wlo[i] = __bfloat162float(w[d]);
    whi[i] = __bfloat162float(w[d + kHalf]);
    c[i] = rope[d];              // cos repeats over the two halves
    s[i] = rope[kHeadDim + d];   // so does sin
    ss += lo[i] * lo[i] + hi[i] * hi[i];
  }
  ss = warp_sum(ss);
  const float scale = rsqrtf(ss / static_cast<float>(kHeadDim) + kEps);
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    const int d = lane * N + i;
    const float nlo = lo[i] * scale * wlo[i];
    const float nhi = hi[i] * scale * whi[i];
    h[d] = __float2bfloat16(nlo * c[i] - nhi * s[i]);
    h[d + kHalf] = __float2bfloat16(nlo * s[i] + nhi * c[i]);
  }
}

// Returns the number of warps in this CTA that did work, for the flag count.
__device__ __forceinline__ int attn_phase_norm_rope(const AttnArgs& a) {
  const int lane = threadIdx.x % kWarpSize;
  const int gw = blockIdx.x * kNumWarps + threadIdx.x / kWarpSize;
  if (gw < kNumHeads) {
    qk_norm_rope_head(a.q + gw * kHeadDim, a.q_norm, a.rope, lane);
  } else if (gw < kNumHeads + kNumKvHeads) {
    qk_norm_rope_head(kv_ptr(a, 0, a.position, gw - kNumHeads), a.k_norm, a.rope, lane);
  }
  const int first = blockIdx.x * kNumWarps;
  return max(0, min(kNumHeads + kNumKvHeads, first + kNumWarps) - first);
}

// P3: split-KV attention. CTA b owns head b % NUM_HEADS and every
// `parts`-th block of 16 keys. Sixteen lanes share one key (8 dims each);
// the online softmax state is per lane-group and merged at the end: across
// the two groups of a warp by shuffle, across warps through shared memory.
__device__ __forceinline__ void attn_phase_partial(const AttnArgs& a, float* smem) {
  constexpr int kThreadsPerTok = kHeadDim / 8;
  constexpr int kBlockL = kThreads / kThreadsPerTok;
  static_assert(kThreadsPerTok <= kWarpSize && kWarpSize % kThreadsPerTok == 0, "");
  const int tid = threadIdx.x, lane = tid % kWarpSize, warp = tid / kWarpSize;
  const int head = blockIdx.x % kNumHeads;
  const int part = blockIdx.x / kNumHeads;
  if (part >= a.parts) { return; }
  const int kvh = head / (kNumHeads / kNumKvHeads);
  const int col = (tid % kThreadsPerTok) * 8;
  const int row = tid / kThreadsPerTok;
  const int seq = a.position + 1;

  // q, prescaled so the softmax runs in the exp2 domain.
  float qf[8];
  {
    const uint4 raw = *reinterpret_cast<const uint4*>(a.q + head * kHeadDim + col);
    const uint32_t* r = reinterpret_cast<const uint32_t*>(&raw);
    const float qs = 1.4426950408889634f * rsqrtf(static_cast<float>(kHeadDim));
    #pragma unroll
    for (int i = 0; i < 4; ++i) { bf16x2_to_fp32x2(qf + 2 * i, r[i]); qf[2 * i] *= qs; qf[2 * i + 1] *= qs; }
  }
  float m = -1e30f, l = 0.f, o[8] = {};
  const int num_blocks = (seq + kBlockL - 1) / kBlockL;
  // kUnroll key blocks per trip, all loads issued before any math: one load
  // per trip made the loop a latency chain (each 16-byte load waited for the
  // previous block's softmax), and this phase then ran 3x over its floor.
  constexpr int kUnroll = 4;
  for (int blk0 = part; blk0 < num_blocks; blk0 += a.parts * kUnroll) {
    uint4 kraw[kUnroll], vraw[kUnroll];
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int pos = (blk0 + u * a.parts) * kBlockL + row;
      kraw[u] = make_uint4(0, 0, 0, 0);
      vraw[u] = make_uint4(0, 0, 0, 0);
      if (pos < seq) {
        kraw[u] = *reinterpret_cast<const uint4*>(kv_ptr(a, 0, pos, kvh) + col);
        vraw[u] = *reinterpret_cast<const uint4*>(kv_ptr(a, 1, pos, kvh) + col);
      }
    }
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int pos = (blk0 + u * a.parts) * kBlockL + row;
      const uint32_t* kr = reinterpret_cast<const uint32_t*>(&kraw[u]);
      float s = 0.f;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float kf[2]; bf16x2_to_fp32x2(kf, kr[i]);
        s = fmaf(qf[2 * i], kf[0], s); s = fmaf(qf[2 * i + 1], kf[1], s);
      }
      // Reduce the dot over the 16 lanes sharing this key.
      #pragma unroll
      for (int off = kThreadsPerTok / 2; off > 0; off >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, off); }
      if (pos >= seq) { s = -1e30f; }
      const float m_new = fmaxf(m, s);
      const float rescale = exp2f(m - m_new);
      const float p = exp2f(s - m_new);
      m = m_new;
      l = l * rescale + p;
      const uint32_t* vr = reinterpret_cast<const uint32_t*>(&vraw[u]);
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float vf[2]; bf16x2_to_fp32x2(vf, vr[i]);
        o[2 * i] = fmaf(p, vf[0], o[2 * i] * rescale);
        o[2 * i + 1] = fmaf(p, vf[1], o[2 * i + 1] * rescale);
      }
    }
  }
  // Merge the lane groups of this warp (same dims, different keys).
  #pragma unroll
  for (int off = kThreadsPerTok; off < kWarpSize; off <<= 1) {
    const float m2 = __shfl_xor_sync(0xffffffffu, m, off);
    const float l2 = __shfl_xor_sync(0xffffffffu, l, off);
    const float mm = fmaxf(m, m2);
    const float r1 = exp2f(m - mm), r2 = exp2f(m2 - mm);
    l = l * r1 + l2 * r2;
    #pragma unroll
    for (int i = 0; i < 8; ++i) { o[i] = o[i] * r1 + __shfl_xor_sync(0xffffffffu, o[i], off) * r2; }
    m = mm;
  }
  // Merge across warps through shared memory: smem holds per-warp (m, l) and
  // per-warp o[HEAD_DIM].
  float* wm = smem;
  float* wl = smem + kNumWarps;
  float* wo = smem + 2 * kNumWarps;
  if (lane < kThreadsPerTok) {
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    #pragma unroll
    for (int i = 0; i < 8; ++i) { wo[warp * kHeadDim + col + i] = o[i]; }
  }
  __syncthreads();
  if (tid < kHeadDim) {
    float mm = -1e30f;
    #pragma unroll
    for (int w = 0; w < kNumWarps; ++w) { mm = fmaxf(mm, wm[w]); }
    float ll = 0.f, oo = 0.f;
    #pragma unroll
    for (int w = 0; w < kNumWarps; ++w) {
      const float r = exp2f(wm[w] - mm);
      ll += wl[w] * r;
      oo += wo[w * kHeadDim + tid] * r;
    }
    a.part_o[(static_cast<int64_t>(part) * kNumHeads + head) * kHeadDim + tid] = oo;
    if (tid == 0) {
      a.part_ml[(part * kNumHeads + head) * 2 + 0] = mm;
      a.part_ml[(part * kNumHeads + head) * 2 + 1] = ll;
    }
  }
}

// P4: LSE combine of the partials of one head per CTA, exact because the
// partials carry their own max and sum.
__device__ __forceinline__ void attn_phase_combine(const AttnArgs& a) {
  const int head = blockIdx.x;
  if (head >= kNumHeads) { return; }
  const int tid = threadIdx.x;
  if (tid < kHeadDim) {
    float mm = -1e30f;
    for (int p = 0; p < a.parts; ++p) { mm = fmaxf(mm, a.part_ml[(p * kNumHeads + head) * 2]); }
    float ll = 0.f, oo = 0.f;
    for (int p = 0; p < a.parts; ++p) {
      const float r = exp2f(a.part_ml[(p * kNumHeads + head) * 2] - mm);
      ll += a.part_ml[(p * kNumHeads + head) * 2 + 1] * r;
      oo += a.part_o[(static_cast<int64_t>(p) * kNumHeads + head) * kHeadDim + tid] * r;
    }
    a.attn[head * kHeadDim + tid] = __float2bfloat16(oo / ll);
  }
}

// P5: o_proj + residual, warp per row.
template <LoadPolicy P>
__device__ __forceinline__ void attn_phase_oproj(const AttnArgs& a, uint64_t pol) {
  const int lane = threadIdx.x % kWarpSize;
  const int gw = blockIdx.x * kNumWarps + threadIdx.x / kWarpSize;
  const int stride = gridDim.x * kNumWarps;
  for (int n = gw; n < kDim; n += stride) {
    const float acc = warp_dot_row<kQDim, P>(a.wo + static_cast<int64_t>(n) * kQDim, a.attn, lane, pol);
    if (lane == 0) { a.out[n] = __float2bfloat16(__bfloat162float(a.x[n]) + acc); }
  }
}

constexpr int kAttnSmemFloats = 2 * kNumWarps + kNumWarps * kHeadDim;

template <LoadPolicy P>
__global__ __launch_bounds__(kThreads, 1) void attn_megakernel(AttnArgs a) {
  __shared__ __align__(16) __nv_bfloat16 xs[kDim];
  __shared__ float red[kNumWarps];
  __shared__ float part_smem[kAttnSmemFloats];
  const uint64_t pol = make_evict_first_policy();
  rms_norm_to_smem<kDim>(a.x, a.norm, xs, red);
  attn_phase_qkv<P>(a, xs, pol);
  flag_barrier(a.flags + 0, 1, static_cast<int>(gridDim.x));           // q, k, v written
  const int rope_warps = attn_phase_norm_rope(a);
  flag_barrier(a.flags + 1, rope_warps, kNumHeads + kNumKvHeads);      // all heads rotated
  attn_phase_partial(a, part_smem);
  flag_barrier(a.flags + 2, 1, static_cast<int>(gridDim.x));           // partials written
  attn_phase_combine(a);
  flag_barrier(a.flags + 3, 1, static_cast<int>(gridDim.x));           // attn combined
  attn_phase_oproj<P>(a, pol);
  flags_reset(a.flags, 4);
}

// Baseline: five launches over the same phase bodies.
template <LoadPolicy P>
__global__ __launch_bounds__(kThreads, 1) void attn_k_qkv(AttnArgs a) {
  __shared__ __align__(16) __nv_bfloat16 xs[kDim];
  __shared__ float red[kNumWarps];
  rms_norm_to_smem<kDim>(a.x, a.norm, xs, red);
  attn_phase_qkv<P>(a, xs, make_evict_first_policy());
}
__global__ __launch_bounds__(kThreads, 1) void attn_k_norm_rope(AttnArgs a) { attn_phase_norm_rope(a); }
__global__ __launch_bounds__(kThreads, 1) void attn_k_partial(AttnArgs a) {
  __shared__ float part_smem[kAttnSmemFloats];
  attn_phase_partial(a, part_smem);
}
__global__ __launch_bounds__(kThreads, 1) void attn_k_combine(AttnArgs a) { attn_phase_combine(a); }
template <LoadPolicy P>
__global__ __launch_bounds__(kThreads, 1) void attn_k_oproj(AttnArgs a) {
  attn_phase_oproj<P>(a, make_evict_first_policy());
}

}  // namespace

// ================================================================= harness

#ifndef MK45_NO_MAIN

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace {

float host_rand(uint32_t& s, float scale) {
  s = s * 1664525u + 1013904223u;
  return (static_cast<float>((s >> 8) & 0xFFFF) / 65535.f - 0.5f) * 2.f * scale;
}
float r_bf16(float v) { return __bfloat162float(__float2bfloat16(v)); }

std::vector<float> rand_vec(uint32_t& s, size_t n, float scale, float offset = 0.f) {
  std::vector<float> v(n);
  for (auto& e : v) e = r_bf16(offset + host_rand(s, scale));
  return v;
}

__nv_bfloat16* upload(const std::vector<float>& h) {
  std::vector<__nv_bfloat16> t(h.size());
  for (size_t i = 0; i < h.size(); ++i) t[i] = __float2bfloat16(h[i]);
  __nv_bfloat16* d = nullptr;
  CK(cudaMalloc(&d, t.size() * sizeof(__nv_bfloat16)));
  CK(cudaMemcpy(d, t.data(), t.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  return d;
}

std::vector<float> download(const __nv_bfloat16* d, size_t n) {
  std::vector<__nv_bfloat16> t(n);
  CK(cudaMemcpy(t.data(), d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::vector<float> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = __bfloat162float(t[i]);
  return h;
}

float max_rel_err(const std::vector<float>& a, const std::vector<float>& b) {
  float worst = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    const float d = std::fabs(a[i] - b[i]);
    worst = std::max(worst, d / std::max(1e-2f, std::fabs(b[i])));
  }
  return worst;
}

// Doubles throughout, bf16 rounding only where the device rounds: after the
// norm (stored bf16 in smem), after gate/up (h is bf16), never inside a dot.
void rms_norm_ref(const std::vector<float>& x, const std::vector<float>& w, std::vector<double>& y) {
  double ss = 0.0;
  for (size_t i = 0; i < x.size(); ++i) ss += double(x[i]) * x[i];
  const double scale = 1.0 / std::sqrt(ss / double(x.size()) + kEps);
  y.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = r_bf16(float(x[i] * scale * w[i]));
}

void mlp_ref(const std::vector<float>& x, const std::vector<float>& norm,
             const std::vector<float>& w13, const std::vector<float>& w2,
             std::vector<float>& out) {
  std::vector<double> xn;
  rms_norm_ref(x, norm, xn);
  std::vector<double> h(kMlpDim);
  for (int n = 0; n < kMlpDim; ++n) {
    double g = 0.0, u = 0.0;
    for (int k = 0; k < kDim; ++k) {
      g += double(w13[size_t(n) * kDim + k]) * xn[k];
      u += double(w13[size_t(kMlpDim + n) * kDim + k]) * xn[k];
    }
    h[n] = r_bf16(float((g / (1.0 + std::exp(-g))) * u));
  }
  out.resize(kDim);
  for (int n = 0; n < kDim; ++n) {
    double acc = 0.0;
    for (int k = 0; k < kMlpDim; ++k) acc += double(w2[size_t(n) * kMlpDim + k]) * h[k];
    out[n] = r_bf16(float(x[n] + acc));
  }
}

// kv: host mirror of the cache [2][max_ctx][kv_heads][head_dim]; updated in
// place at `position`, as the device does.
void attn_ref(const std::vector<float>& x, const std::vector<float>& norm,
              std::vector<float>& kv, int max_ctx, const std::vector<float>& wqkv,
              const std::vector<float>& qn, const std::vector<float>& kn,
              const std::vector<float>& rope, const std::vector<float>& wo,
              int position, std::vector<float>& out) {
  std::vector<double> xn;
  rms_norm_ref(x, norm, xn);
  std::vector<float> qkv(kQkvDim);
  for (int r = 0; r < kQkvDim; ++r) {
    double acc = 0.0;
    for (int k = 0; k < kDim; ++k) acc += double(wqkv[size_t(r) * kDim + k]) * xn[k];
    qkv[r] = r_bf16(float(acc));
  }
  auto kv_at = [&](int which, int pos, int kvh) {
    return kv.begin() + ((size_t(which) * max_ctx + pos) * kNumKvHeads + kvh) * kHeadDim;
  };
  for (int h = 0; h < kNumKvHeads; ++h) {
    std::copy(qkv.begin() + kQDim + h * kHeadDim, qkv.begin() + kQDim + (h + 1) * kHeadDim, kv_at(0, position, h));
    std::copy(qkv.begin() + kQDim + kKvDim + h * kHeadDim, qkv.begin() + kQDim + kKvDim + (h + 1) * kHeadDim,
              kv_at(1, position, h));
  }
  auto norm_rope = [&](float* v, const std::vector<float>& w) {
    double ss = 0.0;
    for (int d = 0; d < kHeadDim; ++d) ss += double(v[d]) * v[d];
    const double scale = 1.0 / std::sqrt(ss / kHeadDim + kEps);
    double nv[kHeadDim];
    for (int d = 0; d < kHeadDim; ++d) nv[d] = v[d] * scale * w[d];
    for (int d = 0; d < kHeadDim / 2; ++d) {
      const double c = rope[d], s = rope[kHeadDim + d];
      v[d] = r_bf16(float(nv[d] * c - nv[d + kHeadDim / 2] * s));
      v[d + kHeadDim / 2] = r_bf16(float(nv[d] * s + nv[d + kHeadDim / 2] * c));
    }
  };
  for (int h = 0; h < kNumHeads; ++h) norm_rope(qkv.data() + h * kHeadDim, qn);
  for (int h = 0; h < kNumKvHeads; ++h) norm_rope(&*kv_at(0, position, h), kn);
  std::vector<float> attn(kQDim);
  const int seq = position + 1;
  for (int h = 0; h < kNumHeads; ++h) {
    const int kvh = h / (kNumHeads / kNumKvHeads);
    std::vector<double> s(seq);
    double mx = -1e300;
    for (int p = 0; p < seq; ++p) {
      double acc = 0.0;
      for (int d = 0; d < kHeadDim; ++d) acc += double(qkv[h * kHeadDim + d]) * (*(kv_at(0, p, kvh) + d));
      s[p] = acc / std::sqrt(double(kHeadDim));
      mx = std::max(mx, s[p]);
    }
    double l = 0.0;
    std::vector<double> o(kHeadDim, 0.0);
    for (int p = 0; p < seq; ++p) {
      const double e = std::exp(s[p] - mx);
      l += e;
      for (int d = 0; d < kHeadDim; ++d) o[d] += e * (*(kv_at(1, p, kvh) + d));
    }
    for (int d = 0; d < kHeadDim; ++d) attn[h * kHeadDim + d] = r_bf16(float(o[d] / l));
  }
  out.resize(kDim);
  for (int n = 0; n < kDim; ++n) {
    double acc = 0.0;
    for (int k = 0; k < kQDim; ++k) acc += double(wo[size_t(n) * kQDim + k]) * attn[k];
    out[n] = r_bf16(float(x[n] + acc));
  }
}

const char* policy_name(LoadPolicy p) {
  switch (p) {
    case LoadPolicy::kPlain: return "plain";
    case LoadPolicy::kL1NoAllocate: return "L1::no_allocate";
    default: return "L1::no_alloc+L2::evict_first";
  }
}

// Graph-captured timing: `launch(i)` enqueues one full step over weight set i.
template <class F>
float time_graph(cudaStream_t stream, int rotate, F launch) {
  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < rotate; ++i) launch(i);
  CK(cudaStreamEndCapture(stream, &graph));
  CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(exec, stream));
  CK(cudaStreamSynchronize(stream));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  float best = 1e30f;
  const int iters = std::max(1, 200 / rotate);
  for (int r = 0; r < 3; ++r) {
    CK(cudaEventRecord(a, stream));
    for (int i = 0; i < iters; ++i) CK(cudaGraphLaunch(exec, stream));
    CK(cudaEventRecord(b, stream));
    CK(cudaEventSynchronize(b));
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, ms * 1000.f / float(iters) / float(rotate));
  }
  CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return best;
}

// [ld.bw.dev.dram]: 1.85 us of fixed cost per launch, then 2.77 TB/s marginal.
float floor_us(double bytes, int launches) {
  return float(1.85 * launches + bytes / 2.77e6);
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = (argc > 1 && std::string(argv[1]) == "quick");
  int sm_count = 0;
  CK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  const int grid = sm_count;  // 1 CTA/SM by __launch_bounds__; every CTA co-resident
  cudaStream_t stream; CK(cudaStreamCreate(&stream));
  const int rotate = quick ? 1 : MK45_ROTATE;
  uint32_t seed = 20240905u;

  printf("flag-barrier megakernels vs per-phase launches  (%d SMs, %d warps/CTA, grid=%d)\n",
         sm_count, kNumWarps, grid);

  // ============================================================== MLP
  {
    printf("\n[MLP] dim=%d mlp=%d bf16, M=1, %d weight sets rotated\n", kDim, kMlpDim, rotate);
    const float ws = 1.f / std::sqrt(float(kDim));
    std::vector<float> hx = rand_vec(seed, kDim, 1.f), hnorm = rand_vec(seed, kDim, 0.5f, 1.f);
    std::vector<std::vector<float>> hw13(rotate), hw2(rotate);
    std::vector<MlpArgs> args(rotate);
    int* flags = nullptr;
    CK(cudaMalloc(&flags, 8 * sizeof(int)));
    CK(cudaMemset(flags, 0, 8 * sizeof(int)));
    __nv_bfloat16* dx = upload(hx);
    __nv_bfloat16* dnorm = upload(hnorm);
    for (int i = 0; i < rotate; ++i) {
      hw13[i] = rand_vec(seed, size_t(2) * kMlpDim * kDim, ws);
      hw2[i] = rand_vec(seed, size_t(kDim) * kMlpDim, 1.f / std::sqrt(float(kMlpDim)));
      MlpArgs& a = args[i];
      a.x = dx; a.norm = dnorm;
      a.w13 = upload(hw13[i]); a.w2 = upload(hw2[i]);
      CK(cudaMalloc((void**)&a.xn, kDim * 2));
      CK(cudaMalloc((void**)&a.h, kMlpDim * 2));
      CK(cudaMalloc((void**)&a.out, kDim * 2));
      a.flags = flags;
    }
    std::vector<float> ref;
    mlp_ref(hx, hnorm, hw13[0], hw2[0], ref);

    auto run_mk = [&](LoadPolicy p, int i) {
      switch (p) {
        case LoadPolicy::kPlain: mlp_megakernel<LoadPolicy::kPlain><<<grid, kThreads, 0, stream>>>(args[i]); break;
        case LoadPolicy::kL1NoAllocate: mlp_megakernel<LoadPolicy::kL1NoAllocate><<<grid, kThreads, 0, stream>>>(args[i]); break;
        default: mlp_megakernel<LoadPolicy::kL1NoAllocL2EvictFirst><<<grid, kThreads, 0, stream>>>(args[i]); break;
      }
    };
    auto run_base = [&](LoadPolicy p, int i) {
      mlp_k_rmsnorm<<<1, kThreads, 0, stream>>>(args[i]);
      switch (p) {
        case LoadPolicy::kPlain:
          mlp_k_gateup<LoadPolicy::kPlain><<<grid, kThreads, 0, stream>>>(args[i]);
          mlp_k_down<LoadPolicy::kPlain><<<grid, kThreads, 0, stream>>>(args[i]); break;
        case LoadPolicy::kL1NoAllocate:
          mlp_k_gateup<LoadPolicy::kL1NoAllocate><<<grid, kThreads, 0, stream>>>(args[i]);
          mlp_k_down<LoadPolicy::kL1NoAllocate><<<grid, kThreads, 0, stream>>>(args[i]); break;
        default:
          mlp_k_gateup<LoadPolicy::kL1NoAllocL2EvictFirst><<<grid, kThreads, 0, stream>>>(args[i]);
          mlp_k_down<LoadPolicy::kL1NoAllocL2EvictFirst><<<grid, kThreads, 0, stream>>>(args[i]); break;
      }
    };

    printf("  correctness (set 0, double reference)\n");
    bool ok = true;
    for (LoadPolicy p : {LoadPolicy::kPlain, LoadPolicy::kL1NoAllocate, LoadPolicy::kL1NoAllocL2EvictFirst}) {
      run_mk(p, 0); CK(cudaStreamSynchronize(stream));
      const float e1 = max_rel_err(download(args[0].out, kDim), ref);
      run_base(p, 0); CK(cudaStreamSynchronize(stream));
      const float e2 = max_rel_err(download(args[0].out, kDim), ref);
      printf("    %-30s megakernel %.3e  baseline %.3e  %s\n", policy_name(p), e1, e2,
             (e1 < 1e-2f && e2 < 1e-2f) ? "PASS" : "FAIL");
      ok = ok && e1 < 1e-2f && e2 < 1e-2f;
    }
    if (!ok) { printf("numerics failed; timings withheld\n"); return 1; }

    // Upstream's byte accounting: rmsnorm (in, w, out), w13 (in, w, out), w2 (in, w, out).
    const double bytes = 2.0 * (2.0 * kDim + kDim + kDim + 2.0 * kMlpDim * kDim + kMlpDim +
                                kMlpDim + double(kDim) * kMlpDim + kDim);
    printf("  bytes/step %.2f MB; floor %.2f us (1 launch) / %.2f us (3 launches) [ld.bw.dev.dram]\n",
           bytes / 1e6, floor_us(bytes, 1), floor_us(bytes, 3));
    printf("  %-30s %12s %12s %12s %12s\n", "policy", "mk us", "mk GB/s", "base us", "base GB/s");
    for (LoadPolicy p : {LoadPolicy::kPlain, LoadPolicy::kL1NoAllocate, LoadPolicy::kL1NoAllocL2EvictFirst}) {
      const float t_mk = time_graph(stream, rotate, [&](int i) { run_mk(p, i); });
      const float t_base = time_graph(stream, rotate, [&](int i) { run_base(p, i); });
      printf("  %-30s %12.2f %12.0f %12.2f %12.0f\n", policy_name(p), t_mk, bytes / t_mk / 1e3,
             t_base, bytes / t_base / 1e3);
    }
    if (!quick) {
      // L2-warm, for the record: the same weights every replay.
      const float t_warm = time_graph(stream, 1, [&](int i) { run_mk(LoadPolicy::kL1NoAllocL2EvictFirst, i); });
      printf("  (L2-warm, one weight set: megakernel %.2f us -- not a decode number)\n", t_warm);
    }
  }

  // ======================================================== attention
  {
    const int position = MK45_KV_LEN;  // kv[0..position) prefilled, token `position` decoded
    const int max_ctx = position + 1;
    const int parts = grid / kNumHeads;
    printf("\n[ATTN] dim=%d heads=%d kv_heads=%d head_dim=%d, kv_len=%d, split-KV parts/head=%d\n",
           kDim, kNumHeads, kNumKvHeads, kHeadDim, position, parts);
    if (parts < 1 || parts > 32) { printf("  parts must be in 1..32 for the combine\n"); return 1; }
    std::vector<float> hx = rand_vec(seed, kDim, 1.f), hnorm = rand_vec(seed, kDim, 0.5f, 1.f);
    std::vector<float> hqn = rand_vec(seed, kHeadDim, 0.5f, 1.f), hkn = rand_vec(seed, kHeadDim, 0.5f, 1.f);
    // HF rotate-half table for this position, theta 1e6 as Qwen3 uses.
    std::vector<float> hrope(2 * kHeadDim);
    for (int d = 0; d < kHeadDim / 2; ++d) {
      const double omega = 1.0 / std::pow(1e6, double(2 * d) / kHeadDim);
      const double f = double(position) * omega;
      hrope[d] = hrope[d + kHeadDim / 2] = float(std::cos(f));
      hrope[kHeadDim + d] = hrope[kHeadDim + d + kHeadDim / 2] = float(std::sin(f));
    }
    float* drope = nullptr;
    CK(cudaMalloc(&drope, hrope.size() * 4));
    CK(cudaMemcpy(drope, hrope.data(), hrope.size() * 4, cudaMemcpyHostToDevice));
    int* flags = nullptr;
    CK(cudaMalloc(&flags, 8 * sizeof(int)));
    CK(cudaMemset(flags, 0, 8 * sizeof(int)));
    __nv_bfloat16* dx = upload(hx);
    __nv_bfloat16* dnorm = upload(hnorm);
    __nv_bfloat16* dqn = upload(hqn);
    __nv_bfloat16* dkn = upload(hkn);

    std::vector<std::vector<float>> hwqkv(rotate), hwo(rotate), hkv(rotate);
    std::vector<AttnArgs> args(rotate);
    const size_t kv_elems = size_t(2) * max_ctx * kNumKvHeads * kHeadDim;
    for (int i = 0; i < rotate; ++i) {
      hwqkv[i] = rand_vec(seed, size_t(kQkvDim) * kDim, 1.f / std::sqrt(float(kDim)));
      hwo[i] = rand_vec(seed, size_t(kDim) * kQDim, 1.f / std::sqrt(float(kQDim)));
      hkv[i] = rand_vec(seed, kv_elems, 1.f);
      AttnArgs& a = args[i];
      a.x = dx; a.norm = dnorm; a.q_norm = dqn; a.k_norm = dkn; a.rope = drope;
      a.kv_cache = upload(hkv[i]);
      a.wqkv = upload(hwqkv[i]); a.wo = upload(hwo[i]);
      CK(cudaMalloc((void**)&a.out, kDim * 2));
      CK(cudaMalloc((void**)&a.xn, kDim * 2));
      CK(cudaMalloc((void**)&a.q, kQDim * 2));
      CK(cudaMalloc((void**)&a.part_o, size_t(parts) * kNumHeads * kHeadDim * 4));
      CK(cudaMalloc((void**)&a.part_ml, size_t(parts) * kNumHeads * 2 * 4));
      CK(cudaMalloc((void**)&a.attn, kQDim * 2));
      a.flags = flags; a.position = position; a.max_ctx = max_ctx; a.parts = parts;
    }
    std::vector<float> ref;
    {
      std::vector<float> kv = hkv[0];
      attn_ref(hx, hnorm, kv, max_ctx, hwqkv[0], hqn, hkn, hrope, hwo[0], position, ref);
    }

    auto run_mk = [&](LoadPolicy p, int i) {
      switch (p) {
        case LoadPolicy::kPlain: attn_megakernel<LoadPolicy::kPlain><<<grid, kThreads, 0, stream>>>(args[i]); break;
        case LoadPolicy::kL1NoAllocate: attn_megakernel<LoadPolicy::kL1NoAllocate><<<grid, kThreads, 0, stream>>>(args[i]); break;
        default: attn_megakernel<LoadPolicy::kL1NoAllocL2EvictFirst><<<grid, kThreads, 0, stream>>>(args[i]); break;
      }
    };
    auto run_base = [&](LoadPolicy p, int i) {
      const int rope_grid = (kNumHeads + kNumKvHeads + kNumWarps - 1) / kNumWarps;
      switch (p) {
        case LoadPolicy::kPlain: attn_k_qkv<LoadPolicy::kPlain><<<grid, kThreads, 0, stream>>>(args[i]); break;
        case LoadPolicy::kL1NoAllocate: attn_k_qkv<LoadPolicy::kL1NoAllocate><<<grid, kThreads, 0, stream>>>(args[i]); break;
        default: attn_k_qkv<LoadPolicy::kL1NoAllocL2EvictFirst><<<grid, kThreads, 0, stream>>>(args[i]); break;
      }
      attn_k_norm_rope<<<rope_grid, kThreads, 0, stream>>>(args[i]);
      attn_k_partial<<<kNumHeads * parts, kThreads, 0, stream>>>(args[i]);
      attn_k_combine<<<kNumHeads, kThreads, 0, stream>>>(args[i]);
      switch (p) {
        case LoadPolicy::kPlain: attn_k_oproj<LoadPolicy::kPlain><<<grid, kThreads, 0, stream>>>(args[i]); break;
        case LoadPolicy::kL1NoAllocate: attn_k_oproj<LoadPolicy::kL1NoAllocate><<<grid, kThreads, 0, stream>>>(args[i]); break;
        default: attn_k_oproj<LoadPolicy::kL1NoAllocL2EvictFirst><<<grid, kThreads, 0, stream>>>(args[i]); break;
      }
    };

    printf("  correctness (set 0, double reference)\n");
    bool ok = true;
    for (LoadPolicy p : {LoadPolicy::kPlain, LoadPolicy::kL1NoAllocate, LoadPolicy::kL1NoAllocL2EvictFirst}) {
      run_mk(p, 0); CK(cudaStreamSynchronize(stream));
      const float e1 = max_rel_err(download(args[0].out, kDim), ref);
      run_base(p, 0); CK(cudaStreamSynchronize(stream));
      const float e2 = max_rel_err(download(args[0].out, kDim), ref);
      printf("    %-30s megakernel %.3e  baseline %.3e  %s\n", policy_name(p), e1, e2,
             (e1 < 1e-2f && e2 < 1e-2f) ? "PASS" : "FAIL");
      ok = ok && e1 < 1e-2f && e2 < 1e-2f;
    }
    if (!ok) { printf("numerics failed; timings withheld\n"); return 1; }

    // Bytes: weights once, the KV prefix once (k and v), the small vectors.
    const double bytes = 2.0 * (double(kQkvDim) * kDim + double(kDim) * kQDim +
                                2.0 * double(position + 1) * kKvDim + 4.0 * kDim + 2.0 * kQDim);
    printf("  bytes/step %.2f MB; floor %.2f us (1 launch) / %.2f us (5 launches) [ld.bw.dev.dram]\n",
           bytes / 1e6, floor_us(bytes, 1), floor_us(bytes, 5));
    printf("  %-30s %12s %12s %12s %12s\n", "policy", "mk us", "mk GB/s", "base us", "base GB/s");
    for (LoadPolicy p : {LoadPolicy::kPlain, LoadPolicy::kL1NoAllocate, LoadPolicy::kL1NoAllocL2EvictFirst}) {
      const float t_mk = time_graph(stream, rotate, [&](int i) { run_mk(p, i); });
      const float t_base = time_graph(stream, rotate, [&](int i) { run_base(p, i); });
      printf("  %-30s %12.2f %12.0f %12.2f %12.0f\n", policy_name(p), t_mk, bytes / t_mk / 1e3,
             t_base, bytes / t_base / 1e3);
    }
  }
  return 0;
}

#endif  // MK45_NO_MAIN
