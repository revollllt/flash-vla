// gmem_atomic.cu -- isolate global-atomic throughput and counter latency.
//
// Two kernels, because they answer two different questions and mixing them
// would confound both:
//
//   rate_kernel      how many atomic ops per second, as a function of the
//                    instruction (red vs atom), the width, the scope, and how
//                    many threads share an address.  No memory traffic other
//                    than the atomics themselves.
//   pingpong_kernel  how long from one CTA's release-increment to another
//                    CTA's acquire-observe.  Measured as a ping-pong so the
//                    number never crosses two SMs' unsynchronised clocks.
//
// Every axis is a runtime argument except the instruction, which has to be a
// template parameter -- inline PTX cannot take an opcode from a variable, and
// a switch inside the loop would put a branch in the thing being measured.
//
// Build: nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -arch=sm_90a
//        gmem_atomic.cu -lcuda

#include <cuda.h>
#include <cstdint>

namespace atomprobe {

// ~1.2 s at 1.8 GHz. A spin loop that never exits burns the whole Slurm
// allocation and reports nothing; this reports WHERE it hung.
constexpr long long WATCHDOG_CYCLES = 1ll << 31;

enum Op {
  RED_U32 = 0, ATOM_U32, RED_F32, ATOM_F32, RED_F16X2, RED_BF16X2,
  RED_V2F32, RED_V4F32, ATOM_CAS, ATOM_EXCH,
  RED_U32_CTA, RED_U32_SYS, ATOM_U32_CTA, ATOM_U32_SYS,
  OP_COUNT
};

// One op. `acc` exists so a returning atomic cannot be dead-code eliminated,
// and so consecutive ops stay INDEPENDENT: chaining the return value into the
// next address would measure latency, which is a different number and belongs
// to pingpong_kernel.
template <int OP>
__device__ __forceinline__ void one_op(void* p, unsigned v, unsigned& acc) {
  unsigned d;
  if constexpr (OP == RED_U32)
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
  else if constexpr (OP == RED_U32_CTA)
    asm volatile("red.relaxed.cta.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
  else if constexpr (OP == RED_U32_SYS)
    asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
  else if constexpr (OP == ATOM_U32) {
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(d) : "l"(p), "r"(v) : "memory"); acc += d;
  } else if constexpr (OP == ATOM_U32_CTA) {
    asm volatile("atom.relaxed.cta.global.add.u32 %0, [%1], %2;"
                 : "=r"(d) : "l"(p), "r"(v) : "memory"); acc += d;
  } else if constexpr (OP == ATOM_U32_SYS) {
    asm volatile("atom.relaxed.sys.global.add.u32 %0, [%1], %2;"
                 : "=r"(d) : "l"(p), "r"(v) : "memory"); acc += d;
  } else if constexpr (OP == RED_F32) {
    float f = 1.0f;
    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(p), "f"(f) : "memory");
  } else if constexpr (OP == ATOM_F32) {
    float f = 1.0f, o;
    asm volatile("atom.relaxed.gpu.global.add.f32 %0, [%1], %2;"
                 : "=f"(o) : "l"(p), "f"(f) : "memory"); acc += (unsigned)o;
  } else if constexpr (OP == RED_F16X2) {
    unsigned h = 0x3c003c00u;   // {1.0h, 1.0h}
    asm volatile("red.relaxed.gpu.global.add.noftz.f16x2 [%0], %1;"
                 :: "l"(p), "r"(h) : "memory");
  } else if constexpr (OP == RED_BF16X2) {
    unsigned h = 0x3f803f80u;   // {1.0bf, 1.0bf}
    asm volatile("red.relaxed.gpu.global.add.noftz.bf16x2 [%0], %1;"
                 :: "l"(p), "r"(h) : "memory");
  } else if constexpr (OP == RED_V2F32) {
    float a = 1.0f, b = 1.0f;
    asm volatile("red.relaxed.gpu.global.add.v2.f32 [%0], {%1, %2};"
                 :: "l"(p), "f"(a), "f"(b) : "memory");
  } else if constexpr (OP == RED_V4F32) {
    float a = 1.0f;
    asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %1, %1, %1};"
                 :: "l"(p), "f"(a) : "memory");
  } else if constexpr (OP == ATOM_CAS) {
    asm volatile("atom.relaxed.gpu.global.cas.b32 %0, [%1], %2, %3;"
                 : "=r"(d) : "l"(p), "r"(v), "r"(v) : "memory"); acc += d;
  } else if constexpr (OP == ATOM_EXCH) {
    asm volatile("atom.relaxed.gpu.global.exch.b32 %0, [%1], %2;"
                 : "=r"(d) : "l"(p), "r"(v) : "memory"); acc += d;
  }
}

// Contention is set by `n_addr`: every thread picks `slot = gtid & (n_addr-1)`
// and stays there, so the degree of sharing is threads/n_addr exactly. Address
// PLACEMENT is a separate axis, `stride_b` -- 4 B keeps a warp inside one
// 128 B sector, 128 B gives it a line each, 4096 B a page each. The address is
// computed ONCE, outside the loop, so the measurement contains the atomic unit
// and not the address arithmetic.
template <int OP>
__global__ void rate_kernel(uint8_t* __restrict__ base, int n_addr_mask,
                            int stride_b, int trip, unsigned* __restrict__ sink) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t* p = base + (size_t)(gtid & n_addr_mask) * stride_b;
  unsigned acc = 0;
  for (int i = 0; i < trip; ++i) one_op<OP>(p, 1u, acc);
  // Never true, but the compiler cannot prove it, so `acc` stays live.
  if (acc == 0xffffffffu) sink[gtid] = acc;
}

__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// Ping-pong: CTA 0 advances the counter on even values, CTA 1 on odd. One full
// round is two hops, so the host divides by 2*rounds and gets one arrive->
// observe. Timing this from the host avoids comparing clock64() across two
// SMs, which are not synchronised and would make the number unreadable.
//
// `advance_atomic` picks the idiom under test: a release-increment (what a
// task-graph counter does) or a plain release-store (what a flag does).
// `n_pollers` adds CTAs that only OBSERVE, to measure whether observation cost
// grows with the number of observers -- the thing that decides how many
// consumers one counter can feed.
__global__ void pingpong_kernel(unsigned* __restrict__ ctr, int rounds,
                                int advance_atomic, long long* __restrict__ dbg) {
  if (threadIdx.x != 0) { __syncthreads(); return; }
  const int me = blockIdx.x;
  const long long t0 = clock64();

  if (me < 2) {
    for (int r = 0; r < rounds; ++r) {
      const unsigned want = (unsigned)(2 * r + me);
      while (ld_acq(ctr) != want) {
        if (clock64() - t0 > WATCHDOG_CYCLES) {
          if (dbg) { dbg[me * 2] = 1; dbg[me * 2 + 1] = r; }
          __threadfence_system(); __trap();
        }
      }
      if (advance_atomic)
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(ctr) : "memory");
      else
        asm volatile("st.release.gpu.global.u32 [%0], %1;"
                     :: "l"(ctr), "r"(want + 1u) : "memory");
    }
  } else {
    // Observer only: keep reading until the run is over, so it contributes
    // acquire-load traffic to the same line without advancing anything.
    const unsigned done = (unsigned)(2 * rounds);
    while (ld_acq(ctr) < done) {
      if (clock64() - t0 > WATCHDOG_CYCLES) {
        if (dbg) { dbg[me * 2] = 2; }
        __threadfence_system(); __trap();
      }
    }
  }
  __syncthreads();
}

}  // namespace atomprobe

extern "C" {

int atom_probe_rate(int op, void* base, int n_addr, int stride_b, int n_ctas,
                    int n_threads, int trip, void* sink, void* stream) {
  using namespace atomprobe;
  if (n_addr < 1 || (n_addr & (n_addr - 1))) return 1201;   // must be 2^k
  const int mask = n_addr - 1;
  auto s = (cudaStream_t)stream;
#define LAUNCH(K)                                                            \
  case K: rate_kernel<K><<<n_ctas, n_threads, 0, s>>>(                       \
              (uint8_t*)base, mask, stride_b, trip, (unsigned*)sink); break;
  switch (op) {
    LAUNCH(RED_U32) LAUNCH(ATOM_U32) LAUNCH(RED_F32) LAUNCH(ATOM_F32)
    LAUNCH(RED_F16X2) LAUNCH(RED_BF16X2) LAUNCH(RED_V2F32) LAUNCH(RED_V4F32)
    LAUNCH(ATOM_CAS) LAUNCH(ATOM_EXCH) LAUNCH(RED_U32_CTA) LAUNCH(RED_U32_SYS)
    LAUNCH(ATOM_U32_CTA) LAUNCH(ATOM_U32_SYS)
    default: return 1202;
  }
#undef LAUNCH
  return (int)cudaGetLastError();
}

int atom_probe_pingpong(void* ctr, int rounds, int advance_atomic,
                        int n_pollers, void* dbg, void* stream) {
  using namespace atomprobe;
  atomprobe::pingpong_kernel<<<2 + n_pollers, 32, 0, (cudaStream_t)stream>>>(
      (unsigned*)ctr, rounds, advance_atomic, (long long*)dbg);
  return (int)cudaGetLastError();
}

int atom_probe_op_count() { return atomprobe::OP_COUNT; }

}  // extern "C"
