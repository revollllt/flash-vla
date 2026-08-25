// tma_ring.cu -- isolate TMA delivery rate from everything else.
//
// The FFN task-loop confounds six things at once (CTA count, ring depth,
// bytes/TMA, box geometry, wgmma retirement coupling, counter polling).  This
// probe strips all of them but the copy engine: N producer warps per CTA, each
// owning a PRIVATE depth-deep ring of frame_b-byte TMA frames.  No wgmma, no
// counters, no consumer warp.  The only thing measured is how fast the machine
// delivers TMA boxes.
//
// Sweep axes (all runtime, so one .so serves the whole sweep):
//   grid          -- CTAs
//   blockDim/32   -- producer warps per CTA           (Q3)
//   depth         -- frames outstanding per warp      (Q1: in-flight bytes)
//   frame_b       -- bytes per TMA                    (Q1: bytes/transaction)
//   descriptor    -- contiguous vs strided box        (Q2)
//   coord wrap    -- L2-resident vs cold DRAM footprint
//
// Build: nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -arch=sm_90a
//        --expt-relaxed-constexpr -I$CUTLASS_DIR/include tma_ring.cu -lcuda

#include <cuda.h>
#include <cstdint>
#include <cstdio>

#include <cutlass/arch/barrier.h>

namespace tmaprobe {

using FullBar = cutlass::arch::ClusterTransactionBarrier;

constexpr int MAX_DEPTH = 16;
constexpr int MAX_WARPS = 8;
// A persistent-style probe hangs rather than fails, so every wait has a
// deadline (same lesson as the task-loop watchdog).
constexpr long long WATCHDOG_CYCLES = 1ll << 31;  // ~1.2 s at 1.8 GHz

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void tma_2d(const CUtensorMap* map, void* dst,
                                       int32_t c0, int32_t c1, uint64_t* bar) {
  uint32_t d = smem_u32(dst), b = smem_u32(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      :: "r"(d), "l"(map), "r"(c0), "r"(c1), "r"(b) : "memory");
}

__device__ __forceinline__ void wait_wd(uint64_t* bar, uint32_t phase,
                                        long long* dbg, int site) {
  uint32_t addr = smem_u32(bar), done = 0;
  long long t0 = clock64();
  while (!done) {
    asm volatile("{\n .reg .pred p;\n"
                 " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done) : "r"(addr), "r"(phase));
    if (!done && clock64() - t0 > WATCHDOG_CYCLES) {
      if (dbg) {
        dbg[blockIdx.x * 2] = site;
        dbg[blockIdx.x * 2 + 1] = threadIdx.x;
        __threadfence_system();
      }
      __trap();
    }
  }
}

extern __shared__ uint8_t pool[];

// Coordinate walk, per issue:
//   c0 = (idx & mask0) * step0            -- position along the fastest dim
//   c1 = ((idx >> shift0) & mask1) * step1 -- position along the strided dim
// The host guarantees mask0+1 and mask1+1 are powers of two and
// shift0 == log2(mask0+1), so the whole walk is shifts and masks.  A runtime
// IDIV on the issue path would put the integer divider in the measurement
// instead of the copy engine.  A contiguous descriptor passes mask0 = 0.
__global__ __launch_bounds__(MAX_WARPS * 32, 1)
void tma_ring_kernel(const __grid_constant__ CUtensorMap map,
                     int depth, int frame_b, int trip,
                     int mask0, int shift0, int step0,
                     int mask1, int step1,
                     long long* __restrict__ dbg) {
  const int warp  = threadIdx.x >> 5;
  const int nwarp = blockDim.x >> 5;

  uint64_t* bars = reinterpret_cast<uint64_t*>(
      pool + (size_t)nwarp * depth * frame_b);

  if (threadIdx.x == 0) {
    FullBar* b = reinterpret_cast<FullBar*>(bars);
    for (int i = 0; i < nwarp * depth; ++i) b[i].init(1);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  uint8_t* mypool = pool + (size_t)warp * depth * frame_b;
  FullBar* mybar  = reinterpret_cast<FullBar*>(bars) + warp * depth;
  uint32_t ph[MAX_DEPTH] = {};

  // Each (CTA, warp) walks a disjoint stream so the sweep measures delivery,
  // not L2 broadcast of one hot address.
  const int base = blockIdx.x * nwarp + warp;
  const int lane0 = (threadIdx.x & 31) == 0;

  for (int g = 0; g < trip; ++g) {
    const int s = g % depth;
    if (g >= depth) {
      wait_wd(reinterpret_cast<uint64_t*>(&mybar[s]), ph[s], dbg, 1);
      ph[s] ^= 1;
    }
    __syncwarp();
    const int idx = base * trip + g;
    const int32_t c0 = (idx & mask0) * step0;
    const int32_t c1 = ((idx >> shift0) & mask1) * step1;
    if (lane0) {
      mybar[s].arrive_and_expect_tx(frame_b);
      tma_2d(&map, mypool + (size_t)s * frame_b, c0, c1,
             reinterpret_cast<uint64_t*>(&mybar[s]));
    }
  }

  // Drain: an outstanding TMA writing smem the CTA has retired is UB.
  const int tail = trip < depth ? trip : depth;
  for (int j = 0; j < tail; ++j) {
    const int s = (trip - tail + j) % depth;
    wait_wd(reinterpret_cast<uint64_t*>(&mybar[s]), ph[s], dbg, 2);
    ph[s] ^= 1;
  }
  __syncthreads();
}

}  // namespace tmaprobe

extern "C" {

// Encode a 2-D tiled tensor map.  `inner` is the fastest-varying extent in
// ELEMENTS; `strides[0]` is therefore inner*2 for a packed bf16 row.  Returns
// the CUresult so the host can ENUMERATE legality (Q4) rather than assume it.
int tma_probe_encode(void* out_map, const void* p,
                     unsigned long long inner, unsigned long long outer,
                     unsigned int box_inner, unsigned int box_outer,
                     int swizzle) {
  CUtensorMap* m = (CUtensorMap*)out_map;
  // The driver's own typedefs: cuuint64_t is `unsigned long` on LP64, so
  // `unsigned long long` here is a hard type error, not a narrowing warning.
  cuuint64_t dims[2]    = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)(inner * 2)};
  cuuint32_t box[2] = {(cuuint32_t)box_inner, (cuuint32_t)box_outer};
  cuuint32_t es[2]  = {1, 1};
  CUresult rc = cuTensorMapEncodeTiled(
      m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), dims,
      strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
      (CUtensorMapSwizzle)swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int)rc;
}

int tma_probe_launch(const void* map, int n_ctas, int n_warps,
                     int depth, int frame_b, int trip,
                     int mask0, int shift0, int step0,
                     int mask1, int step1,
                     void* dbg, void* stream) {
  using namespace tmaprobe;
  if (depth > MAX_DEPTH || n_warps > MAX_WARPS || depth < 1 || n_warps < 1)
    return 1101;
  size_t smem = (size_t)n_warps * depth * frame_b
              + (size_t)n_warps * depth * sizeof(uint64_t);
  if (smem > 227u * 1024u) return 1102;
  static size_t attr_set = 0;
  if (smem > attr_set) {
    cudaError_t e = cudaFuncSetAttribute(
        tma_ring_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return (int)e;
    attr_set = smem;
  }
  tma_ring_kernel<<<n_ctas, n_warps * 32, smem, (cudaStream_t)stream>>>(
      *(const CUtensorMap*)map, depth, frame_b, trip,
      mask0, shift0, step0, mask1, step1, (long long*)dbg);
  return (int)cudaGetLastError();
}

int tma_probe_map_bytes() { return (int)sizeof(CUtensorMap); }

}  // extern "C"
