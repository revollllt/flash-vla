// Template 04 -- bulk-store epilogue inside a persistent task loop (sm90).
//
// Closes the pipeline: how an accumulator leaves the register file, and how one
// launch serves many tiles.  The rules:
//
//   1. Publishing crosses proxies.  The epilogue writes shared memory with
//      ordinary stores (generic proxy) and the TMA reads it with the async
//      proxy.  Every writing thread needs fence.proxy.async.shared::cta and
//      then a barrier before one lane issues the store; a release fence placed
//      behind scattered stores instead of before the issue is the classic bug.
//   2. `.read` completion is about shared memory, not global.  wait_group.read
//      says the staging frame may be refilled; the global write may still be in
//      flight.  Only a later fence or kernel boundary orders it for a reader.
//   3. A swizzled staging tile must be written through the matching layout, not
//      linear indices.  This template stores SWIZZLE_NONE so plain indexing is
//      correct; if the map names a swizzle, write the tile through the CuTe
//      atom that the swizzle denotes (template 03).
//   4. Size a persistent grid from occupancy, not from the SM count: a grid
//      below ~3x SMs leaves the tail exposed [sched.ctas.sm.knee], and 32 CTAs
//      cost 1.63x what 128 do for identical bytes [ld.ctas.dev.knee].
//   5. Static striding beats a global work counter here.  One shared counter
//      tops out at 1.36 Gop/s [atom.rate.addr] and a task-graph hop costs
//      ~650 ns [atom.lat.dev.hop]; take a dynamic queue only when tasks are
//      genuinely uneven.
//
// Persistent kernels also break naive timing -- see the measuring-persistent-
// kernels wiki entry before quoting a number from one.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-PTX: cp\.async\.bulk\.tensor\.2d\.global\.shared::cta\.bulk_group
// CHECK-PTX: cp\.async\.bulk\.commit_group
// CHECK-PTX: cp\.async\.bulk\.wait_group\.read
// CHECK-PTX: fence\.proxy\.async\.shared::cta

#include <cuda_bf16.h>

#include "sm90_common.cuh"

namespace {

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kThreads = tmpl::kWarpgroupThreads;
constexpr int kTileElems = kTileM * kTileN;

}  // namespace

// A task is the coordinate the CTA needs to place its output; keep it a plain
// POD so the descriptor array is a single coalesced read.
struct Task {
  int32_t row_block;
  int32_t col_block;
};

__global__ __launch_bounds__(kThreads) void persistent_epilogue_kernel(
    const __grid_constant__ CUtensorMap out_map,
    const Task* __restrict__ tasks, int32_t n_tasks,
    const float* __restrict__ src) {
  // Double-staged so the next tile's epilogue can be built while the previous
  // store drains; one frame would serialize on wait_group.read.
  __shared__ alignas(1024) __nv_bfloat16 stage[2][kTileElems];

  const int32_t tid = static_cast<int32_t>(threadIdx.x);

  // Static stride over the task list: every CTA's next task is gridDim.x away,
  // so no CTA ever queries a shared cursor.
  int32_t frame = 0;
  for (int32_t t = static_cast<int32_t>(blockIdx.x); t < n_tasks;
       t += static_cast<int32_t>(gridDim.x)) {
    const Task task = tasks[t];

    // Stand-in for the mainloop of template 03: the accumulator this epilogue
    // publishes would live in registers at this point.
    for (int32_t i = tid; i < kTileElems; i += kThreads) {
      const float v = src[(t * kTileElems + i) % (n_tasks * kTileElems)];
      stage[frame][i] = __float2bfloat16(v);
    }

    // Order THIS thread's generic-proxy stores before any async-proxy read of
    // them, then make every thread's stores visible before the single issuing
    // lane runs.  Both steps are needed: the fence alone does not synchronize
    // threads, and the barrier alone does not cross proxies.
    tmpl::fence_proxy_async_shared();
    __syncthreads();

    if (tid == 0) {
      tmpl::tma_store_2d(&out_map, stage[frame], task.col_block * kTileN,
                         task.row_block * kTileM);
      tmpl::tma_store_commit();
      // Keep one store in flight; this only claims the OTHER frame is free.
      tmpl::tma_store_wait<1>();
    }
    // Required, not decorative: only tid 0 tracks store completion, so the
    // other threads must not run two iterations ahead and refill a frame whose
    // store it has not yet retired.
    frame ^= 1;
    __syncthreads();
  }

  // Drain before the kernel ends: an outstanding bulk group is not ordered for
  // the next kernel by the launch boundary alone.
  if (tid == 0) { tmpl::tma_store_wait<0>(); }
}

// Grid sizing belongs with the launch, not the kernel: query occupancy and
// multiply, rather than hard-coding a CTA count.
int persistent_grid(const void* kernel, int threads, size_t smem_bytes) {
  int sm_count = 0, blocks_per_sm = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, threads,
                                                smem_bytes);
  return sm_count * (blocks_per_sm > 0 ? blocks_per_sm : 1);
}

CUresult make_output_map(CUtensorMap* map, const __nv_bfloat16* base, uint64_t n,
                         uint64_t m) {
  return tmpl::encode_tile_map_2d(map, base, n, m, kTileN, kTileM,
                                  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                  CU_TENSOR_MAP_SWIZZLE_NONE);
}
