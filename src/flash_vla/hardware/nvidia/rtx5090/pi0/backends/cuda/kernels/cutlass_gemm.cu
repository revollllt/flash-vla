// The route's GEMMs, on CUTLASS stream-K rather than cuBLAS.
//
// Two reasons, and the second is the one that matters longer.
//
// Speed. `lab/sm120/pi0_gemm_wave_probe.py` showed cuBLAS running these shapes
// one wave short: at M=768 a 128 x 128 tiling is 96 CTAs on a 170-SM part, and
// 768 x 2048 x 2048 takes the same 49.4 us as 1024 x 2048 x 2048 because both
// are one wave. Every dimension Pi0 uses is a power of two, so no tile grid
// lands near 170 -- 96, 192, 384 and nothing between. Stream-K does not tile
// the output at all: it hands each CTA a contiguous slice of the
// (tile x K-chunk) iteration space and reduces the tiles that end up split.
// That is the decomposition the shape needs, and a hand-written attempt at it
// (kept in `tiled_gemm.cu`) reached 92% of cuBLAS on large shapes but lost the
// fixup cost on small ones.
//
// Control. Programmatic dependent launch needs the producer AND the consumer
// to be kernels this repo can put `griddepcontrol` in. A cuBLAS GEMM between
// two hand-written kernels breaks that chain, and in this route a cuBLAS GEMM
// sat between nearly every pair.
//
// CUTLASS 4.7's sm_120 collective builder is narrow-precision only --
// `static_assert(UseF8f6f4, "Non-blockscaled collective builder only supports
// F8F6F4 MMA")` -- so bf16 here goes through the 2.x device API on the Sm80
// collective, which is `mma.sync` plus `cp.async` and compiles for sm_120a.
//
// Planning is split from launching because stream-K carries a barrier
// workspace. `plan` does the host-side setup once, during warmup; `run` is a
// bare launch and is safe inside a CUDA-graph capture. A replay needs no
// re-initialization: the kernel resets its own barriers, which was verified by
// running an initialized operator five times and checking the result each time.
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "pdl.cuh"

namespace {

using Element = cutlass::bfloat16_t;
using Acc = float;
using RowMajor = cutlass::layout::RowMajor;
//: 8 bf16 is a 16-byte access; every leading dimension Pi0 uses is a multiple.
constexpr int kAlign = 8;
using Epilogue = cutlass::epilogue::thread::LinearCombination<Element, kAlign,
                                                              Acc, Acc>;
using StreamK = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

template <class Tile, class Warp, int kStages>
using Gemm = cutlass::gemm::device::GemmUniversal<
    Element, RowMajor, Element, RowMajor, Element, RowMajor, Acc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, Warp,
    cutlass::gemm::GemmShape<16, 8, 16>, Epilogue, StreamK, kStages, kAlign,
    kAlign>;

//: Tile and warp shapes go through the macro as plain integers: a
//: `GemmShape<a, b, c>` in a macro argument is three preprocessor arguments.
#define FLASH_VLA_GEMM_T(tm, tn, tk, wm, wn, wk, st)                           \
  Gemm<cutlass::gemm::GemmShape<tm, tn, tk>,                                   \
       cutlass::gemm::GemmShape<wm, wn, wk>, st>

// The tiles this route needs, one per distinct winner in the sweep over every
// deployed shape (`lab/sm120/pi0_cutlass_sweep.py`). A bigger tile is more
// efficient per SM and a smaller one puts more CTAs on the part; stream-K
// removes the wave quantization but not that trade, so the shape still picks.
#define FLASH_VLA_CUTLASS_CONFIGS(X)                                           \
  X(0, 128, 128, 64, 64, 64, 64, 3)                                            \
  X(1, 128, 64, 64, 64, 32, 64, 4)                                             \
  X(2, 64, 128, 64, 32, 64, 64, 4)                                             \
  X(3, 64, 64, 64, 32, 32, 64, 5)                                              \
  X(4, 32, 64, 64, 32, 32, 64, 6)                                              \
  X(5, 128, 128, 32, 64, 64, 32, 4)                                            \
  X(6, 32, 128, 32, 32, 64, 32, 5)                                             \
  X(7, 16, 128, 64, 16, 64, 64, 4)                                             \
  X(8, 64, 64, 32, 32, 32, 32, 6)                                              \
  X(9, 32, 64, 32, 32, 32, 32, 8)                                              \
  X(10, 64, 128, 32, 32, 64, 32, 5)                                            \
  X(11, 32, 128, 32, 32, 32, 32, 4)

//: Distinct from a CUDA error: this tile does not apply to this shape, which
//: is an ordinary answer when the host is choosing a config, not a failure.
//: A setup failure returns this plus the CUTLASS status so the reason survives
//: the C boundary -- a tile can be rejected for shape, for shared memory, or
//: for a workspace the caller sized wrong, and those are not the same bug.
constexpr int kCannotImplement = 1000;

//: This library's own PDL switch. The pointwise library has a separate one;
//: they are set together from the host.
bool g_pdl = false;

// CUTLASS's own entry point with the two `griddepcontrol` instructions around
// it. The wait can only go at the very top here, because the mainloop is not
// this repo's to open -- and that is the honest limit of wrapping somebody
// else's kernel. It still pays: measured on a chain shaped like Pi0's, a wait
// at the top is 1.120x under graph replay against 1.164x for a wait placed
// after the producer-independent weight read (`lab/sm120/pdl_unit.cu`). Most
// of the gain is the launch and CTA-scheduling overlap, and that part needs no
// access to the mainloop.
template <class Operator>
__global__ void pdl_kernel_entry(typename Operator::Params params) {
  extern __shared__ int shared_base[];
  auto *storage = reinterpret_cast<typename Operator::SharedStorage *>(shared_base);
  // NOT `cutlass::arch::wait_on_dependent_grids()`. That one is behind
  // CUTLASS_GDC_ENABLED, which needs CUTLASS_ENABLE_GDC_FOR_SM100 defined, and
  // without it BOTH the wait and the trigger compile to nothing -- so the
  // kernel is launched early by the attribute and never waits. That is a race,
  // not a slow kernel, and it showed up as `replay_identical: false` with the
  // model's cosine down at 0.99973. These two are unconditional.
  flash_vla::rtx5090::pdl_wait();
  Operator::invoke(params, *storage);
  flash_vla::rtx5090::pdl_trigger();
}

struct Plan {
  virtual ~Plan() = default;
  virtual cutlass::Status run(cudaStream_t stream) = 0;
};

//: Reaches `params_`, which `GemmUniversalBase` keeps protected, so the launch
//: can be made with the PDL attribute and this file's entry point instead of
//: CUTLASS's.
template <class G>
struct PdlGemm : G {
  using GemmKernel = typename G::GemmKernel;

  cutlass::Status run_pdl(cudaStream_t stream) {
    constexpr int kSmem = int(sizeof(typename GemmKernel::SharedStorage));
    if (kSmem >= (48 << 10)) {
      static bool opted = false;
      if (!opted) {
        if (cudaFuncSetAttribute(pdl_kernel_entry<GemmKernel>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kSmem) != cudaSuccess)
          return cutlass::Status::kErrorInternal;
        opted = true;
      }
    }
    const dim3 grid = this->params_.get_grid_dims();
    const dim3 block(GemmKernel::kThreadCount, 1, 1);
    if (pdl_launch(true, pdl_kernel_entry<GemmKernel>, grid, block, kSmem,
                   stream, this->params_) != cudaSuccess)
      return cutlass::Status::kErrorInternal;
    return cutlass::Status::kSuccess;
  }
};

template <class G>
struct PlanImpl : Plan {
  PdlGemm<G> gemm;
  cutlass::Status run(cudaStream_t stream) override {
    return g_pdl ? gemm.run_pdl(stream) : gemm.run(stream);
  }
};

//: A tile whose shared storage does not fit is rejected here rather than
//: launched. `can_implement` does not check it, and the launch failure it
//: causes is unspecified and kills the context, which during a config sweep
//: takes down every measurement after it rather than one row.
template <class G>
bool fits_shared_memory() {
  static int cap = 0;
  if (cap == 0 &&
      cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0)
          != cudaSuccess)
    return false;
  return int(sizeof(typename G::GemmKernel::SharedStorage)) <= cap;
}

//: d = alpha * a @ b + beta * c. `c` with ldc == 0 broadcasts one row, which
//: is how a bias rides in the epilogue; `d` may alias `c`, which is how a
//: residual does.
template <class G>
typename G::Arguments make_args(int m, int k, int n, int ldc, float beta,
                                const void *a, const void *b, const void *c,
                                void *d) {
  return typename G::Arguments(
      cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, /*batch=*/1,
      {1.0f, beta}, a, b, c, d,
      int64_t(m) * k, int64_t(k) * n, int64_t(m) * n, int64_t(m) * n,
      int64_t(k), int64_t(n), int64_t(ldc), int64_t(n));
}

}  // namespace

extern "C" {

int cutlass_gemm_config_count() {
#define FLASH_VLA_COUNT(id, tm, tn, tk, wm, wn, wk, st) +1
  return 0 FLASH_VLA_CUTLASS_CONFIGS(FLASH_VLA_COUNT);
#undef FLASH_VLA_COUNT
}

//: Bytes of stream-K barrier and partial workspace this shape needs.
int cutlass_gemm_workspace(int config, int m, int k, int n, long long *bytes) {
  switch (config) {
#define FLASH_VLA_WS(id, tm, tn, tk, wm, wn, wk, st)                           \
  case id: {                                                                   \
    using G = FLASH_VLA_GEMM_T(tm, tn, tk, wm, wn, wk, st);                     \
    *bytes = (long long)G::get_workspace_size(                                 \
        make_args<G>(m, k, n, n, 1.0f, nullptr, nullptr, nullptr, nullptr));   \
    return 0;                                                                  \
  }
    FLASH_VLA_CUTLASS_CONFIGS(FLASH_VLA_WS)
#undef FLASH_VLA_WS
    default:
      return (int)cudaErrorInvalidValue;
  }
}

// Host-side setup, called during warmup. Allocates nothing on the device: the
// caller owns `ws`, so the buffer is stable for the graph that will replay it.
int cutlass_gemm_plan(int config, int m, int k, int n, int ldc, float beta,
                      const void *a, const void *b, const void *c, void *d,
                      void *ws, void **handle) {
  switch (config) {
#define FLASH_VLA_PLAN(id, tm, tn, tk, wm, wn, wk, st)                         \
  case id: {                                                                   \
    using G = FLASH_VLA_GEMM_T(tm, tn, tk, wm, wn, wk, st);                     \
    if (!fits_shared_memory<G>()) return kCannotImplement;                     \
    auto args = make_args<G>(m, k, n, ldc, beta, a, b, c, d);                  \
    auto *p = new PlanImpl<G>();                                               \
    if (p->gemm.can_implement(args) != cutlass::Status::kSuccess) {            \
      delete p;                                                                \
      return kCannotImplement;                                                 \
    }                                                                          \
    cutlass::Status setup = p->gemm.initialize(args, ws);                      \
    if (setup != cutlass::Status::kSuccess) {                                   \
      delete p;                                                                \
      return kCannotImplement + (int)setup;                                     \
    }                                                                          \
    *handle = p;                                                               \
    return 0;                                                                  \
  }
    FLASH_VLA_CUTLASS_CONFIGS(FLASH_VLA_PLAN)
#undef FLASH_VLA_PLAN
    default:
      return (int)cudaErrorInvalidValue;
  }
}

int cutlass_gemm_run(void *handle, void *stream) {
  if (handle == nullptr) return (int)cudaErrorInvalidValue;
  cutlass::Status s = ((Plan *)handle)->run((cudaStream_t)stream);
  if (s != cutlass::Status::kSuccess) return (int)cudaErrorLaunchFailure;
  return (int)cudaGetLastError();
}

int cutlass_gemm_set_pdl(int on) {
  g_pdl = on != 0;
  return 0;
}

int cutlass_gemm_destroy(void *handle) {
  delete (Plan *)handle;
  return 0;
}

}  // extern "C"
