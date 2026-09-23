// MXFP8 x MXFP8 -> BF16 GEMMs for the Pi0.5 backbone FFN on sm_120a.
//
// D = A * B^T + beta * C: A [M, K] E4M3 row-major, B [N, K] E4M3 with K
// contiguous, one UE8M0 scale per 32 elements along K in the 128x4 swizzled
// layout quant_ops writes (flash_vla/quantization/formats.py). The block-scaled
// MMA accumulates in FP32 and the epilogue rounds alpha * acc + beta * C to BF16
// once. C aliases D, so beta = 1 adds the product to D in place (the residual)
// and beta = 0 never reads it. Built with CUTLASS_ENABLE_GDC_FOR_SM100, which
// also covers sm_120: the kernels wait on their predecessor before the first
// global read and launch with programmatic dependent launch.
//
// Row buckets: the prefix has 968 rows, 3 x 256 image tokens and 200 prompt
// slots. When the prompt leaves row 896 masked, the last 128-row tile is all
// padding. A GEMM is then planned twice, M = 896 and M = 968, both plans are
// launched, and the one the bf16 prefix mask (mask[896] < 0: short) does not
// select exits at once. The scale layout is 128-row-block major, so the
// 896-row operands are prefixes of the 968-row ones.
#include <cstdint>
#include <memory>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

// A kernel type must not have internal linkage: nvcc's host stub for
// device_kernel<RowBucket<...>> cannot be generated inside an anonymous namespace.
namespace flash_vla_mxfp8 {
using Output = cutlass::bfloat16_t;
constexpr int32_t kShortRows = 896;

// Base, run only when the prefix mask selects its row bucket; without a mask
// it always runs. An exiting kernel still waits on its predecessor, so the
// programmatic dependency chain stays transitive for the kernel after it.
template <class Base>
struct RowBucket : Base {
  using BaseArguments = typename Base::Arguments;
  struct Arguments : BaseArguments {
    Output const* mask = nullptr;   // bf16 [968] additive prefix mask, or null
  };
  struct Params : Base::Params {
    Output const* mask = nullptr;
    bool short_bucket = false;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    Params params;
    static_cast<typename Base::Params&>(params) = Base::to_underlying_arguments(args, workspace);
    params.mask = args.mask;
    params.short_bucket = cute::get<0>(args.problem_shape) == kShortRows;
    return params;
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    if (params.mask != nullptr &&
        (float(params.mask[kShortRows]) < 0.f) != params.short_bucket) {
      cutlass::arch::wait_on_dependent_grids();
      cutlass::arch::launch_dependent_grids();
      return;
    }
    Base::operator()(params, smem);
  }
};

}  // namespace flash_vla_mxfp8

namespace {
using namespace cute;
using flash_vla_mxfp8::Output;
using flash_vla_mxfp8::RowBucket;

using Element = cutlass::float_e4m3_t;
using ScaleFactor = cutlass::float_ue8m0_t;

// One tile configuration: CTA tile TileM x TileN x 128, kernel schedule and
// tile scheduler (persistent data-parallel or Stream-K). TileM is 128: the scale
// factors' TMA box covers 128 rows.
template <int TileM, int TileN, class Schedule, class Scheduler>
struct Config {
  using TileShape = Shape<Int<TileM>, Int<TileN>, _128>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape,
      Shape<_1, _1, _1>, cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      Output, cutlass::layout::RowMajor, 8, Output, cutlass::layout::RowMajor, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      cutlass::mx_float8_t<Element>, cutlass::layout::RowMajor, 16,
      cutlass::mx_float8_t<Element>, cutlass::layout::ColumnMajor, 16, float, TileShape,
      Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      Schedule>::CollectiveOp;
  using Kernel = RowBucket<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue, Scheduler>>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Cooperative = cutlass::gemm::collective::KernelScheduleAuto;
using Pingpong = cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120;
using Persistent = cutlass::gemm::PersistentScheduler;
using StreamK = cutlass::gemm::StreamKScheduler;

// Keep CUTLASS status values distinct from cudaError_t at the C boundary.
constexpr int32_t kCutlassError = 1000;

template <class Gemm>
typename Gemm::Arguments arguments(int32_t m, int32_t n, int32_t k, float beta, const void* a,
                                   const void* a_scale, const void* b, const void* b_scale,
                                   void* d, const void* mask) {
  using Kernel = typename Gemm::GemmKernel;
  using ScaleLayout = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto problem = make_shape(m, n, k, 1);
  auto output_stride = cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, {m, n, 1});
  using BaseArguments = typename Kernel::BaseArguments;
  typename Gemm::Arguments args;
  static_cast<BaseArguments&>(args) = BaseArguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {static_cast<Element const*>(a),
       cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, 1}),
       static_cast<Element const*>(b),
       cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1}),
       static_cast<ScaleFactor const*>(a_scale), ScaleLayout::tile_atom_to_shape_SFA(problem),
       static_cast<ScaleFactor const*>(b_scale), ScaleLayout::tile_atom_to_shape_SFB(problem)},
      {{1.f, beta}, static_cast<Output const*>(d), output_stride, static_cast<Output*>(d),
       output_stride}};
  args.mask = static_cast<Output const*>(mask);
  return args;
}

// A planned GEMM: bound pointers and initialized parameters. Stream-K counts
// its peers in the workspace and never clears the counters, so every launch
// clears them first.
struct Plan {
  virtual ~Plan() = default;
  virtual cutlass::Status run(cudaStream_t stream) = 0;
};

template <class Gemm>
struct BoundPlan final : Plan {
  Gemm gemm;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  bool clears_workspace = false;

  cutlass::Status run(cudaStream_t stream) override {
    if (clears_workspace && cudaMemsetAsync(workspace, 0, workspace_bytes, stream) != cudaSuccess)
      return cutlass::Status::kErrorInternal;
    return gemm.run(stream, nullptr, /*launch_with_pdl=*/true);
  }
};

template <class Gemm, int32_t Splits = 1>
int64_t workspace_of(int32_t m, int32_t n, int32_t k) {
  auto args = arguments<Gemm>(m, n, k, 0.f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  if constexpr (Splits > 1) args.scheduler.splits = Splits;
  const cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) return -(kCutlassError + static_cast<int32_t>(status));
  return static_cast<int64_t>(Gemm::get_workspace_size(args));
}

// Splits > 1 makes a Stream-K scheduler decompose K into that many equal parts.
template <class Gemm, bool IsStreamK, int32_t Splits = 1>
int32_t plan_of(int32_t m, int32_t n, int32_t k, float beta, const void* a, const void* a_scale,
                const void* b, const void* b_scale, void* d, const void* mask, void* workspace,
                cudaStream_t stream, void** handle) {
  auto plan = std::make_unique<BoundPlan<Gemm>>();
  auto args = arguments<Gemm>(m, n, k, beta, a, a_scale, b, b_scale, d, mask);
  if constexpr (IsStreamK) args.scheduler.splits = Splits;
  plan->workspace = workspace;
  plan->workspace_bytes = Gemm::get_workspace_size(args);
  plan->clears_workspace = IsStreamK && plan->workspace_bytes > 0;
  cutlass::Status status = Gemm::can_implement(args);
  if (status == cutlass::Status::kSuccess) status = plan->gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) return kCutlassError + static_cast<int32_t>(status);
  *handle = plan.release();
  return 0;
}

// Tile configurations, by the index the Python side passes.
using Config0 = Config<128, 128, Cooperative, Persistent>::Gemm;
using Config1 = Config<128, 64, Cooperative, Persistent>::Gemm;
using Config2 = Config<128, 32, Cooperative, Persistent>::Gemm;
using Config3 = Config<128, 128, Cooperative, StreamK>::Gemm;
using Config4 = Config<128, 64, Cooperative, StreamK>::Gemm;
// Configs 5, 6 and 7 are Config3 with K split in 2, 3 and 4.
using Config8 = Config<128, 128, Pingpong, Persistent>::Gemm;
// 256-wide tiles leave room for fewer than two pipeline stages in 99 KiB.
constexpr int32_t kConfigs = 9;
constexpr int32_t kUnknownConfig = kCutlassError + static_cast<int32_t>(cutlass::Status::kInvalid);
}  // namespace

extern "C" int32_t mxfp8_gemm_configs() { return kConfigs; }

extern "C" int64_t mxfp8_gemm_workspace(int32_t config, int32_t m, int32_t n, int32_t k) {
  switch (config) {
    case 0: return workspace_of<Config0>(m, n, k);
    case 1: return workspace_of<Config1>(m, n, k);
    case 2: return workspace_of<Config2>(m, n, k);
    case 3: return workspace_of<Config3>(m, n, k);
    case 4: return workspace_of<Config4>(m, n, k);
    case 5: return workspace_of<Config3, 2>(m, n, k);
    case 6: return workspace_of<Config3, 3>(m, n, k);
    case 7: return workspace_of<Config3, 4>(m, n, k);
    case 8: return workspace_of<Config8>(m, n, k);
    default: return -kUnknownConfig;
  }
}

// mask: the bf16 prefix mask for a row-bucket plan (M = 896 or 968), else null.
extern "C" int32_t mxfp8_gemm_plan(int32_t config, int32_t m, int32_t n, int32_t k, float beta,
                                   const void* a, const void* a_scale, const void* b,
                                   const void* b_scale, void* d, const void* mask,
                                   void* workspace, cudaStream_t stream, void** handle) {
  switch (config) {
    case 0: return plan_of<Config0, false>(m, n, k, beta, a, a_scale, b, b_scale, d, mask, workspace, stream, handle);
    case 1: return plan_of<Config1, false>(m, n, k, beta, a, a_scale, b, b_scale, d, mask, workspace, stream, handle);
    case 2: return plan_of<Config2, false>(m, n, k, beta, a, a_scale, b, b_scale, d, mask, workspace, stream, handle);
    case 3: return plan_of<Config3, true>(m, n, k, beta, a, a_scale, b, b_scale, d, mask, workspace, stream, handle);
    case 4: return plan_of<Config4, true>(m, n, k, beta, a, a_scale, b, b_scale, d, mask, workspace, stream, handle);
    case 5: return plan_of<Config3, true, 2>(m, n, k, beta, a, a_scale, b, b_scale, d, mask, workspace, stream, handle);
    case 6: return plan_of<Config3, true, 3>(m, n, k, beta, a, a_scale, b, b_scale, d, mask, workspace, stream, handle);
    case 7: return plan_of<Config3, true, 4>(m, n, k, beta, a, a_scale, b, b_scale, d, mask, workspace, stream, handle);
    case 8: return plan_of<Config8, false>(m, n, k, beta, a, a_scale, b, b_scale, d, mask, workspace, stream, handle);
    default: return kUnknownConfig;
  }
}

extern "C" int32_t mxfp8_gemm_run(void* handle, cudaStream_t stream) {
  const cutlass::Status status = static_cast<Plan*>(handle)->run(stream);
  if (status != cutlass::Status::kSuccess) return kCutlassError + static_cast<int32_t>(status);
  return static_cast<int32_t>(cudaPeekAtLastError());
}

extern "C" void mxfp8_gemm_destroy(void* handle) { delete static_cast<Plan*>(handle); }
