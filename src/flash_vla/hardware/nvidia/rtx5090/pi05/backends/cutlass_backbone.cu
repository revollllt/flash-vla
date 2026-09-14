// BF16 backbone GEMMs with a linear FP32 epilogue, alpha=1 and beta=0 or 1.
// Config 0 of lab/pi05/cutlass_gemm_screen.py: one Sm80 Stream-K collective
// compiled for sm_120a. Gate/up output rounds to BF16 before the separate GELU.
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_streamk_with_broadcast.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"

namespace {
using Element = cutlass::bfloat16_t;
using RowMajor = cutlass::layout::RowMajor;
using Epilogue = cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>;
using Gemm = cutlass::gemm::device::GemmUniversal<
    Element, RowMajor, Element, RowMajor, Element, RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 3, 8, 8>;

Gemm::Arguments arguments(int32_t m, int32_t k, int32_t n, float beta,
                          const void* a, const void* b, void* output) {
  return Gemm::Arguments(
      cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1,
      {1.f, beta}, a, b, output, output,
      int64_t(m) * k, int64_t(k) * n, int64_t(m) * n, int64_t(m) * n,
      int64_t(k), int64_t(n), int64_t(n), int64_t(n));
}

// Keep CUTLASS status values distinct from cudaError_t at the C boundary.
constexpr int32_t kCutlassError = 1000;

// The Stream-K broadcast epilogue receives the fully reduced FP32 accumulator.
// Preserve the separate BF16 GEMM store numerically before gate/residual math.
class RoundedGatedResidual {
 public:
  using ElementOutput = Element;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementT = Element;
  using ElementVector = Element;
  static constexpr int kCount = 8;
  static constexpr bool kIsHeavy = false;
  static constexpr bool kIsSingleSource = true;
  static constexpr bool kStoreZ = true;
  static constexpr bool kStoreT = false;
  using FragmentOutput = cutlass::Array<Element, kCount>;
  using FragmentAccumulator = cutlass::Array<float, kCount>;
  using FragmentCompute = cutlass::Array<float, kCount>;
  struct Params {};

  CUTLASS_HOST_DEVICE explicit RoundedGatedResidual(Params const&) {}
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

  CUTLASS_DEVICE void operator()(
      FragmentOutput& result, FragmentOutput&, FragmentAccumulator const& accum,
      FragmentOutput const& residual, FragmentCompute const& gate) const {
    cutlass::NumericArrayConverter<Element, float, kCount> round_bf16;
    cutlass::NumericArrayConverter<float, Element, kCount> to_float;
    const FragmentCompute projected = to_float(round_bf16(accum));
    const FragmentCompute source = to_float(residual);
    FragmentCompute updated;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i)
      updated[i] = __fadd_rn(__fmul_rn(projected[i], gate[i]), source[i]);
    result = round_bf16(updated);
  }

  // Required by the broadcast concept; the selected route always reads C.
  CUTLASS_DEVICE void operator()(
      FragmentOutput& result, FragmentOutput&, FragmentAccumulator const& accum,
      FragmentCompute const& gate) const {
    cutlass::NumericArrayConverter<Element, float, kCount> round_bf16;
    cutlass::NumericArrayConverter<float, Element, kCount> to_float;
    const FragmentCompute projected = to_float(round_bf16(accum));
    FragmentCompute updated;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) updated[i] = __fmul_rn(projected[i], gate[i]);
    result = round_bf16(updated);
  }
};

// Config 9: one new template type in the existing native library. The unique
// epilogue also keeps its per-kernel CUTLASS initialization state distinct.
using DownDefaults = cutlass::gemm::device::GemmUniversalStreamkWithBroadcast<
    Element, RowMajor, Element, RowMajor, Element, RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>, RoundedGatedResidual,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 8, 8, 8>;

using DownEpilogueBase = DownDefaults::GemmKernel::Epilogue;

// epilogue_with_broadcast.h::reduce advances C by fragment index but omits Z.
// M50 N1024 K4096 with the 32x64 tile then leaves later row fragments unwritten.
// Mirror epilogue.h::reduce so each reduction block stores its own fragment.
class DownEpilogue : public DownEpilogueBase {
 public:
  using DownEpilogueBase::DownEpilogueBase;

  CUTLASS_DEVICE void reduce(
      int peer_begin, int peer_end, int fragment, void* workspace,
      OutputOp const& op, ElementVector const* gate,
      OutputTileIterator destination, OutputTileIterator source,
      TensorTileIterator tensor, cutlass::MatrixCoord const& problem,
      cutlass::MatrixCoord const& offset) {
    destination += fragment;
    DownEpilogueBase::reduce(peer_begin, peer_end, fragment, workspace, op,
                            gate, destination, source, tensor, problem, offset);
  }
};

using DownGemm = cutlass::gemm::device::GemmUniversalBase<
    cutlass::gemm::kernel::GemmStreamkWithFusedEpilogue<
        DownDefaults::GemmKernel::Mma, DownEpilogue,
        cutlass::gemm::threadblock::ThreadblockSwizzleStreamK>>;

DownGemm::Arguments down_arguments(const void* a, const void* b, const void* gate,
                                    void* output) {
  DownGemm::Arguments args;
  args.problem_size = {50, 1024, 4096};
  args.ptr_A = a;
  args.ptr_B = b;
  args.ptr_C = output;
  args.ptr_D = output;
  args.ptr_Vector = const_cast<void*>(gate);
  args.lda = 4096;
  args.ldb = args.ldc = args.ldd = 1024;
  args.ldr = 0;  // One gate vector broadcast over every output row.
  return args;
}
}  // namespace

extern "C" int64_t backbone_gemm_workspace(int32_t m, int32_t k, int32_t n) {
  return static_cast<int64_t>(
      Gemm::get_workspace_size(arguments(m, k, n, 1.f, nullptr, nullptr, nullptr)));
}

extern "C" int32_t backbone_gemm_plan(
    int32_t m, int32_t k, int32_t n, float beta, const void* a, const void* b,
    void* output, void* workspace, void* stream, void** handle) {
  auto args = arguments(m, k, n, beta, a, b, output);
  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess)
    return kCutlassError + static_cast<int32_t>(status);
  auto* plan = new Gemm;
  status = plan->initialize(args, workspace, static_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess) {
    delete plan;
    return kCutlassError + static_cast<int32_t>(status);
  }
  *handle = plan;
  return 0;
}

extern "C" int32_t backbone_gemm_run(void* handle, void* stream) {
  const cutlass::Status status =
      static_cast<Gemm*>(handle)->run(static_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess)
    return kCutlassError + static_cast<int32_t>(status);
  return static_cast<int32_t>(cudaGetLastError());
}

extern "C" void backbone_gemm_destroy(void* handle) {
  delete static_cast<Gemm*>(handle);
}

extern "C" int64_t expert_down_workspace() {
  return static_cast<int64_t>(
      DownGemm::get_workspace_size(down_arguments(nullptr, nullptr, nullptr, nullptr)));
}

extern "C" int32_t expert_down_plan(
    const void* a, const void* b, const void* gate, void* output,
    void* workspace, void* stream, void** handle) {
  const auto args = down_arguments(a, b, gate, output);
  cutlass::Status status = DownGemm::can_implement(args);
  if (status != cutlass::Status::kSuccess)
    return kCutlassError + static_cast<int32_t>(status);
  auto* plan = new DownGemm;
  status = plan->initialize(args, workspace, static_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess) {
    delete plan;
    return kCutlassError + static_cast<int32_t>(status);
  }
  *handle = plan;
  return 0;
}

extern "C" int32_t expert_down_run(void* handle, void* stream) {
  const cutlass::Status status =
      static_cast<DownGemm*>(handle)->run(static_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess)
    return kCutlassError + static_cast<int32_t>(status);
  return static_cast<int32_t>(cudaGetLastError());
}

extern "C" void expert_down_destroy(void* handle) {
  delete static_cast<DownGemm*>(handle);
}
