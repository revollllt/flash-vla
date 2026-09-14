// BF16 backbone GEMMs with a linear FP32 epilogue, alpha=1 and beta=0 or 1.
// Config 0 of lab/pi05/cutlass_gemm_screen.py: one Sm80 Stream-K collective
// compiled for sm_120a. Gate/up output rounds to BF16 before the separate GELU.
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
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
