#pragma once

#include "tile/sm90/gemm.cuh"

namespace flash_vla::pi05::sm90::ffn {

// The FFN GEMM contract: bf16 operands in shared memory, f32 accumulation,
// one math warpgroup.  The tile library selects the wgmma instruction from
// (M, N, K, majors); the task bodies name their geometry through this alias
// so the N=64 GatedUp and N=32 DownResidual choices stay visible at the call
// site instead of as spelled-out instruction names.
template <int M, int N, int K, flash_vla::sm90::Major MajorA,
          flash_vla::sm90::Major MajorB>
using Gemm = flash_vla::sm90::MmaSelector<
    flash_vla::sm90::BF16, flash_vla::sm90::BF16, M, N, K,
    flash_vla::sm90::Operand::kSmem, flash_vla::sm90::Operand::kSmem,
    MajorA, MajorB>;

}  // namespace flash_vla::pi05::sm90::ffn
