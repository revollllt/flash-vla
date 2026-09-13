// A CUTLASS epilogue that finishes a gated feed-forward in the GEMM.
//
// The gated form is `out = gelu(x @ gate_w) * (x @ up_w)`, and the route spells
// it as two GEMMs and a third kernel that reads both expansions and writes one.
// At the backbone's shape that third pass moves 75.5 MB -- gate and up in, out
// out -- for one elementwise function.
//
// The two GEMMs write the same tile coordinates, so the second one's epilogue
// can read the first one's output as its C operand and finish the expression
// there. Running the UP projection first and the GATE projection second gives
// `D = gelu(alpha * AB) * C`, which is what this computes: the activation goes
// on the accumulator, which is the gate, and the multiply on the source, which
// is the up. The separate pass disappears and with it 50.3 MB of traffic.
//
// `LinearCombinationGeneric` cannot express it -- that one is
// `activation(alpha * AB + beta * C)`, with the activation after the sum.
#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

namespace flash_vla {
namespace rtx5090 {

/// D = gelu_tanh(alpha * accumulator) * C
///
/// `alpha` is kept so the operator matches the shape CUTLASS expects, but the
/// route always passes 1. `beta` is unused: C is a factor here, not a term, so
/// there is nothing for it to scale. The source is always needed.
template <typename ElementOutput_, int Count, typename ElementAccumulator_ = ElementOutput_,
          typename ElementCompute_ = ElementAccumulator_,
          cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
class GeluMul {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  static cutlass::FloatRoundStyle const kRound = Round;
  //: The activation is a transcendental, so the epilogue is not cheap and
  //: CUTLASS should schedule it as such.
  static bool const kIsHeavy = true;

  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentCompute = cutlass::Array<ElementCompute, kCount>;

  struct Params {
    ElementCompute alpha;
    ElementCompute beta;
    ElementCompute const *alpha_ptr;
    ElementCompute const *beta_ptr;

    CUTLASS_HOST_DEVICE
    Params()
        : alpha(ElementCompute(1)), beta(ElementCompute(1)),
          alpha_ptr(nullptr), beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta)
        : alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) {}
  };

 private:
  ElementCompute alpha_;

 public:
  CUTLASS_HOST_DEVICE
  explicit GeluMul(Params const &params) : alpha_(params.alpha) {}

  //: Always: C is the up projection, and without it the result is not the
  //: gated feed-forward at all.
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  //: Stream-K reduces the partial accumulators before the epilogue runs, so a
  //: split never sees a partial gate through the activation.
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentOutput const &source) const {
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> to_acc;
    cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> to_src;
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> to_out;

    FragmentCompute gate = to_acc(accumulator);
    FragmentCompute up = to_src(source);

    cutlass::multiplies<FragmentCompute> mul;
    if (alpha_ != ElementCompute(1)) {
      cutlass::multiply_add<FragmentCompute> scale;
      gate = scale(gate, alpha_, FragmentCompute());
    }
    //: `GELU_taylor` is the tanh approximation, which is the one Pi0 uses;
    //: plain `GELU` in CUTLASS is the erf form and is a different function.
    cutlass::epilogue::thread::GELU_taylor<FragmentCompute> activation;
    return to_out(mul(activation(gate), up));
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    //: Reached only if a caller says the source is not needed, which this
    //: operator never does. Returning the un-gated activation keeps the
    //: function total rather than silently returning zeros.
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> to_acc;
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> to_out;
    cutlass::epilogue::thread::GELU_taylor<FragmentCompute> activation;
    return to_out(activation(to_acc(accumulator)));
  }
};

}  // namespace rtx5090
}  // namespace flash_vla
