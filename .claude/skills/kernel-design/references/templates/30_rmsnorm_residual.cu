// Template 30 -- RMSNorm with a fused residual add, and PDL placed on purpose.
//
// The simplest of the glue ops and the one whose mistakes are hardest to see,
// because a norm that is subtly wrong still trains and still generates text.
//
//   1. Accumulate in fp32, always.  The inputs are bf16 and the sum of squares
//      over a few thousand elements will not survive bf16's 8 mantissa bits.
//      This costs nothing: the upcast rides the load.
//   2. Two traversals of the row, not one.  The row is too long to hold in
//      registers, so it is read once for the sum and once for the scale.  That
//      makes the kernel read 2x + write 1x its row, which IS its roofline --
//      any "optimization" that does not reduce that traversal count is noise.
//   3. Fuse the residual add INTO the norm.  A transformer block writes
//      `x = x + attn(x)` then norms it; done as two kernels that is an extra
//      full read and write of the row plus a launch.
//   4. `weight_bias` is not padding.  Passing 0 gives `w * x_hat`, passing 1
//      gives `(1 + w) * x_hat`, which is the Gemma convention.  One kernel,
//      two model families, no branch in the inner loop.
//
// PDL PLACEMENT -- the part that is easy to get mechanically wrong.
//
// The wait is NOT at the top.  `weight` is a model parameter: the previous
// kernel did not write it, so its load has no reason to sit behind the
// dependency and every reason to overlap the producer's tail.  Only `input` and
// `residual` are producer data, so the wait goes immediately above them.  The
// cost is holding the weight row in registers across the reduction, and that
// register-against-latency trade is exactly what the ablation in
// wiki/pdl-placement.md measures -- it is not free and it is not always a win.
//
// The trigger is at the end, and here that is close to a NO-OP rather than an
// optimization: the kickoff fires automatically once every CTA exits, so a
// trigger on the last line buys almost nothing.  It cannot move earlier,
// because the dependent reads `output` and nothing is written until pass 2
// finishes.  The honest conclusion is that RMSNorm is a poor PDL *producer* and
// a good PDL *consumer*; a chain that wants overlap here should get it from the
// next kernel waiting late, not from this one triggering early.
//
// The release fence before the trigger is required, not defensive: NVIDIA's own
// header says the trigger "provides no memory visibility guarantee itself".
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-PTX: griddepcontrol\.wait
// CHECK-PTX: griddepcontrol\.launch_dependents
// CHECK-PTX: membar\.gl
// CHECK-PTX: shfl\.sync\.bfly
// CHECK-PTX: rsqrt\.approx
// CHECK-PTX: ld\.global(\.nc)?\.v4

#include "elementwise_sm90.cuh"

namespace {

using Element = __nv_bfloat16;
constexpr int kVec = tmpl::vec_size_of<Element>;  // 8 bf16 per 16-byte load
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / tmpl::kWarpThreads;
// Fixed hidden size, as these kernels always are in a deployed model.  It is
// also what makes the parameter prefetch below expressible: a dynamic row
// cannot be held in a fixed register array.
constexpr int kVecsPerThread = 2;
constexpr int kRowElems = kVecsPerThread * kThreads * kVec;  // 4096

}  // namespace

// One CTA per token.  `residual` is read AND written: the block's running
// residual stream advances here, which is the whole reason the add is fused in
// rather than left to a separate kernel.
__global__ __launch_bounds__(kThreads) void rmsnorm_residual_kernel(
    const Element* __restrict__ input, Element* __restrict__ residual,
    const Element* __restrict__ weight, Element* __restrict__ output,
    float weight_bias, float eps) {
  __shared__ float smem[kWarps];

  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowElems;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);

  // ABOVE THE WAIT: producer-independent.  These loads issue while the previous
  // kernel is still draining its tail.
  tmpl::FloatVec<Element, kVec> w[kVecsPerThread];
  #pragma unroll
  for (int32_t v = 0; v < kVecsPerThread; ++v) {
    w[v].cast_load(weight + (v * kThreads + tid) * kVec);
  }

  // Everything below reads what the producer wrote.
  tmpl::pdl_wait();

  // Pass 1: add the residual, take the sum of squares of the RESULT (the norm
  // is over x + residual), and publish the updated residual so pass 2 does not
  // have to re-add it.
  float sum_sq = 0.f;
  #pragma unroll
  for (int32_t v = 0; v < kVecsPerThread; ++v) {
    const int32_t i = (v * kThreads + tid) * kVec;
    tmpl::FloatVec<Element, kVec> x, r;
    x.cast_load(input + row + i);
    r.cast_load(residual + row + i);
    #pragma unroll
    for (int32_t j = 0; j < kVec; ++j) {
      x[j] += r[j];
      sum_sq += x[j] * x[j];
    }
    x.cast_store(residual + row + i);
  }

  const float rms_rcp = rsqrtf(
      tmpl::block_reduce<false>(sum_sq, smem, kWarps) / static_cast<float>(kRowElems) + eps);

  // Pass 2: re-read the row we just wrote.  It is L2-resident by construction,
  // so this traversal is much cheaper than the first.
  #pragma unroll
  for (int32_t v = 0; v < kVecsPerThread; ++v) {
    const int32_t i = (v * kThreads + tid) * kVec;
    tmpl::FloatVec<Element, kVec> x;
    x.cast_load(residual + row + i);
    #pragma unroll
    for (int32_t j = 0; j < kVec; ++j) {
      x[j] = x[j] * rms_rcp * (weight_bias + w[v][j]);
    }
    x.cast_store(output + row + i);
  }

  // The dependent reads `output`; the trigger orders nothing on its own.
  tmpl::pdl_release_fence();
  tmpl::pdl_trigger();
}
