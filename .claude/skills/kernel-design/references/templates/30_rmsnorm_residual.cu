// Template 30 -- RMSNorm with a fused residual add (sm90).
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
//      full read and write of the row plus a launch.  Fused, the residual is
//      added while the row is already in registers for the sum, and the updated
//      residual is stored on the way past -- the next block needs it too.
//   4. `weight_bias` is not padding.  Passing 0 gives `w * x_hat`, passing 1
//      gives `(1 + w) * x_hat`, which is the Gemma convention.  One kernel,
//      two model families, no branch in the inner loop.
//
// PDL brackets the whole thing.  A norm is microseconds of work; at decode
// shapes the launch ramp is comparable to the kernel [launch.lat.dev.ramp].
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-PTX: griddepcontrol\.wait
// CHECK-PTX: griddepcontrol\.launch_dependents
// CHECK-PTX: shfl\.sync\.bfly
// CHECK-PTX: rsqrt\.approx
// CHECK-PTX: ld\.global(\.nc)?\.v4

#include "elementwise_sm90.cuh"

namespace {

using Element = __nv_bfloat16;
constexpr int kVec = tmpl::vec_size_of<Element>;  // 8 bf16 per 16-byte load
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / tmpl::kWarpThreads;

}  // namespace

// One CTA per token.  `residual` is read AND written: the block's running
// residual stream advances here, which is the whole reason the add is fused in
// rather than left to a separate kernel.
__global__ __launch_bounds__(kThreads) void rmsnorm_residual_kernel(
    const Element* __restrict__ input, Element* __restrict__ residual,
    const Element* __restrict__ weight, Element* __restrict__ output,
    int32_t d, float weight_bias, float eps) {
  __shared__ float smem[kWarps];

  const int64_t row = static_cast<int64_t>(blockIdx.x) * d;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t vecs = d / kVec;

  // The dependent grid's prologue can overlap the producer's tail; nothing
  // below reads producer output before this point.
  tmpl::pdl_wait();

  // Pass 1: add the residual, keep the sum of squares of the RESULT (the norm
  // is over x + residual, not over x), and publish the updated residual now so
  // pass 2 does not have to recompute or re-add it.
  float sum_sq = 0.f;
  #pragma unroll 1
  for (int32_t i = tid; i < vecs; i += kThreads) {
    tmpl::FloatVec<Element, kVec> x, r;
    x.cast_load(input + row + i * kVec);
    r.cast_load(residual + row + i * kVec);
    #pragma unroll
    for (int32_t j = 0; j < kVec; ++j) {
      x[j] += r[j];
      sum_sq += x[j] * x[j];
    }
    x.cast_store(residual + row + i * kVec);
  }

  const float rms_rcp =
      rsqrtf(tmpl::block_reduce<false>(sum_sq, smem, kWarps) / static_cast<float>(d) + eps);

  // Pass 2: re-read the row we just wrote.  It is L2-resident by construction,
  // so this traversal is much cheaper than the first -- which is why holding it
  // in registers would buy less than it costs in occupancy.
  #pragma unroll 1
  for (int32_t i = tid; i < vecs; i += kThreads) {
    tmpl::FloatVec<Element, kVec> x, w;
    x.cast_load(residual + row + i * kVec);
    w.cast_load(weight + i * kVec);
    #pragma unroll
    for (int32_t j = 0; j < kVec; ++j) {
      x[j] = x[j] * rms_rcp * (weight_bias + w[j]);
    }
    x.cast_store(output + row + i * kVec);
  }

  // Released after the stores are issued, before this CTA's own teardown.
  tmpl::pdl_launch_dependents();
}
