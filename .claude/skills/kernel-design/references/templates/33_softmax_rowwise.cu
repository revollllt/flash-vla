// Template 33 -- row-wise softmax, and when the online form is worth it (sm90).
//
// This is the standalone softmax -- a logit row, a routing row, a sampling
// distribution -- not attention's. Template 12 owns that one, and the
// difference is worth stating because the same words describe two different
// kernels:
//
//   * ATTENTION's softmax is online because it MUST be: the row is produced a
//     tile at a time by a GEMM and never exists in full, so the max and sum are
//     carried and corrected as tiles arrive.
//   * A STANDALONE softmax has the whole row in memory. Online buys one fewer
//     traversal (2 instead of 3), not correctness.
//
// The traversal count is the roofline, so that is the whole optimization:
//
//     naive : read for max, read for sum, read for normalize   -- 3 reads
//     online: read carrying (max, sum) together, read to write -- 2 reads
//
// The online pass works because a running sum under an old max is corrected by
// one multiply when the max grows: `sum = sum * exp(m_old - m_new) + ...`.
// That correction is the same identity template 13's split-KV combine uses to
// merge partial softmaxes, and the same one template 12 applies per tile.
//
// Numerics, both non-negotiable:
//   1. Subtract the max before exponentiating. Logits reach 30+ and expf(30)
//      overflows fp16 and dents fp32; the subtraction is what makes softmax
//      stable, not a refinement of it.
//   2. Accumulate in fp32 even for a bf16 row.
//
// exp2f with log2(e) folded in, as in template 12: the hardware has a fast path
// for exp2 and the constant costs one multiply for the whole row.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-PTX: griddepcontrol\.wait
// CHECK-PTX: griddepcontrol\.launch_dependents
// CHECK-PTX: ex2\.approx
// CHECK-PTX: shfl\.sync\.bfly
// CHECK-PTX: ld\.global(\.nc)?\.v4

#include "elementwise_sm90.cuh"

namespace {

using Element = __nv_bfloat16;
constexpr int kVec = tmpl::vec_size_of<Element>;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / tmpl::kWarpThreads;
constexpr float kLog2e = 1.4426950408889634f;

}  // namespace

// One CTA per row. Two traversals: the first carries a running (max, sum) pair
// per thread, the second writes.
__global__ __launch_bounds__(kThreads) void softmax_rowwise_kernel(
    const Element* __restrict__ input, Element* __restrict__ output, int32_t d) {
  __shared__ float smem[kWarps];

  const int64_t row = static_cast<int64_t>(blockIdx.x) * d;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t vecs = d / kVec;

  // PDL-WAIT: before the first read of `input`.
  // DERIVED from the data dependency, never swept.
  tmpl::pdl_wait();

  // Per-thread running state over the slice this thread strides across.
  float m = -INFINITY, l = 0.f;
  #pragma unroll 1
  for (int32_t i = tid; i < vecs; i += kThreads) {
    tmpl::FloatVec<Element, kVec> v;
    v.cast_load(input + row + i * kVec);
    #pragma unroll
    for (int32_t j = 0; j < kVec; ++j) {
      const float m_new = fmaxf(m, v[j]);
      // One correction when the max moves, then the new term. This is the
      // whole online trick; without it the sum would need its own pass.
      l = l * exp2f((m - m_new) * kLog2e) + exp2f((v[j] - m_new) * kLog2e);
      m = m_new;
    }
  }

  // PDL-TRIGGER: after pass 1 has read the whole row.
  // SWEPT: not yet measured. Pass 2 re-reads from L2 rather than from the
  // producer, so points inside pass 2 are legal candidates too.
  tmpl::pdl_trigger();

  // Combining two threads' (m, l) is the same rescale-and-add, so the block
  // reduction has to be done in two steps rather than one: the sums are only
  // comparable once every thread is on the block max.
  const float m_block = tmpl::block_reduce<true>(m, smem, kWarps);
  __syncthreads();
  l *= exp2f((m - m_block) * kLog2e);
  const float l_block = tmpl::block_reduce<false>(l, smem, kWarps);

  const float inv = 1.f / l_block;
  #pragma unroll 1
  for (int32_t i = tid; i < vecs; i += kThreads) {
    tmpl::FloatVec<Element, kVec> v;
    v.cast_load(input + row + i * kVec);
    #pragma unroll
    for (int32_t j = 0; j < kVec; ++j) {
      v[j] = exp2f((v[j] - m_block) * kLog2e) * inv;
    }
    v.cast_store(output + row + i * kVec);
  }

}
