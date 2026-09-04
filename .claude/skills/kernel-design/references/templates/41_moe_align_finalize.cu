// Template 41 -- the MoE pipeline around the grouped GEMM (sm90).
//
// Template 14 does the expert GEMM and assumes someone handed it a block-to-
// expert map. This is that someone, plus the combine on the way out. Between
// them they are most of what a fused MoE layer costs that is not matmul:
//
//     route (top-k) -> ALIGN -> permute -> grouped GEMM -> FINALIZE
//
// WHY THE ALIGN KERNEL EXISTS. Routing gives each expert a data-dependent,
// arbitrary token count. A GEMM wants whole tiles. The align pass sorts token
// indices by expert AND PADS EACH EXPERT'S RUN UP TO THE GEMM'S BLOCK SIZE, so
// every block the scheduler hands out belongs to exactly one expert and is
// full. That is what lets template 14's inner loop have no predication and no
// indirection -- the irregularity is paid once here, in a kernel that moves
// indices rather than activations.
//
//   * The padding is not waste to be minimised away. Removing it moves a
//     branch into the GEMM's mainloop, where it costs far more than the empty
//     rows do.
//   * The scatter is an atomicAdd per token on a per-expert cursor. Contention
//     is bounded by the expert count, not the token count, which is why this is
//     cheap [atom.rate.addr] -- and why a routing scheme with very few hot
//     experts makes it suddenly expensive.
//
// WHY FINALIZE FUSES THE SHARED EXPERT. Each token's output is the weighted sum
// of its top-k expert results, gathered back through the permutation. A model
// with a shared expert then adds that too. Done separately, the add is a full
// extra read and write of the activation plus a launch; done here it rides the
// gather that was already happening.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
// The permutation is the part to test first: an off-by-one in the inverse map
// silently mixes tokens between experts and still produces plausible output.
//
// CHECK-PTX: atom\.global\.add\.u32
// CHECK-PTX: griddepcontrol\.wait
// CHECK-PTX: griddepcontrol\.launch_dependents
// CHECK-PTX: ld\.global(\.nc)?\.v4
// CHECK-PTX: st\.global

#include "elementwise_sm90.cuh"

namespace {

using Element = __nv_bfloat16;
constexpr int kVec = tmpl::vec_size_of<Element>;
constexpr int kThreads = 256;
// Must equal the grouped GEMM's BLOCK_M, or the padding does not line up with
// the tiles it exists to fill.
constexpr int32_t kBlockM = 64;
constexpr int32_t kMaxExperts = 256;

}  // namespace

// Single CTA: the padded prefix sum is over experts, not tokens, so there is
// nothing to parallelise across and a second launch would cost more than the
// work. Produces the block-to-expert map template 14 consumes.
__global__ __launch_bounds__(kThreads) void moe_align_prefix_kernel(
    const int32_t* __restrict__ expert_counts,  // (num_experts,)
    int32_t* __restrict__ expert_offsets,       // (num_experts + 1,) padded
    int32_t* __restrict__ block_expert,         // (max_blocks,) -1 = padding
    int32_t* __restrict__ num_blocks_out, int32_t num_experts) {
  __shared__ int32_t s_off[kMaxExperts + 1];

  const int32_t tid = static_cast<int32_t>(threadIdx.x);

  // Serial scan over experts. num_experts is in the hundreds, so a parallel
  // scan's setup would exceed the scan.
  if (tid == 0) {
    int32_t run = 0;
    for (int32_t e = 0; e < num_experts; ++e) {
      s_off[e] = run;
      // Round UP to a whole number of GEMM blocks: this rounding is the entire
      // purpose of the pass.
      run += (expert_counts[e] + kBlockM - 1) / kBlockM * kBlockM;
    }
    s_off[num_experts] = run;
    *num_blocks_out = run / kBlockM;
  }
  __syncthreads();

  for (int32_t e = tid; e <= num_experts; e += kThreads) {
    expert_offsets[e] = s_off[e];
  }

  // Every block gets the expert that owns it. A block past the last expert is
  // marked -1 so the GEMM's scheduler can skip it before arming any barrier --
  // template 14's skip-before-arm rule.
  const int32_t total_blocks = s_off[num_experts] / kBlockM;
  for (int32_t b = tid; b < total_blocks; b += kThreads) {
    const int32_t row = b * kBlockM;
    int32_t owner = -1;
    for (int32_t e = 0; e < num_experts; ++e) {
      if (row >= s_off[e] && row < s_off[e + 1]) { owner = e; break; }
    }
    block_expert[b] = owner;
  }
}

// Scatters token indices into their expert's padded run. One atomic per
// (token, k) pair against a per-expert cursor: the contention is the expert
// count, not the token count.
__global__ __launch_bounds__(kThreads) void moe_scatter_kernel(
    const int32_t* __restrict__ topk_expert,   // (tokens, k)
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ cursor,              // (num_experts,) zeroed
    int32_t* __restrict__ sorted_token_ids,    // padded slots pre-filled
    int32_t* __restrict__ expanded_to_permuted,
    int32_t num_expanded) {
  const int32_t idx = static_cast<int32_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (idx >= num_expanded) { return; }

  const int32_t e = topk_expert[idx];
  const int32_t slot = expert_offsets[e] + atomicAdd(&cursor[e], 1);
  sorted_token_ids[slot] = idx;
  // The inverse map, built here rather than by a second pass: finalize needs to
  // go the other way and deriving it later would mean another sort.
  expanded_to_permuted[idx] = slot;
}

// Gathers each token's top-k expert outputs, weights them, and adds the shared
// expert in the same traversal.
__global__ __launch_bounds__(kThreads) void moe_finalize_kernel(
    const Element* __restrict__ permuted_out,  // grouped GEMM output
    const int32_t* __restrict__ expanded_to_permuted,
    const float* __restrict__ topk_weights,    // (tokens, k)
    const Element* __restrict__ shared_out,    // may be null
    Element* __restrict__ out, int32_t d, int32_t topk) {
  const int32_t token = static_cast<int32_t>(blockIdx.x);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);

  // PDL-WAIT: before the first read of the grouped GEMM's output.
  // DERIVED from the data dependency, never swept.
  tmpl::pdl_wait();

  for (int32_t i = tid * kVec; i < d; i += kThreads * kVec) {
    float acc[kVec] = {};

    for (int32_t k = 0; k < topk; ++k) {
      const int32_t src = expanded_to_permuted[token * topk + k];
      const float w = topk_weights[token * topk + k];
      tmpl::FloatVec<Element, kVec> v;
      v.cast_load(permuted_out + static_cast<int64_t>(src) * d + i);
      #pragma unroll
      for (int32_t j = 0; j < kVec; ++j) { acc[j] += w * v[j]; }
    }

    // The shared expert rides the gather rather than costing its own pass.
    if (shared_out != nullptr) {
      tmpl::FloatVec<Element, kVec> s;
      s.cast_load(shared_out + static_cast<int64_t>(token) * d + i);
      #pragma unroll
      for (int32_t j = 0; j < kVec; ++j) { acc[j] += s[j]; }
    }

    tmpl::FloatVec<Element, kVec> o;
    #pragma unroll
    for (int32_t j = 0; j < kVec; ++j) { o[j] = acc[j]; }
    o.cast_store(out + static_cast<int64_t>(token) * d + i);
  }

  // PDL-TRIGGER: after the gather loop.
  // SWEPT: not yet measured. This kernel has almost no tail, so the useful
  // range is narrow -- entry and post-wait are the other candidates.
  tmpl::pdl_trigger();
}
