// Template 32 -- RoPE, and why the two layouts are not equally cheap (sm90).
//
// RoPE rotates each pair of head-dimension components by an angle that depends
// on position.  Which two components form a pair is a checkpoint convention,
// and the two conventions in use cost different amounts:
//
//   * INTERLEAVED (GPT-J): pairs are (2i, 2i+1) -- ADJACENT.  A thread that
//      loaded a 16-byte vector already holds both halves of every pair it owns,
//      so the partner is `vec[i ^ 1]`, a register.  Zero extra traffic.
//   * HALF (GPT-NeoX, Llama HF): pairs are (i, i + rotary_dim/2) -- OPPOSITE
//      ENDS of the head.  The partner is a SECOND LOAD from the other half, and
//      the sign of the rotation term depends on which half the thread is in.
//
// The interleaved layout is strictly cheaper per element, and it also halves
// the cos/sin traffic: a pair shares one angle, so the table is indexed `i / 2`
// and is half the size.  Under the half layout each element of the pair needs
// its own table entry at its own index.
//
// This is a layout decision, not a kernel decision -- which means it can be
// bought offline.  Permuting a checkpoint's head dimension once at load turns a
// half-layout model into an interleaved one and removes the extra load from
// every token of every forward pass afterwards.  That is the same trade as
// template 23's weight repack: pay once in the packer, not per launch.
//
// One deliberate simplification: the cos/sin tables are read one element at a
// time here so the index arithmetic stays visible.  A production kernel loads
// them vectorized like everything else -- at these sizes the table traffic is
// the same order as the data traffic, so scalar loads there are not free.
//
// Both layouts are instantiated so both paths reach the PTX.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
// RoPE is a place where a sign or an index convention silently produces a model
// that still generates fluent text and scores badly -- parity against the
// reference implementation is not optional.
//
// CHECK-PTX: griddepcontrol\.wait
// CHECK-PTX: griddepcontrol\.launch_dependents
// CHECK-PTX: ld\.global(\.nc)?\.v4
// CHECK-PTX-COUNT: 2 st\.global\.v4

#include "elementwise_sm90.cuh"

namespace {

using Element = __nv_bfloat16;
constexpr int kVec = tmpl::vec_size_of<Element>;
constexpr int kThreads = 128;

enum class RopeLayout : int { kInterleaved = 0, kHalf = 1 };

}  // namespace

// One CTA per (token, head).  cos/sin are precomputed per position: recomputing
// them here would put transcendentals on a bandwidth-bound kernel's critical
// path for values every head of every token shares.
template <RopeLayout kLayout>
__global__ __launch_bounds__(kThreads) void rope_kernel(
    const Element* __restrict__ x, Element* __restrict__ out,
    const float* __restrict__ cos_table, const float* __restrict__ sin_table,
    const int32_t* __restrict__ positions, int32_t head_dim, int32_t rotary_dim) {
  const int32_t token = static_cast<int32_t>(blockIdx.x);
  const int32_t head = static_cast<int32_t>(blockIdx.y);
  const int64_t base =
      (static_cast<int64_t>(token) * gridDim.y + head) * head_dim;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int64_t angle_base = static_cast<int64_t>(positions[token]) * rotary_dim;

  tmpl::pdl_wait();

  for (int32_t i = tid * kVec; i < head_dim; i += kThreads * kVec) {
    tmpl::FloatVec<Element, kVec> v;
    v.cast_load(x + base + i);

    // Past rotary_dim the components are passed through untouched -- partial
    // rotary (rotary_dim < head_dim) is common and silently dropping the tail
    // is a favourite bug.
    if (i < rotary_dim) {
      const tmpl::FloatVec<Element, kVec> before = v;

      if constexpr (kLayout == RopeLayout::kInterleaved) {
        #pragma unroll
        for (int32_t j = 0; j < kVec; ++j) {
          // One angle per PAIR, so the table index is halved.
          const float c = cos_table[angle_base + (i + j) / 2];
          const float s = sin_table[angle_base + (i + j) / 2];
          // The partner is a register neighbour: no load, no shuffle.
          const float partner = before[j ^ 1];
          v[j] = before[j] * c + ((j % 2 == 0) ? -partner : partner) * s;
        }
      } else {
        // The partner lives half a head away, so it costs a second load, and
        // the load is from a different cache line than the one just fetched.
        const bool lower = (i < rotary_dim / 2);
        const int32_t partner_off = lower ? i + rotary_dim / 2 : i - rotary_dim / 2;
        tmpl::FloatVec<Element, kVec> partner;
        partner.cast_load(x + base + partner_off);
        #pragma unroll
        for (int32_t j = 0; j < kVec; ++j) {
          const float c = cos_table[angle_base + i + j];
          const float s = sin_table[angle_base + i + j];
          v[j] = before[j] * c + (lower ? -partner[j] : partner[j]) * s;
        }
      }
    }
    v.cast_store(out + base + i);
  }

  tmpl::pdl_launch_dependents();
}

template __global__ void rope_kernel<RopeLayout::kInterleaved>(
    const Element*, Element*, const float*, const float*, const int32_t*, int32_t, int32_t);
template __global__ void rope_kernel<RopeLayout::kHalf>(
    const Element*, Element*, const float*, const float*, const int32_t*, int32_t, int32_t);
