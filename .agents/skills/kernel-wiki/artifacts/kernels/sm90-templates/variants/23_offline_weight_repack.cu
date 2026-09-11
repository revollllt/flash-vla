// Template 23 -- offline weight repack, the other half of a quantized kernel.
//
// Templates 20-22 all say the same thing in their headers: the dequant is cheap
// only because the weights were permuted offline.  This is that permutation.
// A quantized kernel and its packer are ONE artifact with two halves; shipping
// the kernel alone produces wrong numbers, not a slowdown, and a checkpoint
// packed for one kernel config cannot be fed to another.
//
// WHY THE PERMUTATION LOOKS LIKE THAT
//
// Marlin packs eight 4-bit values into a word in the order
// {0, 2, 4, 6, 1, 3, 5, 7} rather than sequentially.  That is not arbitrary and
// it is not about memory coalescing -- it is dictated by the dequant.  The
// two-mask lop3 trick (quant_sm90.cuh) extracts nibbles {0,4} into one half2
// and {1,5} into the next.  Under this pack order those are logical values
// {v0,v1} and {v2,v3}: consecutive, exactly what the MMA operand wants.  Under
// sequential packing the same dequant would produce {v0,v4} and {v1,v5}, and
// the kernel would need a shuffle per fragment to repair it.
//
// AWQ stores its nibbles in the order {0, 4, 1, 5, 2, 6, 3, 7}, which is the
// EXACT INVERSE of Marlin's pack order -- applying one after the other is the
// identity.  Both formats solved the same problem from opposite ends.
//
// The consequence is worth knowing before writing a converter: since the two
// orders are inverses, AWQ's stored nibble order for sequential values already
// IS Marlin's pack order, so the NIBBLE half of an AWQ -> Marlin conversion is
// a no-op.  What a real AWQ -> Marlin repack still has to do is the tile
// permutation (gathering each tensor-core tile's ownership) and, for GPTQ
// act-order, the K permutation.  That is why the AWQ repack is the shorter of
// the two upstream kernels.
//
// Chain verified end to end by emulating each step over a full group:
//   AWQ storage -> de-interleave -> sequential -> Marlin pack -> the two-mask
//   lop3 dequant emits (v0,v1) (v2,v3) (v4,v5) (v6,v7), in order.
//
// TWO PHILOSOPHIES, AND WHEN EACH IS RIGHT
//
//   * Marlin: permute INTO one kernel's layout.  Fastest, because every
//     addressing decision is baked into the data.  Costs one layout per kernel
//     configuration and a conversion pass per checkpoint.
//   * humming: NORMALIZE every source format into one canonical dense packing,
//     and let the JIT specialize the kernel instead of the data.  One layout
//     serves any bit width under 8; the format-specific work moves into
//     codegen.  Pays a little addressing generality at run time for not having
//     a layout explosion across formats x kernels.
//
// Pick by how many (format, kernel) pairs you must support.  One pair: permute.
// A matrix of them: normalize.
//
// Both packers are also where format quirks get canonicalized once instead of
// per kernel launch -- negative zero folded to zero, a group's base exponent
// factored out of block scales (see template 22's residual-delta contract).
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
// A repack in particular MUST be validated by round-tripping through the
// matching unpack and comparing against the source tensor -- the permutations
// here are exactly the kind of thing that is silently wrong.
//
// CHECK-GRADE: structural
// CHECK-PTX: cp\.async\.cg\.shared\.global
// CHECK-PTX: cp\.async\.wait_group
// CHECK-PTX: shl\.b32
// CHECK-PTX: st\.global

#include <cstdint>

#include "sm90_common.cuh"

namespace {

// Marlin's repack tile.  16 rows of K by 64 columns of N is one tensor-core
// tile group; the packer works in exactly the unit the kernel reads.
constexpr int kTileK = 16;
constexpr int kTileN = 64;
constexpr int kPackFactor = 8;  // 4-bit values per uint32
constexpr int kThreads = 256;

// Nibble i of the output word carries logical value kMarlinPack[i].
// Verified property: with this order, the two-mask lop3 dequant yields
// {v0,v1} then {v2,v3} from one word, and {v4,v5} then {v6,v7} from word >> 8.
__device__ __constant__ int kMarlinPack[8] = {0, 2, 4, 6, 1, 3, 5, 7};

// AWQ's storage order, the inverse of the above.  Gathering with it returns a
// sequentially-ordered group, which is what any re-pack starts from.
__device__ __constant__ int kAwqGather[8] = {0, 4, 1, 5, 2, 6, 3, 7};

}  // namespace

// ------------------------------------------------------ Marlin-style repack

// Reads a row-major GPTQ tile and writes Marlin's layout.  One thread owns the
// eight values that will share one output word, so the interleave is a local
// reorder rather than a cross-thread shuffle -- the packer is arranged around
// the same ownership the kernel will have.
__global__ __launch_bounds__(kThreads) void marlin_repack_kernel(
    const uint32_t* __restrict__ src,   // GPTQ: (K/8, N), 8 values per word
    const uint32_t* __restrict__ k_perm,// optional act-order permutation, or null
    uint32_t* __restrict__ dst, int32_t n_size, int32_t k_tiles) {
  __shared__ uint32_t stage[2][kTileK * kTileN / kPackFactor];

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t words_per_tile = kTileK * kTileN / kPackFactor;

  auto fetch = [&](int32_t tile, int32_t buf) {
    for (int32_t i = tid * 4; i < words_per_tile; i += kThreads * 4) {
      tmpl::cp_async_16(&stage[buf][i],
                        src + static_cast<int64_t>(tile) * words_per_tile + i);
    }
    tmpl::cp_async_commit();
  };

  fetch(0, 0);

  for (int32_t tile = 0; tile < k_tiles; ++tile) {
    const int32_t buf = tile & 1;
    if (tile + 1 < k_tiles) { fetch(tile + 1, buf ^ 1); }
    // One tile in flight while this one is consumed; the packer is pure
    // bandwidth, so its own pipeline matters as much as the kernel's.
    tmpl::cp_async_wait<1>();
    __syncthreads();

    if (tid < kTileN) {
      uint32_t vals[8];
      #pragma unroll
      for (int32_t i = 0; i < 8; ++i) {
        // act-order (GPTQ desc_act) reorders K, so the gather index is data,
        // not arithmetic.  Folding it in here is why the kernel's inner loop
        // has no indirection at all.
        const int32_t k = k_perm ? static_cast<int32_t>(k_perm[i]) : i;
        const uint32_t word = stage[buf][(k / kPackFactor) * kTileN + tid];
        vals[i] = (word >> ((k % kPackFactor) * 4)) & 0xF;
      }

      uint32_t packed = 0;
      #pragma unroll
      for (int32_t i = 0; i < 8; ++i) {
        packed |= vals[kMarlinPack[i]] << (i * 4);
      }
      dst[static_cast<int64_t>(tile) * n_size + tid] = packed;
    }
    __syncthreads();
  }
}

// ------------------------------------------------- humming-style normalize

// Undoes a source format's interleave, producing sequentially-ordered values.
// This is the first half of "normalize, then let the JIT specialize": once the
// data is in one canonical order, one packer serves every bit width and one
// kernel generator serves every format.
__global__ __launch_bounds__(kThreads) void awq_deinterleave_kernel(
    const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
    int64_t num_words) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (idx >= num_words) { return; }

  const uint32_t word = src[idx];
  uint32_t out = 0;
  #pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    const uint32_t v = (word >> (kAwqGather[i] * 4)) & 0xF;
    out |= v << (i * 4);
  }
  dst[idx] = out;
}

// Dense N-bit packing, values crossing word boundaries.  This is what lets one
// layout cover every width under 8 bits: at 4 or 8 the values align and this
// degenerates to shifts, but at 3, 5 or 6 a value straddles two words and the
// spill has to be written explicitly.  A packer that only handles the aligned
// widths quietly limits the formats the whole stack can serve.
template <uint32_t kNumBits>
__global__ __launch_bounds__(kThreads) void pack_nbit_kernel(
    const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
    int64_t num_values) {
  constexpr uint32_t kMask = (1u << kNumBits) - 1u;
  // 32 values in, kNumBits words out: the smallest group whose bit count is a
  // whole number of words for any width.
  const int64_t group = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (group * 32 >= num_values) { return; }

  uint32_t out[kNumBits] = {};
  #pragma unroll
  for (uint32_t i = 0; i < 32; ++i) {
    const uint32_t bit = i * kNumBits;
    const uint32_t word = bit / 32, off = bit % 32;
    const uint32_t v = src[group * 32 + i] & kMask;
    out[word] |= v << off;
    if (off + kNumBits > 32) {
      // The straddle: the high part belongs to the next word.
      out[word + 1] |= v >> (32 - off);
    }
  }
  #pragma unroll
  for (uint32_t i = 0; i < kNumBits; ++i) {
    dst[group * kNumBits + i] = out[i];
  }
}

// 4-bit is the common case; 6-bit is instantiated to keep the straddling path
// compiled and asserted, since that is the branch an aligned-only packer omits.
template __global__ void pack_nbit_kernel<4>(const uint32_t*, uint32_t*, int64_t);
template __global__ void pack_nbit_kernel<6>(const uint32_t*, uint32_t*, int64_t);
