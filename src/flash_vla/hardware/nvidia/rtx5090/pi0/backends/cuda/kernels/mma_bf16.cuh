// Shared bf16 `mma.sync` fragment helpers for the RTX 5090 expert kernels.
//
// sm_120 has neither `wgmma` nor `tcgen05` -- both are rejected by ptxas for
// this target, see the hardware axis's measured/isa-support.md -- so
// `mma.sync` is the only tensor-core path on this part and every mainloop in
// this directory is built from m16n8k16. The fragment layouts below are PTX
// ISA 9.3 section 9.7.14.1 ("Matrix Fragments for mma.m16n8k16").
#pragma once

#include <cstdint>
#include <cuda_bf16.h>

namespace flash_vla {
namespace rtx5090 {

//: D += A * B for one m16n8k16 tile, bf16 inputs accumulated in fp32. Consumer
//: Blackwell runs fp32 accumulate at half the fp16-accumulate rate, 512
//: FLOP/cycle/SM [mma.rate.sm.bf16]; fp32 is still what these kernels need.
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

//: Two adjacent bf16 as the 32-bit register half an mma operand is made of.
__device__ __forceinline__ uint32_t pack(const __nv_bfloat16 *p) {
  return (uint32_t(__bfloat16_as_ushort(p[1])) << 16) | __bfloat16_as_ushort(p[0]);
}

// A (16 x 16, row-major [m][k]): lane L holds rows L/4 and L/4+8 at column
// pairs (L%4)*2 and (L%4)*2+8.
//
// Callers must pad `ld` so the eight row groups land on different banks: the
// row stride in 4-byte banks is ld/2, and ld/2 % 32 == 0 puts all eight on one.
// An unpadded 256-wide bf16 tile does exactly that, and cost the attention
// kernel a measured 8-way conflict on every one of these loads.
__device__ __forceinline__ void load_a(uint32_t (&a)[4],
                                       const __nv_bfloat16 *tile, int32_t ld,
                                       int32_t k0, int32_t lane) {
  const int32_t r = lane >> 2, c = (lane & 3) * 2;
  a[0] = pack(tile + int64_t(r) * ld + k0 + c);
  a[1] = pack(tile + int64_t(r + 8) * ld + k0 + c);
  a[2] = pack(tile + int64_t(r) * ld + k0 + c + 8);
  a[3] = pack(tile + int64_t(r + 8) * ld + k0 + c + 8);
}

// B (16 x 8, `.col` so it is indexed [n][k]): lane L holds column n0 + L/4 at
// the same two row pairs. Any caller whose operand is naturally [k][n] has to
// transpose it on the way into shared memory.
__device__ __forceinline__ void load_b(uint32_t (&b)[2],
                                       const __nv_bfloat16 *tile, int32_t ld,
                                       int32_t n0, int32_t k0, int32_t lane) {
  const int32_t n = n0 + (lane >> 2), c = (lane & 3) * 2;
  b[0] = pack(tile + int64_t(n) * ld + k0 + c);
  b[1] = pack(tile + int64_t(n) * ld + k0 + c + 8);
}

// `ldmatrix` builds the same fragments in one instruction instead of eight
// scalar loads and a pack, by doing the lane shuffle in hardware. Each lane
// supplies the address of one 8-element row and the unit redistributes them.
//
// Two requirements the callers must meet, both checked by construction here:
// every row address must be 16-byte aligned, so the padded row stride must be
// a multiple of 8 bf16; and the rows an instruction gathers should start in
// different banks, which the same padding already gives.

//: A (16 x 16) from a row-major [m][k] tile. Lanes 0-7 and 8-15 supply the two
//: row halves of the first k octet, 16-31 the second, which is exactly the
//: a[0..3] order `mma.m16n8k16` wants.
__device__ __forceinline__ void ldmatrix_a(uint32_t (&a)[4],
                                           const __nv_bfloat16 *tile,
                                           int32_t ld, int32_t k0,
                                           int32_t lane) {
  const __nv_bfloat16 *p =
      tile + int64_t(lane & 15) * ld + k0 + ((lane >> 4) << 3);
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}

//: B (16 x 8) from a [n][k] tile. Only the first 16 lanes' addresses are read
//: for `.x2`; the rest are computed the same way so every address stays in
//: range of the tile.
__device__ __forceinline__ void ldmatrix_b(uint32_t (&b)[2],
                                           const __nv_bfloat16 *tile,
                                           int32_t ld, int32_t n0, int32_t k0,
                                           int32_t lane) {
  const __nv_bfloat16 *p =
      tile + int64_t(n0 + (lane & 7)) * ld + k0 + (((lane >> 3) & 1) << 3);
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(b[0]), "=r"(b[1]) : "r"(s));
}

//: B (16 x 8) from a tile stored [k][n] -- the layout a row-major weight
//: already has. `.trans` transposes each 8x8 as it loads, so the staging path
//: never has to: storing [n][k] means scattering eight 2-byte values per
//: thread, and with any 16-byte-aligned row stride the n step of 8 puts all of
//: them in one bank. Here the store is one 16-byte vector in natural order.
__device__ __forceinline__ void ldmatrix_b_trans(uint32_t (&b)[2],
                                                 const __nv_bfloat16 *tile,
                                                 int32_t ldn, int32_t n0,
                                                 int32_t k0, int32_t lane) {
  const __nv_bfloat16 *p =
      tile + int64_t(k0 + (lane & 7) + (((lane >> 3) & 1) << 3)) * ldn + n0;
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(b[0]), "=r"(b[1]) : "r"(s));
}

//: D (16 x 8 fp32): lane L holds rows L/4 and L/4+8 at columns (L%4)*2 and +1.
//: The two columns are adjacent, which is what lets a RoPE epilogue rotate a
//: pair without leaving registers.
__device__ __forceinline__ int32_t d_row(int32_t lane, int32_t i) {
  return (lane >> 2) + (i >> 1) * 8;
}

__device__ __forceinline__ int32_t d_col(int32_t lane, int32_t i) {
  return (lane & 3) * 2 + (i & 1);
}

}  // namespace rtx5090
}  // namespace flash_vla
