// Register-side dequantization for mixed-precision GEMM on sm90.
//
// sm90 has NO sub-8-bit tensor core: `mma_sm90_gmma.hpp` contains not one e2m1
// atom, and the narrowest wgmma input is fp8.  Every INT4 and FP4 kernel on
// H100 therefore unpacks weights to bf16/fp16 IN REGISTERS between the shared
// tile and the MMA.  That unpack sits on the critical path of a
// weight-bandwidth-bound kernel, so it is written as bit arithmetic, never as
// integer-to-float conversion.
//
// The shared trick: a low-precision field dropped into the mantissa of a FIXED
// exponent yields `2^e + v` for free, and one subtract removes the 2^e.  A
// float format's fields can likewise be shifted into a wider format's fields,
// with the exponent-bias difference folded into a single multiply.  Both cost
// one or two integer ops per two values.
//
// On sm100/sm120 this file is the wrong answer: those architectures take
// e2m1 and its block scales natively (`mma_sm100_umma.hpp`, `mma_sm120.hpp`),
// so the unpack disappears into the instruction.  Porting a kernel built on
// this header to Blackwell means deleting it, not translating it.
//
// Methods follow Marlin (IST-DASLab, Apache-2.0) as carried by vLLM/SGLang,
// which in turn follows FasterTransformer's interleaved numeric conversion.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace tmpl {

// Three-input logical op from the LUT immediate.  Spelled explicitly because
// ptxas does not always contract `(a & b) | c` into one LOP3.
template <int lut>
__device__ __forceinline__ int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Byte permute: builds a 4-byte result by selecting, per output byte, one of
// the eight bytes of {a, b} named by a nibble of `s`.  That is a table lookup
// with an 8-entry table held in two registers -- the reason every 4-bit ->
// 8-bit conversion below is a LUT rather than arithmetic.
//
// A selector nibble with bit 3 set switches to sign-replicate mode instead of
// byte select, which is why callers mask selectors with 0x7 and not 0xF.
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(s));
  return res;
}

__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
  return static_cast<uint32_t>(lop3<(0xF0 & 0xCC) | 0xAA>(
      static_cast<int>(a), static_cast<int>(b), static_cast<int>(c)));
}

// ------------------------------------------------------- INT4 -> bf16 / fp16

// Four packed uint4 -> two bf16x2.  0x4300 is bf16 128.0, so masking a nibble
// into the low mantissa gives 128 + v exactly; 0x4308 is 128 + 8, so one
// subtract yields the symmetric int4 value v - 8.
//
// bf16 has 7 mantissa bits, so only one nibble fits per lane and the high
// nibbles need a shift.  Set `fold_bias` false when a group zero-point will be
// applied later anyway -- then the -8 is folded into that instead.
template <bool fold_bias = true>
__device__ __forceinline__ void dequant_u4_to_bf16x2(int q, __nv_bfloat162* out) {
  constexpr uint32_t MASK = 0x000f000f;
  constexpr uint32_t EX = 0x43004300;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  out[0] = *reinterpret_cast<__nv_bfloat162*>(&lo);
  out[1] = *reinterpret_cast<__nv_bfloat162*>(&hi);
  if constexpr (fold_bias) {
    constexpr uint32_t SUB = 0x43084308;  // 128 + 8
    out[0] = __hsub2(out[0], *reinterpret_cast<const __nv_bfloat162*>(&SUB));
    out[1] = __hsub2(out[1], *reinterpret_cast<const __nv_bfloat162*>(&SUB));
  }
}

// fp16 has 10 mantissa bits, enough for two nibbles per 16-bit lane, so the
// high half is extracted with a SECOND MASK rather than a shift and its
// factor-of-16 is folded into the fma that removes the bias.  That is one
// instruction fewer than the bf16 path per pair.
//   0x6400 = 1024.0, 0x6408 = 1024 + 8, 0x2c00 = 1/16, 0xd480 = -(1024+8)/16.
template <bool fold_bias = true>
__device__ __forceinline__ void dequant_u4_to_f16x2(int q, __half2* out) {
  constexpr uint32_t LO = 0x000f000f;
  constexpr uint32_t HI = 0x00f000f0;
  constexpr uint32_t EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  if constexpr (fold_bias) {
    constexpr uint32_t SUB = 0x64086408;
    constexpr uint32_t MUL = 0x2c002c00;
    constexpr uint32_t ADD = 0xd480d480;
    out[0] = __hsub2(*reinterpret_cast<__half2*>(&lo),
                     *reinterpret_cast<const __half2*>(&SUB));
    out[1] = __hfma2(*reinterpret_cast<__half2*>(&hi),
                     *reinterpret_cast<const __half2*>(&MUL),
                     *reinterpret_cast<const __half2*>(&ADD));
  } else {
    out[0] = *reinterpret_cast<__half2*>(&lo);
    out[1] = *reinterpret_cast<__half2*>(&hi);
  }
}

// --------------------------------------------------------- FP4 (e2m1) -> bf16

// e2m1 is sign(1) exponent(2) mantissa(1).  Shifting the exponent+mantissa
// right by (bf16_exp_bits - fp4_exp_bits) lands them in bf16's fields; the
// sign moves separately.  The exponent BIAS difference (127 - 1 = 126) cannot
// be shifted in, so it rides one multiply by 2^126 -- which is why this
// returns a value that must then be scaled, and why the block scale below is
// folded into the same multiply in practice.
//
// Four e2m1 values packed in one int -> two bf16x2.
__device__ __forceinline__ void dequant_e2m1_to_bf16x2(int q, __nv_bfloat162* out) {
  constexpr int kShift = 8 - 2;  // bf16 exponent bits - fp4 exponent bits
  constexpr int MASK = 0x70007000;
  int o1 = (q & 0x80008000) | ((q & MASK) >> kShift);
  q <<= 4;
  int o2 = (q & 0x80008000) | ((q & MASK) >> kShift);
  // Reverse order is intentional: the packed nibbles are stored in the order
  // the MMA operand wants, not in value order.
  out[1] = *reinterpret_cast<const __nv_bfloat162*>(&o1);
  out[0] = *reinterpret_cast<const __nv_bfloat162*>(&o2);
}

// Same for fp16; the bias correction is 2^(16-2) = 2^14 and small enough to
// be an ordinary fp16 constant, unlike bf16's 2^126.
__device__ __forceinline__ void dequant_e2m1_to_f16x2(int q, __half2* out) {
  constexpr int kShift = 5 - 2;
  constexpr int MASK = 0x70007000;
  int o1 = (q & 0x80008000) | ((q & MASK) >> kShift);
  q <<= 4;
  int o2 = (q & 0x80008000) | ((q & MASK) >> kShift);
  out[1] = *reinterpret_cast<const __half2*>(&o1);
  out[0] = *reinterpret_cast<const __half2*>(&o2);
  constexpr int kBias = (1 << (5 - 1)) - (1 << (2 - 1));  // 14
  const __half2 bias = __float2half2_rn(float(1 << kBias));
  out[0] = __hmul2(out[0], bias);
  out[1] = __hmul2(out[1], bias);
}

// ------------------------------------------------- e2m1 -> e4m3, scale folded

// Eight packed e2m1 -> eight e4m3, WITH the block scale already applied, in
// about six instructions for all eight values.  Method from humming
// (inclusionAI/humming, Apache-2.0), `datatype/dequant_fused.cuh`.
//
// e2m1 has 8 magnitudes, so the magnitude table is 8 bytes -- exactly one prmt
// source pair.  The trick is that the table is not a constant: it is BUILT per
// call by an integer multiply-add, and `exp_offset` shifts every entry's e4m3
// exponent field at once.  Applying the MXFP4 block scale therefore costs two
// integer ops for the whole group instead of a multiply per value.
//
// Verified semantics (emulating prmt/lop3 over the full input domain):
//   exp_offset = 6 is the identity table {0, .5, 1, 1.5, 2, 3, 4, 6};
//   each +/-1 doubles/halves every entry; index 0 stays exactly zero at every
//   offset, which is what the `- 0x00000400` term buys.
//
// TWO CONTRACTS THE CALLER OWNS, or this silently produces wrong numbers:
//
//  1. `exp_offset` is a small DELTA, not the absolute ue8m0 exponent.  Folding
//     a full block exponent into an e4m3 weight overflows it (e4m3 maxes at
//     448).  The group's base exponent must be factored out offline and
//     re-applied once on the accumulator; humming's `process_mxfp4_w4a8` pass
//     is what computes that residual (and maps -0 to +0 while it is there).
//  2. The nibble order is a PRE-PERMUTED layout, matched to this routine.
//     Output word w byte j takes its magnitude from input nibble 4w+j but its
//     sign from input bit 8j+3 (w=0) or 8j+7 (w=1).  Those coincide only for a
//     packing that puts each value's sign and magnitude at that pair -- the
//     same offline-packer contract Marlin has.  Copying this routine without
//     its packer produces garbage, not a slowdown.
__device__ __forceinline__ void unpack_e2m1_to_e4m3x8(uint32_t qb,
                                                      uint32_t exp_offset,
                                                      uint32_t (&out)[2]) {
  // Two registers = the 8-entry magnitude table, exponent-shifted in place.
  const uint32_t tbl_lo =
      (exp_offset * 0x08080800u) + ((0x03020100u << 2) - 0x00000400u);
  const uint32_t tbl_hi = (exp_offset * 0x08080808u) + (0x07060504u << 2);

  // Mask to 3 bits: drops the sign AND keeps prmt in byte-select mode.
  const uint32_t sel = qb & 0x77777777u;
  const uint32_t mag[2] = {prmt(tbl_lo, tbl_hi, sel),
                           prmt(tbl_lo, tbl_hi, sel >> 16)};

  // Signs re-attached in one op each; the two shifts pick the alternating
  // sign bits the permuted layout placed for this purpose.
  out[0] = lop3_and_or(qb << 4, 0x80808080u, mag[0]);
  out[1] = lop3_and_or(qb, 0x80808080u, mag[1]);
}

// --------------------------------------------------------------- block scales

// MXFP4's scale is ue8m0: an exponent BYTE with no sign and no mantissa.  It
// is already a bf16 exponent field, so the conversion is a shift -- no
// multiply, no table, and the product with the weight is exact because the
// scale is a power of two.  This is the whole reason MX picked e8m0.
// (2^-127 would flush to zero here; real models do not reach it.)
__device__ __forceinline__ void dequant_ue8m0_to_bf16x2(int q, __nv_bfloat162* out) {
  int o1 = (q & 0xff00ff00) >> 1;
  q <<= 8;
  int o2 = (q & 0xff00ff00) >> 1;
  out[1] = *reinterpret_cast<const __nv_bfloat162*>(&o1);
  out[0] = *reinterpret_cast<const __nv_bfloat162*>(&o2);
}

// NVFP4's scale is e4m3, which has a mantissa, so it is a field-shift like the
// FP4 unpack: sign down one, exponent+mantissa right by (8 - 4).  A real
// multiply is then unavoidable, and the format pays for it with finer scales.
__device__ __forceinline__ void dequant_e4m3_to_bf16x2(int q, __nv_bfloat162* out) {
  constexpr int kShift = 8 - 4;
  constexpr int MASK = 0x7f007f00;
  int o1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> kShift);
  q <<= 8;
  int o2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> kShift);
  out[1] = *reinterpret_cast<const __nv_bfloat162*>(&o1);
  out[0] = *reinterpret_cast<const __nv_bfloat162*>(&o2);
}

}  // namespace tmpl
