#pragma once

namespace flash_vla::pi05::sm90::ffn {

// CuTe-facing names for the two GEMM contracts.  Keeping the atom selection
// in one place makes the N=64 GatedUp and N=32 DownResidual choices visible at the call site.
template <typename Atom>
struct GemmTraits {
  using AtomType = Atom;
  using TiledMma = decltype(cute::make_tiled_mma(Atom{}));
};

using DownResidualGemm = GemmTraits<
    cute::SM90_64x32x16_F32BF16BF16_SS<
        cute::GMMA::Major::K, cute::GMMA::Major::MN>>;

using GatedUpGemm = GemmTraits<
    cute::SM90_64x64x16_F32BF16BF16_SS<
        cute::GMMA::Major::MN, cute::GMMA::Major::MN>>;

}  // namespace flash_vla::pi05::sm90::ffn
