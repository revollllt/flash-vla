// Device parity cases for the SM90 tile primitives (tile/sm90/*.cuh).
//
// Each case is one CTA computing one C = A * B^T (or A * B for an MN-major
// B) tile through the library only: g2s (TMA or cp.async) -> smem tile ->
// [s2r] -> gemm() -> r2s -> s2g (TMA store or bulk store).  The host side
// (tests/tile_sm90/__main__.py) owns the reference and the tolerances; this file
// owns nothing but the wiring, so a failure names a primitive.
//
// Build: see tests/tile_sm90/__main__.py (nvcc -arch=sm_90a, -I tile root, -lcuda).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#include "tile/sm90/sm90.cuh"
#include "tile/sm90/tma_host.cuh"

// ---- compile-time table check: every wgmma entry names a CuTe atom with
// traits, and the selector reproduces the atoms the production kernels use.
namespace tile_sm90_table_check {

namespace tile = flash_vla::sm90;
using tile::BF16; using tile::F16; using tile::E4M3; using tile::E5M2;
using tile::Operand; using tile::Major;

template <class Atom>
constexpr bool has_traits_v = (sizeof(cute::MMA_Traits<Atom>) > 0);

template <int N>
struct CheckN {
  static_assert(has_traits_v<typename tile::WgmmaAtom<BF16, BF16, N, Operand::kSmem, Major::K, Major::K>::type>);
  static_assert(has_traits_v<typename tile::WgmmaAtom<BF16, BF16, N, Operand::kSmem, Major::MN, Major::MN>::type>);
  static_assert(has_traits_v<typename tile::WgmmaAtom<BF16, BF16, N, Operand::kReg, Major::K, Major::MN>::type>);
  static_assert(has_traits_v<typename tile::WgmmaAtom<F16, F16, N, Operand::kSmem, Major::K, Major::K>::type>);
  static_assert(has_traits_v<typename tile::WgmmaAtom<F16, F16, N, Operand::kReg, Major::K, Major::K>::type>);
  static_assert(has_traits_v<typename tile::WgmmaAtom<E4M3, E4M3, N, Operand::kSmem, Major::K, Major::K>::type>);
  static_assert(has_traits_v<typename tile::WgmmaAtom<E4M3, E4M3, N, Operand::kReg, Major::K, Major::K>::type>);
  static_assert(has_traits_v<typename tile::WgmmaAtom<E4M3, E5M2, N, Operand::kSmem, Major::K, Major::K>::type>);
  static_assert(has_traits_v<typename tile::WgmmaAtom<E5M2, E4M3, N, Operand::kReg, Major::K, Major::K>::type>);
  static_assert(has_traits_v<typename tile::WgmmaAtom<E5M2, E5M2, N, Operand::kSmem, Major::K, Major::K>::type>);
  static constexpr bool ok = true;
};
template <int... Ns>
constexpr bool check_all(std::integer_sequence<int, Ns...>) {
  return (CheckN<(Ns + 1) * 8>::ok && ...);
}
static_assert(check_all(std::make_integer_sequence<int, 32>{}));

using DownResidual = tile::MmaSelector<BF16, BF16, 64, 32, 128, Operand::kSmem, Operand::kSmem, Major::K, Major::MN>;
static_assert(std::is_same_v<DownResidual::TiledMma,
    decltype(cute::make_tiled_mma(cute::SM90_64x32x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::MN>{}))>);
using Proj = tile::MmaSelector<BF16, BF16, 64, 64, 64>;
static_assert(std::is_same_v<Proj::TiledMma,
    decltype(cute::make_tiled_mma(cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>{}))>);
static_assert(tile::MmaSelector<BF16, BF16, 64, 512, 64>::kAtomN == 256);
static_assert(tile::MmaSelector<BF16, BF16, 64, 320, 64>::kAtomN == 160);
static_assert(tile::MmaSelector<E4M3, E4M3, 64, 64, 64>::kAtomK == 32);
using Sync = tile::MmaSelector<BF16, BF16, 64, 32, 32, Operand::kReg, Operand::kReg, Major::K, Major::K, 128>;
static_assert(Sync::kUseMmaSync && Sync::kWarpsM == 4 && !tile::is_wgmma_v<Sync::TiledMma>);
static_assert(tile::swizzle_bytes_for<BF16, 64>() == 128 && tile::swizzle_bytes_for<BF16, 16>() == 32);
static_assert(tile::TmaTile2D<BF16, 128, 64>::kBoxes == 2 && tile::TmaTile2D<BF16, 128, 64>::kBytes == 16384);

}  // namespace tile_sm90_table_check

// A named namespace: nvcc mis-generates the host stub of a __grid_constant__
// kernel template that lives in an anonymous namespace.
namespace tile_sm90_test {

namespace tile = flash_vla::sm90;
using namespace cute;

enum class Load { kTma, kCpAsync };
enum class Out { kBf16Tma, kBf16Bulk, kF32Bulk };

// Byte map of the dynamic shared allocation (after 1024-byte alignment).
constexpr int kOffA = 0;
constexpr int kOffB = 16384;
constexpr int kOffC = 32768;
constexpr int kOffBar = 65536;
constexpr int kSmemBytes = kOffBar + 64 + 1024;  // + alignment slack

template <class TA_, class TB_, int M_, int N_, int K_, tile::Operand LocA_,
          tile::Operand LocB_, tile::Major MajB_, int Threads_, Load Load_,
          Out Out_>
struct Cfg {
  using TA = TA_;
  using TB = TB_;
  static constexpr int M = M_, N = N_, K = K_, kThreads = Threads_;
  static constexpr tile::Operand LocA = LocA_, LocB = LocB_;
  static constexpr tile::Major MajB = MajB_;
  static constexpr Load kLoad = Load_;
  static constexpr Out kOut = Out_;

  static constexpr int kSwA = tile::swizzle_bytes_for<TA, K>();
  static constexpr int kSwB = MajB == tile::Major::K
                                  ? tile::swizzle_bytes_for<TB, K>()
                                  : tile::swizzle_bytes_for<TB, N>();
  using SmemA = typename tile::SmemTileLayout<TA, M, K, tile::Major::K, kSwA>::type;
  using SmemB = typename tile::SmemTileLayout<TB, N, K, MajB, kSwB>::type;
  // TMA tiles: (inner, outer) as stored in global memory.
  using TileA = tile::TmaTile2D<TA, K, M, kSwA>;
  using TileB = cute::conditional_t<MajB == tile::Major::K,
                                    tile::TmaTile2D<TB, K, N, kSwB>,
                                    tile::TmaTile2D<TB, N, K, kSwB>>;
  static constexpr int kSwC = tile::swizzle_bytes_for<tile::BF16, N>();
  using SmemC16 = typename tile::SmemTileLayout<tile::BF16, M, N, tile::Major::K, kSwC>::type;
  using Sel = tile::MmaSelector<TA, TB, M, N, K, LocA, LocB, tile::Major::K,
                                MajB, Threads_>;
  static_assert(M * K * sizeof(TA) <= kOffB - kOffA, "A tile fits");
  static_assert(N * K * sizeof(TB) <= kOffC - kOffB, "B tile fits");
  static_assert(M * N * 4 <= kOffBar - kOffC, "C tile fits");
};

__device__ __forceinline__ unsigned char* align_1024(unsigned char* p) {
  const uintptr_t v = reinterpret_cast<uintptr_t>(p);
  return reinterpret_cast<unsigned char*>((v + 1023) & ~uintptr_t(1023));
}

template <class C>
__global__ void __launch_bounds__(C::kThreads) case_kernel(
    const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ CUtensorMap tmC, const typename C::TA* __restrict__ gA,
    const typename C::TB* __restrict__ gB, void* __restrict__ gC) {
  using TA = typename C::TA;
  using TB = typename C::TB;
  extern __shared__ unsigned char raw_smem[];
  unsigned char* base = align_1024(raw_smem);
  auto* sA = reinterpret_cast<TA*>(base + kOffA);
  auto* sB = reinterpret_cast<TB*>(base + kOffB);
  unsigned char* sC = base + kOffC;
  auto* bar = reinterpret_cast<uint64_t*>(base + kOffBar);
  auto* full = reinterpret_cast<tile::FullBarrier*>(bar);
  const int tid = static_cast<int>(threadIdx.x);

  Tensor sAt = make_tensor(make_smem_ptr(sA), typename C::SmemA{});
  Tensor sBt = make_tensor(make_smem_ptr(sB), typename C::SmemB{});

  // ---- g2s
  if constexpr (C::kLoad == Load::kTma) {
    if (tid == 0) {
      full->init(1);
      tile::fence_barrier_init();
    }
    __syncthreads();
    if (tile::warp_id() == 0 && tile::elect_one()) {
      full->arrive_and_expect_tx(C::TileA::kBytes + C::TileB::kBytes);
      C::TileA::load(&tmA, sA, 0, 0, bar);
      C::TileB::load(&tmB, sB, 0, 0, bar);
    }
    full->wait(0);
  } else {
    Tensor gAt = make_tensor(make_gmem_ptr(gA), Shape<Int<C::M>, Int<C::K>>{},
                             Stride<Int<C::K>, _1>{});
    Tensor gBt = make_tensor(make_gmem_ptr(gB), Shape<Int<C::N>, Int<C::K>>{},
                             Stride<Int<C::K>, _1>{});
    auto cpA = tile::make_g2s_cp_async_copy<TA, C::kThreads, C::K>();
    auto cpB = tile::make_g2s_cp_async_copy<TB, C::kThreads, C::K>();
    tile::copy_g2s(cpA, tid, gAt, sAt);
    tile::copy_g2s(cpB, tid, gBt, sBt);
    tile::cp_async_commit_group();
    tile::cp_async_wait_group<0>();
    __syncthreads();
  }

  // ---- mma
  auto mma = C::Sel::make();
  Tensor acc = partition_fragment_C(mma, Shape<Int<C::M>, Int<C::N>>{});
  if constexpr (C::Sel::kUseWgmma) {
    if constexpr (C::LocA == tile::Operand::kSmem) {
      tile::gemm_ss<true, 0>(mma, tid, sAt, sBt, acc);
    } else {
      auto thr = mma.get_thread_slice(tid);
      Tensor rA = thr.partition_fragment_A(sAt);
      auto s2r = tile::make_s2r_copy_A<TA>(mma);
      tile::copy_s2r(s2r, tid, sAt, rA);
      tile::gemm_rs<true, 0>(mma, tid, rA, sBt, acc);
    }
  } else {
    auto thr = mma.get_thread_slice(tid);
    Tensor rA = thr.partition_fragment_A(sAt);
    Tensor rB = thr.partition_fragment_B(sBt);
    auto cA = tile::make_s2r_copy_A<TA>(mma);
    auto cB = tile::make_s2r_copy_B<TB>(mma);
    tile::copy_s2r(cA, tid, sAt, rA);
    tile::copy_s2r(cB, tid, sBt, rB);
    tile::gemm<true>(mma, rA, rB, acc);
  }

  // ---- r2s + s2g
  if constexpr (C::kOut == Out::kBf16Tma) {
    Tensor sCt = make_tensor(make_smem_ptr(reinterpret_cast<tile::BF16*>(sC)),
                             typename C::SmemC16{});
    auto r2s = tile::make_r2s_copy_C<tile::BF16>(mma);
    tile::copy_r2s(r2s, tid, tile::convert_fragment<tile::BF16>(acc), sCt);
    tile::fence_proxy_async_shared();
    __syncthreads();
    if (tid == 0) {
      tile::tma_store_2d(&tmC, sC, 0, 0);
      tile::store_commit_group();
      tile::store_wait_group<0>();
    }
  } else if constexpr (C::kOut == Out::kBf16Bulk) {
    Tensor sCt = make_tensor(make_smem_ptr(reinterpret_cast<tile::BF16*>(sC)),
                             Layout<Shape<Int<C::M>, Int<C::N>>, Stride<Int<C::N>, _1>>{});
    auto r2s = tile::make_r2s_copy_C<tile::BF16>(mma);
    tile::copy_r2s(r2s, tid, tile::convert_fragment<tile::BF16>(acc), sCt);
    tile::fence_proxy_async_shared();
    __syncthreads();
    if (tid == 0) {
      tile::bulk_store_1d(gC, sC, C::M * C::N * sizeof(tile::BF16));
      tile::store_commit_group();
      tile::store_wait_group<0>();
    }
  } else {
    Tensor sCt = make_tensor(make_smem_ptr(reinterpret_cast<float*>(sC)),
                             Layout<Shape<Int<C::M>, Int<C::N>>, Stride<Int<C::N>, _1>>{});
    auto r2s = tile::make_r2s_copy_C_scalar<float>(mma);
    tile::copy_r2s(r2s, tid, acc, sCt);
    tile::fence_proxy_async_shared();
    __syncthreads();
    if (tid == 0) {
      tile::bulk_store_1d(gC, sC, C::M * C::N * sizeof(float));
      tile::store_commit_group();
      tile::store_wait_group<0>();
    }
  }
}

template <class C>
int run_case(const void* A, const void* B, void* Cout) {
  using TA = typename C::TA;
  using TB = typename C::TB;
  CUtensorMap tmA{}, tmB{}, tmC{};
  if constexpr (C::kLoad == Load::kTma) {
    CUresult ra = tile::encode_tensor_map_2d<TA>(
        &tmA, A, C::K, C::M, C::K * sizeof(TA), C::TileA::kInnerAtom, C::M, C::kSwA);
    CUresult rb;
    if constexpr (C::MajB == tile::Major::K) {
      rb = tile::encode_tensor_map_2d<TB>(
          &tmB, B, C::K, C::N, C::K * sizeof(TB), C::TileB::kInnerAtom, C::N, C::kSwB);
    } else {
      rb = tile::encode_tensor_map_2d<TB>(
          &tmB, B, C::N, C::K, C::N * sizeof(TB), C::TileB::kInnerAtom, C::K, C::kSwB);
    }
    if (ra != CUDA_SUCCESS) return 1000 + static_cast<int>(ra);
    if (rb != CUDA_SUCCESS) return 2000 + static_cast<int>(rb);
  }
  if constexpr (C::kOut == Out::kBf16Tma) {
    CUresult rc = tile::encode_tensor_map_2d<tile::BF16>(
        &tmC, Cout, C::N, C::M, C::N * sizeof(tile::BF16),
        C::kSwC / static_cast<int>(sizeof(tile::BF16)), C::M, C::kSwC);
    if (rc != CUDA_SUCCESS) return 3000 + static_cast<int>(rc);
  }
  cudaError_t e = cudaFuncSetAttribute(
      case_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (e != cudaSuccess) return 4000 + static_cast<int>(e);
  case_kernel<C><<<1, C::kThreads, kSmemBytes>>>(
      tmA, tmB, tmC, static_cast<const TA*>(A), static_cast<const TB*>(B), Cout);
  e = cudaGetLastError();
  if (e != cudaSuccess) return 5000 + static_cast<int>(e);
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) return 6000 + static_cast<int>(e);
  return 0;
}

using tile::BF16;
using tile::E4M3;
using tile::E5M2;
constexpr auto kS = tile::Operand::kSmem;
constexpr auto kR = tile::Operand::kReg;
constexpr auto kMajK = tile::Major::K;
constexpr auto kMajMN = tile::Major::MN;

// One warpgroup, SS wgmma, both K-major, bf16 out through stmatrix + TMA store.
using WgmmaSsBf16KK = Cfg<BF16, BF16, 64, 64, 64, kS, kS, kMajK, 128, Load::kTma, Out::kBf16Tma>;
// SS wgmma with an MN-major B (the FFN DownResidual shape), f32 out.
using WgmmaSsBf16KMN = Cfg<BF16, BF16, 64, 32, 128, kS, kS, kMajMN, 128, Load::kTma, Out::kF32Bulk>;
// RS wgmma: A through ldmatrix into registers, N = 128.
using WgmmaRsBf16 = Cfg<BF16, BF16, 64, 128, 64, kR, kS, kMajK, 128, Load::kTma, Out::kF32Bulk>;
// SS wgmma on fp8 (e4m3 x e4m3, TN), f32 out.
using WgmmaSsE4M3 = Cfg<E4M3, E4M3, 64, 64, 64, kS, kS, kMajK, 128, Load::kTma, Out::kF32Bulk>;
// SS wgmma on mixed fp8 (e4m3 x e5m2).
using WgmmaSsE4M3E5M2 = Cfg<E4M3, E5M2, 64, 64, 64, kS, kS, kMajK, 128, Load::kTma, Out::kF32Bulk>;
// Four warps of mma.sync bf16, cp.async g2s, ldmatrix both operands, bf16 out via stmatrix.
using MmaSyncBf16 = Cfg<BF16, BF16, 64, 32, 32, kR, kR, kMajK, 128, Load::kCpAsync, Out::kBf16Bulk>;
// Four warps of mma.sync fp8 (m16n8k32), f32 out.
using MmaSyncE4M3 = Cfg<E4M3, E4M3, 64, 32, 64, kR, kR, kMajK, 128, Load::kCpAsync, Out::kF32Bulk>;
// Two warpgroups stacked along M (M = 128), SS wgmma, f32 out.
using WgmmaSsBf16Wg2 = Cfg<BF16, BF16, 128, 64, 64, kS, kS, kMajK, 256, Load::kTma, Out::kF32Bulk>;

}  // namespace tile_sm90_test

using namespace tile_sm90_test;

extern "C" {

// Returns 0 on success; 1000-6999 encode the failing stage (see run_case).
int tile_sm90_run(const char* name, const void* A, const void* B, void* C) {
  if (!std::strcmp(name, "wgmma_ss_bf16_kk")) return run_case<WgmmaSsBf16KK>(A, B, C);
  if (!std::strcmp(name, "wgmma_ss_bf16_kmn")) return run_case<WgmmaSsBf16KMN>(A, B, C);
  if (!std::strcmp(name, "wgmma_rs_bf16")) return run_case<WgmmaRsBf16>(A, B, C);
  if (!std::strcmp(name, "wgmma_ss_e4m3")) return run_case<WgmmaSsE4M3>(A, B, C);
  if (!std::strcmp(name, "wgmma_ss_e4m3_e5m2")) return run_case<WgmmaSsE4M3E5M2>(A, B, C);
  if (!std::strcmp(name, "mma_sync_bf16")) return run_case<MmaSyncBf16>(A, B, C);
  if (!std::strcmp(name, "mma_sync_e4m3")) return run_case<MmaSyncE4M3>(A, B, C);
  if (!std::strcmp(name, "wgmma_ss_bf16_wg2")) return run_case<WgmmaSsBf16Wg2>(A, B, C);
  return -1;
}

}  // extern "C"
