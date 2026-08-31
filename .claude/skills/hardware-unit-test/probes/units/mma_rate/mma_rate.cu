// mma_rate -- the hut ABI over three tensor-core instruction families.
//
// Eight bespoke entry points (mma_probe_rate / _check / _cfg / _cfg_count /
// _sync_rate / _sync_check / _ldm_rate / _ldm_check / _ldm_cfg / _ldm_cfg_count)
// collapse into the nine uniform symbols in hut/unit.hpp. The family is chosen
// by `mode`, so a fourth costs a case rather than another extern "C" signature
// the Python side has to mirror by hand.

#include "mma_rate.cuh"

namespace mr = hut::mma_rate;

enum : int32_t {
  MODE_WGMMA    = 0,   // wgmma SS; cfg indexes the (N, n_groups, wait) table
  MODE_MMA_SYNC = 1,   // mma.sync m16n8k16 from registers; opt[0] = n_accum
  MODE_MMA_LDM  = 2,   // mma.sync fed by ldmatrix; cfg indexes the LDM table
};

// A unit has ONE hut_cfg, and this unit has two instantiation tables. The cfg
// space is partitioned rather than the ABI widened: which kernels exist is a
// real constraint (each row is a compiled kernel and its own register
// allocation), so the host must be able to enumerate them, and duplicating the
// tables in Python is exactly what the uniform ABI exists to prevent.
//
//   [0, wgmma rows)                     MODE_WGMMA,    fields 0..2
//   [kLdmCfgBase, + ldm rows)           MODE_MMA_LDM,  fields 3..4
//
// MODE_MMA_SYNC has no table: it takes n_accum in opt[0] and n_threads in the
// launch geometry, and every (1|2|4|8) x (32|64|128|256) pair is instantiated.
constexpr int32_t kLdmCfgBase = 100;

extern "C" {

const char* hut_name() { return "mma_rate"; }

uint32_t hut_flags() {
  // No HUT_NEEDS_COLD: every operand is resident in shared memory or registers
  // before the timed loop starts, so flushing L2 would change the launch cost
  // and nothing that is being measured.
  return HUT_HAS_CHECK | HUT_REPORTS_SM;
}

int32_t hut_cfg_count() {
  int32_t n = 0;
#define X(IDX, N, NG, W) ++n;
  MMA_CONFIGS
#undef X
  return n;
}

int32_t hut_cfg(int32_t cfg, int32_t field) {
  if (cfg >= kLdmCfgBase) {
    switch (cfg - kLdmCfgBase) {
#define Y(IDX, AM, BN) \
  case IDX: return field == 3 ? (AM) : (field == 4 ? (BN) : HUT_ERR_BAD_CFG);
      LDM_CONFIGS
#undef Y
      default: return HUT_ERR_BAD_CFG;
    }
  }
  switch (cfg) {
#define X(IDX, N, NG, W) \
  case IDX: return field == 0 ? (N) : (field == 1 ? (NG) : (W));
    MMA_CONFIGS
#undef X
    default: return HUT_ERR_BAD_CFG;
  }
}

const char* hut_cfg_name(int32_t field) {
  switch (field) {
    case 0: return "N";            // wgmma tile N
    case 1: return "n_groups";     // wgmma issued before one commit
    case 2: return "wait";         // groups left outstanding
    case 3: return "a_tiles_m";    // ldm: A tiles of 16 rows
    case 4: return "b_tiles_n";    // ldm: B tiles of 8 columns
    default: return nullptr;
  }
}

const char* hut_opt_name(int32_t i) {
  switch (i) {
    case 0: return "n_accum";      // independent accumulators, MODE_MMA_SYNC
    case 1: return "check_n";      // wgmma tile N for hut_check
    default: return nullptr;
  }
}

int32_t hut_smem(const HutParams* p) {
  if (p->mode == MODE_WGMMA) {
    const int32_t n = hut_cfg(p->cfg, 0);
    if (n < 0) return n;
    return static_cast<int32_t>(sizeof(hut::bf16)) * (mr::M_TILE + n) * mr::K_TILE;
  }
  if (p->mode == MODE_MMA_LDM) {
    const int32_t am = hut_cfg(p->cfg, 3), bn = hut_cfg(p->cfg, 4);
    if (am < 0 || bn < 0) return HUT_ERR_BAD_CFG;
    return static_cast<int32_t>(sizeof(hut::bf16)) * (am * 16 * 16 + bn * 8 * 16);
  }
  return 0;   // MODE_MMA_SYNC holds both operands in registers
}

int32_t hut_launch(const HutParams* p, const HutBuffers* b, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  const auto* a = static_cast<const hut::bf16*>(p->operand_a);
  const auto* bb = static_cast<const hut::bf16*>(p->operand_b);
  auto* sink = static_cast<float*>(b->sink);
  auto* cyc = static_cast<long long*>(b->cycles_a);
  if (p->k_tile_count < 1) return HUT_ERR_BAD_PARAM;
  const int32_t nt = p->n_threads;

  if (p->mode == MODE_MMA_SYNC) {
#define S(NA, NTH)                                                           \
  if (p->opt[0] == (NA) && nt == (NTH)) {                                    \
    mr::mma_sync_rate_kernel<NA, NTH><<<p->n_ctas, NTH, 0, s>>>(             \
        a, bb, p->k_tile_count, sink, cyc);                                  \
    return static_cast<int32_t>(cudaGetLastError());                         \
  }
    S(1, 32) S(1, 64) S(1, 128) S(1, 256)
    S(2, 32) S(2, 64) S(2, 128) S(2, 256)
    S(4, 32) S(4, 64) S(4, 128) S(4, 256)
    S(8, 32) S(8, 64) S(8, 128) S(8, 256)
#undef S
    return HUT_ERR_BAD_CFG;
  }

  if (p->mode == MODE_MMA_LDM) {
    const int32_t smem = hut_smem(p);
    if (smem < 0) return smem;
    switch (p->cfg - kLdmCfgBase) {
#define Y(IDX, AM, BN)                                                       \
  case IDX: {                                                                \
    if (nt == 128)                                                           \
      mr::mma_sync_ldm_rate_kernel<AM, BN, 128><<<p->n_ctas, 128, smem, s>>>( \
          a, bb, p->k_tile_count, sink, cyc);                                \
    else if (nt == 256)                                                      \
      mr::mma_sync_ldm_rate_kernel<AM, BN, 256><<<p->n_ctas, 256, smem, s>>>( \
          a, bb, p->k_tile_count, sink, cyc);                                \
    else return HUT_ERR_BAD_PARAM;                                           \
    break;                                                                   \
  }
      LDM_CONFIGS
#undef Y
      default: return HUT_ERR_BAD_CFG;
    }
    return static_cast<int32_t>(cudaGetLastError());
  }

  if (p->mode != MODE_WGMMA) return HUT_ERR_BAD_PARAM;
  const int32_t smem = hut_smem(p);
  if (smem < 0) return smem;
  switch (p->cfg) {
#define X(IDX, N, NG, W)                                                     \
  case IDX: {                                                                \
    if (nt == 128)                                                           \
      mr::rate_kernel<N, NG, W, 128><<<p->n_ctas, 128, smem, s>>>(           \
          a, bb, p->k_tile_count, sink, cyc);                                \
    else if (nt == 256)                                                      \
      mr::rate_kernel<N, NG, W, 256><<<p->n_ctas, 256, smem, s>>>(           \
          a, bb, p->k_tile_count, sink, cyc);                                \
    else return HUT_ERR_BAD_PARAM;                                           \
    break;                                                                   \
  }
    MMA_CONFIGS
#undef X
    default: return HUT_ERR_BAD_CFG;
  }
  return static_cast<int32_t>(cudaGetLastError());
}

int32_t hut_check(const HutParams* p, const HutBuffers* b, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  const auto* a = static_cast<const hut::bf16*>(p->operand_a);
  const auto* bb = static_cast<const hut::bf16*>(p->operand_b);
  auto* out = static_cast<float*>(b->out);

  if (p->mode == MODE_MMA_SYNC) {
    mr::mma_sync_check_kernel<<<1, 32, 0, s>>>(a, bb, out);
    return static_cast<int32_t>(cudaGetLastError());
  }
  if (p->mode == MODE_MMA_LDM) {
    const int32_t smem =
        static_cast<int32_t>(sizeof(hut::bf16)) * (16 * 16 + 8 * 16);
    mr::mma_sync_ldm_check_kernel<<<1, 32, smem, s>>>(a, bb, out);
    return static_cast<int32_t>(cudaGetLastError());
  }

  // wgmma: one instruction on real data, D written in row/col order for the
  // host to compare. N comes from opt[1] rather than the rate table, so every
  // instantiated N stays reachable whether or not it is a row of that table.
  const int32_t n = p->opt[1];
  const int32_t smem =
      static_cast<int32_t>(sizeof(hut::bf16)) * (mr::M_TILE + n) * mr::K_TILE;
  switch (n) {
#define C(N)                                                                 \
  case N:                                                                    \
    mr::check_kernel<N><<<1, 128, smem, s>>>(a, bb, out);                    \
    break;
    C(8) C(16) C(32) C(64) C(128) C(256)
#undef C
    default: return HUT_ERR_BAD_CFG;
  }
  return static_cast<int32_t>(cudaGetLastError());
}

}  // extern "C"
