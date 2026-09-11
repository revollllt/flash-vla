// overlap -- the hut ABI over the contention probe.
//
// Four bespoke entry points collapse into the uniform symbols. The two clock
// spans this unit produces map onto cycles_a / cycles_b, which is what those
// two buffers exist for: the copy engine's span and the tensor core's span are
// the two roles of one launch.

#include "overlap.cuh"

namespace ov = hut::overlap;

// mode selects which engine runs. The three share one kernel, one shared-memory
// layout and one descriptor, so an isolated row and a contended row differ ONLY
// in this field. [protocol.md rule 3]
enum : int32_t {
  MODE_BOTH       = 0,
  MODE_TMA_ONLY   = 1,
  MODE_WGMMA_ONLY = 2,
};

extern "C" {

const char* hut_name() { return "overlap"; }

uint32_t hut_flags() {
  // No HUT_HAS_CHECK: this unit issues no instruction the tma and mma units do
  // not already gate. What it measures is the RATIO between two of its own
  // modes, and both members are gated where they are defined.
  return HUT_REPORTS_SM;
}

int32_t hut_cfg_count() {
  int32_t n = 0;
#define X(IDX, N, NG, W) ++n;
  OVERLAP_CONFIGS
#undef X
  return n;
}

int32_t hut_cfg(int32_t cfg, int32_t field) {
  switch (cfg) {
#define X(IDX, N, NG, W) \
  case IDX: return field == 0 ? (N) : (field == 1 ? (NG) : (W));
    OVERLAP_CONFIGS
#undef X
    default: return HUT_ERR_BAD_CFG;
  }
}

const char* hut_cfg_name(int32_t field) {
  switch (field) {
    case 0: return "N";
    case 1: return "n_groups";
    case 2: return "wait";
    default: return nullptr;
  }
}

const char* hut_opt_name(int32_t i) {
  switch (i) {
    case 0: return "k_tiles_mma";   // the consumer's own k-tile count
    default: return nullptr;
  }
}

int32_t hut_smem(const HutParams* p) {
  const int32_t n = hut_cfg(p->cfg, 0);
  if (n < 0) return n;
  const size_t bytes =
      ov::ring_bytes(p->num_producers, p->stages, p->box_bytes)
      + sizeof(hut::bf16) * static_cast<size_t>(ov::M_TILE + n) * ov::K_TILE
      + static_cast<size_t>(p->num_producers) * p->stages
            * sizeof(hut::TransactionBarrier);
  return static_cast<int32_t>(bytes);
}

int32_t hut_launch(const HutParams* p, const HutBuffers* b, void* stream) {
  if (p->num_producers > ov::kMaxProducers || p->stages > ov::kMaxStages ||
      p->num_producers < 0)
    return HUT_ERR_BAD_PARAM;
  auto s = static_cast<cudaStream_t>(stream);
  const int32_t smem = hut_smem(p);
  if (smem < 0) return smem;
  if (smem > 232448) return HUT_ERR_SMEM;

  // The launch is always the FULL thread count, whatever num_producers is: the
  // producer warps that are not used still exist, so the consumer's occupancy
  // and register budget do not move between modes.
  constexpr int32_t nthreads = hut::kWarpgroupThreads + ov::kMaxProducers * 32;

  switch (p->cfg) {
#define X(IDX, N, NG, W)                                                     \
  case IDX: {                                                                \
    auto k = ov::overlap_kernel<N, NG, W, nthreads>;                         \
    cudaError_t e = cudaFuncSetAttribute(                                    \
        k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);               \
    if (e != cudaSuccess) return static_cast<int32_t>(e);                    \
    k<<<p->n_ctas, nthreads, smem, s>>>(                                     \
        *static_cast<const CUtensorMap*>(p->tensor_map), p->stages,          \
        p->box_bytes, p->k_tile_count, p->opt[0], p->num_producers,          \
        p->mode, p->mask0, p->shift0, p->step0, p->mask1, p->step1,          \
        static_cast<long long*>(b->cycles_a),                                \
        static_cast<long long*>(b->cycles_b),                                \
        static_cast<float*>(b->sink),                                        \
        static_cast<long long*>(b->dbg));                                    \
    break;                                                                   \
  }
    OVERLAP_CONFIGS
#undef X
    default: return HUT_ERR_BAD_CFG;
  }
  return static_cast<int32_t>(cudaGetLastError());
}

int32_t hut_check(const HutParams*, const HutBuffers*, void*) {
  return HUT_ERR_NO_CHECK;
}

// This unit drives TMA, so it exports the descriptor encoder. Enumerating
// legality beats assuming it. [hut/tma.cuh]
int32_t hut_encode_tensor_map(void* out, const void* global_address,
                              uint64_t global_dim_0, uint64_t global_dim_1,
                              uint32_t box_dim_0, uint32_t box_dim_1,
                              int32_t swizzle, int32_t tensor_data_type,
                              int32_t elem_bytes) {
  return hut::encode_tensor_map(static_cast<CUtensorMap*>(out), global_address,
                                global_dim_0, global_dim_1, box_dim_0,
                                box_dim_1, swizzle, tensor_data_type,
                                elem_bytes);
}

int32_t hut_tensor_map_bytes() { return static_cast<int32_t>(sizeof(CUtensorMap)); }

}  // extern "C"
