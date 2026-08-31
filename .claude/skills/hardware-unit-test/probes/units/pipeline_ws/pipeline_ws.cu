// pipeline_ws -- the hut ABI over the warp-specialised mainloop.
//
// Four bespoke entry points collapse into the uniform symbols. This unit builds
// its own descriptors through cute::make_tma_copy (one stage of the staged smem
// layout IS the TMA box), so it does NOT export hut_encode_tensor_map: there is
// no descriptor for a caller to enumerate, and exporting a stub would advertise
// a capability the unit does not have.

#include "pipeline_ws.cuh"

namespace pw = hut::pipeline_ws;

extern "C" {

const char* hut_name() { return "pipeline_ws"; }

uint32_t hut_flags() {
  // NEEDS_COLD: the walk is sized to put its footprint clear of L2, so the
  // measurement is only what it claims with the cache flushed between
  // iterations. REPORTS_SM because per-SM claims must read placement.
  return HUT_NEEDS_COLD | HUT_REPORTS_SM;
}

int32_t hut_cfg_count() {
  int32_t n = 0;
#define X(IDX, N, BK, S) ++n;
  WS_CONFIGS
#undef X
  return n;
}

int32_t hut_cfg(int32_t cfg, int32_t field) {
  switch (cfg) {
#define X(IDX, N, BK, S) \
  case IDX: return field == 0 ? (N) : (field == 1 ? (BK) : (S));
    WS_CONFIGS
#undef X
    default: return HUT_ERR_BAD_CFG;
  }
}

const char* hut_cfg_name(int32_t field) {
  switch (field) {
    case 0: return "N";        // wgmma tile N
    case 1: return "BK";       // k-tile width, fixed at 64 by the swizzle atom
    case 2: return "stages";   // PipelineTmaAsync stage count
    default: return nullptr;
  }
}

const char* hut_opt_name(int32_t) { return nullptr; }

int32_t hut_smem(const HutParams* p) {
  switch (p->cfg) {
#define X(IDX, N, BK, S) \
  case IDX: return static_cast<int32_t>(sizeof(pw::Storage<N, BK, S>));
    WS_CONFIGS
#undef X
    default: return HUT_ERR_BAD_CFG;
  }
}

int32_t hut_launch(const HutParams* p, const HutBuffers* b, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  if (p->k_tile_count < 1) return HUT_ERR_BAD_PARAM;
  switch (p->cfg) {
#define X(IDX, N, BK, S)                                                     \
  case IDX:                                                                  \
    return pw::launch_cfg<N, BK, S>(                                         \
        p->operand_a, p->operand_b, p->n_ctas, p->k_tile_count, p->mode,     \
        b->cycles_a, b->cycles_b, b->sink, b->dbg, s);
    WS_CONFIGS
#undef X
    default: return HUT_ERR_BAD_CFG;
  }
}

int32_t hut_check(const HutParams*, const HutBuffers*, void*) {
  // The instructions this unit issues are gated where they are defined: the
  // wgmma by mma_rate's torch comparison, the TMA by tma_ring's. What is new
  // here is the SYNCHRONISATION, and a wrong barrier hand-off hangs rather than
  // returning wrong bytes -- which the watchdog catches by trap site.
  return HUT_ERR_NO_CHECK;
}

}  // extern "C"
