// gmem_atomic.cu -- the translation unit. Kernels live in the .cuh; this is the
// ABI surface and nothing else, so every unit's .cu looks the same.

#include "gmem_atomic.cuh"

namespace ga = hut::gmem_atomic;

// Bytes each instruction updates, so a width sweep reads as bandwidth as well
// as op rate. The two readings disagree exactly when the unit is
// per-transaction, which is what the width axis exists to settle.
// [atom.ratio.width]
static const int32_t kOpBytes[ga::OP_COUNT] = {
    4, 4, 4, 4, 4, 4, 8, 16, 4, 4, 4, 4, 4, 4};

extern "C" {

const char* hut_name() { return "gmem_atomic"; }

// No memory SOURCE: the atomics are the traffic, so there is no walk whose
// coldness could be checked, and NEEDS_COLD would be unsatisfiable rather than
// merely unmet. hut_check is likewise absent -- an atomic's effect is the
// counter value, which the pingpong mode already verifies by construction: a
// missed increment deadlocks the ping-pong instead of returning a wrong rate.
uint32_t hut_flags() { return HUT_NO_SOURCE; }

int32_t hut_cfg_count() { return ga::OP_COUNT; }

int32_t hut_cfg(int32_t cfg, int32_t field) {
  if (cfg < 0 || cfg >= ga::OP_COUNT) return HUT_ERR_BAD_CFG;
  return field == 0 ? kOpBytes[cfg] : HUT_ERR_BAD_CFG;
}

const char* hut_cfg_name(int32_t field) {
  return field == 0 ? "op_bytes" : nullptr;
}

const char* hut_opt_name(int32_t i) {
  switch (i) {
    case 0: return "n_addr";          // addresses shared, power of two
    case 1: return "stride_bytes";    // spacing between them
    case 2: return "rounds";          // pingpong only
    case 3: return "advance_atomic";  // pingpong only: red vs st.release
    default: return nullptr;
  }
}

int32_t hut_smem(const HutParams*) { return 0; }

int32_t hut_check(const HutParams*, const HutBuffers*, void*) {
  return HUT_ERR_NO_CHECK;   // see hut_flags()
}

int32_t hut_launch(const HutParams* p, const HutBuffers* b, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);

  if (p->mode == 1) {
    ga::pingpong_kernel<<<2 + p->n_ctas, 32, 0, s>>>(
        reinterpret_cast<unsigned*>(const_cast<void*>(p->operand_a)),
        p->opt[2], p->opt[3], static_cast<long long*>(b->dbg));
    return static_cast<int32_t>(cudaGetLastError());
  }

  const int32_t n_addr = p->opt[0];
  if (n_addr < 1 || (n_addr & (n_addr - 1))) return HUT_ERR_BAD_PARAM;
  const int32_t mask = n_addr - 1;

#define HUT_LAUNCH_OP(OP)                                                     \
  case ga::OP:                                                                \
    ga::rate_kernel<ga::OP><<<p->n_ctas, p->n_threads, 0, s>>>(               \
        reinterpret_cast<uint8_t*>(const_cast<void*>(p->operand_a)), mask,    \
        p->opt[1], p->k_tile_count, static_cast<unsigned*>(b->sink));         \
    break;
  switch (p->cfg) {
    HUT_LAUNCH_OP(RED_U32)   HUT_LAUNCH_OP(ATOM_U32)
    HUT_LAUNCH_OP(RED_F32)   HUT_LAUNCH_OP(ATOM_F32)
    HUT_LAUNCH_OP(RED_F16X2) HUT_LAUNCH_OP(RED_BF16X2)
    HUT_LAUNCH_OP(RED_V2F32) HUT_LAUNCH_OP(RED_V4F32)
    HUT_LAUNCH_OP(ATOM_CAS)  HUT_LAUNCH_OP(ATOM_EXCH)
    HUT_LAUNCH_OP(RED_U32_CTA) HUT_LAUNCH_OP(RED_U32_SYS)
    HUT_LAUNCH_OP(ATOM_U32_CTA) HUT_LAUNCH_OP(ATOM_U32_SYS)
    default: return HUT_ERR_BAD_CFG;
  }
#undef HUT_LAUNCH_OP
  return static_cast<int32_t>(cudaGetLastError());
}

}  // extern "C"
