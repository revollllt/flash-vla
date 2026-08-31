// tma_ring.cu -- the translation unit: ABI surface only.

#include "tma_ring.cuh"

namespace tr = hut::tma_ring;

// The config table is the BOX GEOMETRY, because that is the axis a descriptor
// fixes and a sweep cannot vary at runtime. Everything else -- CTAs, warps,
// stages, box bytes, the walk -- rides in HutParams.
//   0 contig    box_dim[0] == global_dim[0], so box rows are ADJACENT and the
//               whole box is one contiguous run (the pre-blocked weight case)
//   1 stride2k  128 B strips at a 2 KB stride (the x_pad case)
//   2 stride8k  128 B strips at an 8 KB stride (the hidden case)
static const int32_t kGeomGlobalDim0[3] = {64, 1024, 4096};

extern "C" {

const char* hut_name() { return "tma_ring"; }

uint32_t hut_flags() {
  // Rates are DRAM-sourced by default and meaningless if the walk never leaves
  // L2 -- that mistake invalidated a whole constant. The gate is mandatory here
  // because a wrong descriptor delivers wrong bytes at the right speed.
  return HUT_NEEDS_COLD | HUT_HAS_CHECK | HUT_REPORTS_SM;
}

int32_t hut_cfg_count() { return 3; }

int32_t hut_cfg(int32_t cfg, int32_t field) {
  if (cfg < 0 || cfg >= 3) return HUT_ERR_BAD_CFG;
  return field == 0 ? kGeomGlobalDim0[cfg] : HUT_ERR_BAD_CFG;
}

const char* hut_cfg_name(int32_t field) {
  return field == 0 ? "global_dim_0" : nullptr;
}

const char* hut_opt_name(int32_t i) {
  switch (i) {
    case 0: return "n_check_boxes";   // boxes the gate delivers and compares
    default: return nullptr;
  }
}

int32_t hut_smem(const HutParams* p) {
  if (p->mode == 1) return p->box_bytes + static_cast<int32_t>(sizeof(uint64_t));
  return p->num_producers * p->stages * p->box_bytes
       + p->num_producers * p->stages * static_cast<int32_t>(sizeof(uint64_t));
}

static int32_t set_smem(const void* kernel, int32_t bytes) {
  if (bytes > 232448) return HUT_ERR_SMEM;
  cudaError_t e = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  return e == cudaSuccess ? 0 : static_cast<int32_t>(e);
}

int32_t hut_check(const HutParams* p, const HutBuffers* b, void* stream) {
  if (p->box_bytes < 16 || (p->box_bytes & 15) || p->k_tile_count < 1)
    return HUT_ERR_BAD_PARAM;
  const int32_t smem = hut_smem(p);
  int32_t rc = set_smem(reinterpret_cast<const void*>(tr::check_kernel), smem);
  if (rc != 0) return rc;
  tr::check_kernel<<<1, 32, smem, static_cast<cudaStream_t>(stream)>>>(
      *static_cast<const CUtensorMap*>(p->tensor_map), *p,
      static_cast<uint4*>(b->out), static_cast<int32_t*>(b->sm_id),
      static_cast<long long*>(b->dbg));
  return static_cast<int32_t>(cudaGetLastError());
}

int32_t hut_launch(const HutParams* p, const HutBuffers* b, void* stream) {
  if (p->mode == 1) return hut_check(p, b, stream);
  if (p->stages > tr::kMaxStages || p->stages < 1 ||
      p->num_producers > tr::kMaxWarps || p->num_producers < 1)
    return HUT_ERR_BAD_PARAM;
  const int32_t smem = hut_smem(p);
  int32_t rc = set_smem(reinterpret_cast<const void*>(tr::rate_kernel), smem);
  if (rc != 0) return rc;
  tr::rate_kernel<<<p->n_ctas, p->num_producers * 32, smem,
                    static_cast<cudaStream_t>(stream)>>>(
      *static_cast<const CUtensorMap*>(p->tensor_map), *p,
      static_cast<long long*>(b->dbg), static_cast<int32_t*>(b->sm_id));
  return static_cast<int32_t>(cudaGetLastError());
}

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

int32_t hut_tensor_map_bytes() {
  return static_cast<int32_t>(sizeof(CUtensorMap));
}

}  // extern "C"
