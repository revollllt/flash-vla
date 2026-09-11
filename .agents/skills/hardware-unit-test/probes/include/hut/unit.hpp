// hut/unit.hpp -- the ABI every unit exports, and the only one.
//
// Thirteen bespoke `extern "C"` entry points in five naming conventions, each
// with a hand-written ctypes argtypes list, broke three times in one session --
// every time by appending a trailing pointer. Parameters travel in a POD
// instead, mirrored once on the Python side, so adding an axis is a struct
// change rather than thirteen edits.
//
// One unit per shared object, so the per-unit build cache and compile isolation
// survive: mma_rate's template instantiations alone take two minutes.

#pragma once

#include <cstdint>

extern "C" {

// What a unit declares about itself. The harness reads these and refuses to
// record a constant from a unit whose regime it cannot verify.
enum HutFlags : uint32_t {
  // Rates are meaningless unless the source is genuinely cold; the harness
  // rejects a row whose measured L2 ratio is below COLD_MIN_L2_RATIO.
  // [protocol.md rule 6b]
  HUT_NEEDS_COLD = 1u << 0,
  // hut_check() compares against a host reference and must pass before any rate
  // is read. [rule 11]
  HUT_HAS_CHECK  = 1u << 1,
  // The unit reports CTA placement in HutBuffers::sm_id.
  HUT_REPORTS_SM = 1u << 2,
  // The unit is compute-only: no memory source, so NEEDS_COLD does not apply.
  HUT_NO_SOURCE  = 1u << 3,
};

// Everything a launch needs. Named fields carry the authority's word for the
// quantity (references/vocabulary.md); `opt` is the documented escape hatch for
// a unit's own axes, described by hut_opt_name().
struct HutParams {
  int32_t cfg;              // index into the unit's config table
  int32_t mode;             // which side of the unit runs; 0 is the default
  int32_t n_ctas;
  int32_t n_threads;
  int32_t num_producers;    // CUTLASS PipelineTmaAsync::Params::num_producers
  int32_t num_consumers;    // ditto num_consumers
  int32_t stages;           // CUTLASS Stages -- the smem ring, never "depth"
  int32_t k_tile_count;     // CUTLASS k_tile_count -- never "trip"
  int32_t box_bytes;        // one TMA box: boxDim[0] * boxDim[1] * elem_bytes
  int32_t txn_bytes;        // bytes on ONE barrier: CUTLASS transaction_bytes
  int32_t stage_bytes;      // bytes of one smem stage
  // Coordinate walk. The COORDINATES are a machine quantity and take the PTX
  // operand order; the shift-and-mask encoding is ours, and is ours to avoid an
  // integer divide on the issue path.
  int32_t mask0, shift0, step0, mask1, step1;
  int32_t opt[4];           // unit-specific axes; see hut_opt_name()
  const void* tensor_map;   // CUtensorMap, or nullptr
  const void* operand_a;
  const void* operand_b;
};

// Where a launch writes. All optional: a unit ignores what it does not produce.
struct HutBuffers {
  void* cycles_a;   // per-CTA clock64 span, first role
  void* cycles_b;   // per-CTA clock64 span, second role
  void* sink;       // keeps results live against dead-code elimination
  void* dbg;        // watchdog trap sites, 2 int64 per CTA
  void* sm_id;      // CTA placement, int32 per CTA
  void* out;        // hut_check output, for host comparison
};

const char* hut_name();
uint32_t    hut_flags();
int32_t     hut_cfg_count();
int32_t     hut_cfg(int32_t cfg, int32_t field);
const char* hut_cfg_name(int32_t field);   // nullptr past the last field
const char* hut_opt_name(int32_t i);       // nullptr if opt[i] is unused
int32_t     hut_smem(const HutParams* p);

// OPTIONAL, present only in units that drive TMA. Encodes a 2-D tiled tensor
// map into `out` (which must be 64-byte aligned) and returns the CUresult
// unchanged, so a caller can ENUMERATE descriptor legality rather than assume
// it. Units without TMA do not export it; the Python binding is lazy.
int32_t     hut_encode_tensor_map(void* out, const void* global_address,
                                  uint64_t global_dim_0, uint64_t global_dim_1,
                                  uint32_t box_dim_0, uint32_t box_dim_1,
                                  int32_t swizzle, int32_t tensor_data_type,
                                  int32_t elem_bytes);
int32_t     hut_tensor_map_bytes();
int32_t     hut_check(const HutParams* p, const HutBuffers* b, void* stream);
int32_t     hut_launch(const HutParams* p, const HutBuffers* b, void* stream);

}  // extern "C"

// Error codes. Negative is a unit-level rejection; a positive value is a
// cudaError_t passed through unchanged.
enum : int32_t {
  HUT_ERR_BAD_CFG   = -1,
  HUT_ERR_BAD_PARAM = -2,
  HUT_ERR_SMEM      = -3,
  HUT_ERR_NO_CHECK  = -4,   // HUT_HAS_CHECK not declared
};
