#pragma once

// Moved to tile/common/copy_s2g.cuh: nothing in it is Hopper-specific. This shim
// keeps the sm90 include path and name lookup working unchanged; see
// tile/common/README.md.

#include "tile/common/copy_s2g.cuh"

namespace flash_vla::sm90 {
using namespace flash_vla::tile;
}  // namespace flash_vla::sm90
