#pragma once

// Umbrella for the device-side SM90 tile primitives.  Host tensor-map
// helpers live in tma_host.cuh and are included separately by host code.

#include "tile/sm90/common.cuh"
#include "tile/sm90/barrier.cuh"
#include "tile/sm90/smem_layout.cuh"
#include "tile/sm90/copy_g2s.cuh"
#include "tile/sm90/copy_s2r.cuh"
#include "tile/sm90/copy_r2s.cuh"
#include "tile/sm90/copy_s2g.cuh"
#include "tile/sm90/mma_sync.cuh"
#include "tile/sm90/wgmma.cuh"
#include "tile/sm90/gemm.cuh"
