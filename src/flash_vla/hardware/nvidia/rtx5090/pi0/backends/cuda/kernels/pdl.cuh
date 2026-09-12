// Programmatic dependent launch, for the chain of kernels this route replays.
//
// A dependent grid normally starts when its producer's last CTA exits. With
// PDL it starts when the producer says so, so the consumer's CTAs are
// scheduled and its producer-independent work runs while the producer is still
// finishing. `griddepcontrol` is accepted by ptxas for sm_120, sm_120f and
// sm_120a, and the runtime carries the launch attribute.
//
// The two halves are not symmetric, and the asymmetry decides how each is set
// (see the kernel wiki's `technique-pdl-placement`):
//
//   The WAIT is derived, never swept. It has to sit immediately before the
//   first read of producer data; later is a race, not a slower kernel.
//   Everything producer-independent -- index arithmetic, model parameters, and
//   above all a weight stream -- belongs above it, and that is exactly what
//   PDL overlaps.
//
//   The TRIGGER is swept and cannot be derived. It publishes nothing: the
//   consumer's wait is what orders memory, so every position is functionally
//   correct and only measurement separates them. It is a compile-time knob
//   here so the sweep is a recompile rather than an edit, and a trigger on the
//   last line is close to a no-op because the kickoff already fires when every
//   CTA has exited.
//
// Measured on this part with `lab/sm120/pdl_unit.cu`, a chain of dependent
// kernels shaped like Pi0's -- each reading a large producer-independent
// weight and a small producer-dependent activation -- under graph replay:
//
//   no PDL                              1.80 us/kernel
//   wait at the very top, trigger last  1.60 us   1.120x
//   wait derived, trigger last          1.60 us   1.121x
//   wait derived, trigger after weights 1.54 us   1.164x
//
// So most of the gain is the launch overlap and is available to any kernel;
// the last 4% needs the wait placed properly, which is only possible in a
// kernel this repo owns.
#pragma once

#include <cuda_runtime.h>

//: Trigger positions. A kernel offers the subset that are its own phase
//: boundaries; `kPdlTriggerLast` is always available and is the conservative
//: choice, being what the hardware does anyway.
#define FLASH_VLA_PDL_TRIGGER_LAST 0
#define FLASH_VLA_PDL_TRIGGER_EARLY 1
#define FLASH_VLA_PDL_TRIGGER_MID 2

#ifndef FLASH_VLA_PDL_TRIGGER
#define FLASH_VLA_PDL_TRIGGER FLASH_VLA_PDL_TRIGGER_EARLY
#endif

namespace flash_vla {
namespace rtx5090 {

//: Wait for the producer. A no-op when the grid was not launched with
//: programmatic serialization, so these can be compiled in unconditionally and
//: switched on from the host.
__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

//: Release the dependent grid. Carries no fence: the consumer's wait is what
//: orders memory, so independent work may still move across this.
__device__ __forceinline__ void pdl_trigger() {
#if __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

//: Trigger only at the position this build selected.
template <int kAt>
__device__ __forceinline__ void pdl_trigger_at() {
  if (kAt == FLASH_VLA_PDL_TRIGGER) pdl_trigger();
}

}  // namespace rtx5090
}  // namespace flash_vla

//: Launch with programmatic serialization when `pdl` is set, and as an
//: ordinary launch when it is not, so the two can be measured against each
//: other without a recompile.
template <class Kernel, class... Args>
static inline cudaError_t pdl_launch(bool pdl, Kernel kernel, dim3 grid,
                                     dim3 block, size_t smem,
                                     cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  if (pdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&cfg, kernel, args...);
}
