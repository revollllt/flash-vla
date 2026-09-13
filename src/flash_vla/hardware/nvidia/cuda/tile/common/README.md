# Arch-neutral tile primitives

Header-only CuTe building blocks that carry no Hopper-specific instruction:
element aliases and thread-role helpers, mbarrier wrappers, TMA loads and
stores, `cp.async`, and the host-side tensor-map builders. Every instruction
here assembles for `sm_90a`, `sm_120` and `sm_120a` alike -- see the ISA matrix
in [rtx5090/measured/isa-support.md](../../../rtx5090/measured/isa-support.md).

Namespace: `flash_vla::tile`. Include root is unchanged
(`src/flash_vla/hardware/nvidia/cuda`), so a header is
`#include "tile/common/barrier.cuh"`.

| Header | Owns |
| --- | --- |
| `common.cuh` | element aliases (`BF16`, `F16`, `E4M3`, `E5M2`), `Operand`, `Major`, thread-role helpers, proxy fences, named barriers, the `is_wgmma_v` trait |
| `barrier.cuh` | `FullBarrier` / `EmptyBarrier`, non-blocking parity test, `PhaseRing` |
| `copy_g2s.cuh` | global to shared: TMA 2d/3d, multicast, L2 prefetch, bulk load, `cp.async` |
| `copy_s2g.cuh` | shared to global through the async proxy: TMA store, bulk store, group commit/wait |
| `tma_host.cuh` | host-side `cuTensorMapEncodeTiled` builders |

## What stayed behind, and why

`tile/sm90/` keeps what is genuinely Hopper: `wgmma.cuh` and the `gemm.cuh`
selector that dispatches to it, plus the MMA-adjacent headers that are shaped
around a warpgroup accumulator -- `smem_layout.cuh`, `copy_s2r.cuh`,
`copy_r2s.cuh`, `mma_sync.cuh`. Those are arch-neutral in code but not yet in
intent, and moving them is a separate change with its own reason.

`is_wgmma_v` lives here rather than with `wgmma.cuh` despite its name. It is a
CuTe type trait -- true when a `TiledMma`'s B fragment is a descriptor iterator
-- and needs no Hopper header to compile; on a target with no warpgroup MMA it
is simply always false, which is what lets `gemm.cuh`'s selector and
`copy_r2s.cuh`'s atom choice compile there at all.

## Compatibility

`tile/sm90/<name>.cuh` still exists for each header moved here, as a one-line
include plus `namespace flash_vla::sm90 { using namespace flash_vla::tile; }`.
No kernel's includes or qualified names changed.

That equivalence is checked rather than assumed. `lab/sm120/tile_ptx_gate.sh`
compiles all seven kernels that reach this library to `sm_90a` PTX and diffs it
against a baseline; the move was made under a **byte-identical** result on all
seven. There is no H100 on this machine to run them on, so compiling is the
strongest available evidence -- and it is strong: a refactor that changed
behaviour would have to do so without changing a single PTX instruction.
