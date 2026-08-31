# Agent Note: CuTe reference structure for the Pi0.5 FFN taskloop

Status: implemented

## Decision

The Pi0.5 action-expert FFN keeps its fixed persistent CTA/task-table ABI, but
its SM90 implementation is organized around typed CuTe geometry, barrier views,
warp roles, task bodies, and the C ABI boundary. The implementation vocabulary
uses `GatedUp` and `DownResidual`; `gu` and `dr` remain compatibility mode
labels for existing callers. FlashMLA and DeepGEMM are pinned under
`third_party/` as read-only SM90 style references; their dynamic schedulers are
not imported into this fixed offline task graph.

## Consequences

- Task descriptors, warp roles, barrier regions, and GEMM atoms have named
  ownership boundaries in the CUDA headers.
- Packed gate/up weights are documented as one interleaved 64-column tensor;
  the legacy second pointer remains ABI-compatible and unused by the kernel.
- The current BK=64, 224-thread, 132-CTA geometry is unchanged in this pass.
- FlashMLA's CuTe `sm90::gemm` wrapper is reused for WGMMA choreography, while
  repository-specific task dependencies and static dispatch remain local.
- This cleanup is not claimed as a speedup: the next optimization pass must
  recover the small benchmark regression before changing geometry.

## Alternatives considered

- Importing the upstream runtime schedulers was rejected because the Pi0.5
  launch has a fixed offline task table and does not need runtime work stealing.
- A rename-only pass was rejected because barrier lifetime and warp ownership
  would remain implicit.
- Rewriting the WGMMA path in raw PTX was rejected because it duplicates the
  CuTe/FlashMLA contract and makes review harder.

## Verification

On `lab-H100` with CUDA 13.1 and torch 2.13.0+cu130:

- `nvcc -O3 -std=c++17 --shared -arch=sm_90a` succeeded.
- `ffn_taskloop_parity.py --modes gu,dr,full --replay-check 2 --seed 7`
  passed; worst cosine was `0.9999999` and replay cosine was `1.0`.
- Same-process benchmark (`--modes full --bench --reps 10 --seed 7`) measured
  fused `31.96 us`, GatedUp-only `22.56 us`, DownResidual-only `12.51 us`, and
  TileLang composition `22.94 us`; baseline was `31.63/22.22/12.33 us`.
