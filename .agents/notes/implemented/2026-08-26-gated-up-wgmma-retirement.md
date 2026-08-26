# Agent Note: Retire GatedUp WGMMA groups at the measured pipeline knee

Status: implemented

## Decision

The SM90 GatedUp mainloop keeps its N=64 WGMMA, BK=64 tile, and four-stage TMA
ring. Each BK stage already commits four WGMMA instructions, so it retains one
group rather than tying retirement to the full ring depth. GatedUp barrier
phases are derived arithmetically instead of stored in runtime-indexed local
arrays. DownResidual phase handling is unchanged because applying the same
transformation there regressed its isolated latency.

## Consequences

- GatedUp releases shared-memory frames earlier, allowing its producers to
  reuse the TMA ring without draining the tensor-core pipeline.
- The GatedUp producer and consumer paths no longer perform local-memory phase
  loads/stores.
- BK=128 remains rejected for the row-major activation layout: its activation
  tile requires two legal SW128 TMA boxes and raises register use without a
  latency improvement.
- Moving the scale slice to synchronous loads or a third producer warp remains
  rejected; neither improved the critical path.

## Alternatives considered

- Retaining three WGMMA groups was rejected because it delayed frame retirement
  after the per-stage instruction batch had already filled the measured MMA
  pipeline.
- BK=128 at depths three through five was rejected because its best result was
  within noise of BK=64 while using more registers and shared memory.
- Depth three and five were rejected; the fused TMA/WGMMA consumer needs four
  stages even though the isolated copy-engine unit reaches its issue knee at
  depth three for these box sizes.

## Verification

On `lab-H100`, H100 80GB HBM3, CUDA 13.1 and torch 2.13.0+cu130:

- GatedUp cold-weight latency improved from `22.53 us` to `15.18 us` in the
  experiment matrix; ptxas reported 110 registers and zero spills.
- The isolated production patch measured GatedUp `15.64 us`, DownResidual
  `12.38 us`, fused `25.25 us`, and TileLang composition `22.84 us` with 30
  CUDA-graph samples over three cold weight sets.
- `gu`, `dr`, and `full` comparisons passed with worst cosine `0.9999999`; two
  graph replays passed with cosine `1.0`.
