# Agent Note: GatedUp resident scale does not shorten the critical path

Status: rejected experiment

## Hypothesis

The BK64 GatedUp weight producer issues one packed-weight TMA and one 128-byte
scale bulk copy for each of 16 stages. At the measured 248 ns issue interval,
making the complete 1024-element BF16 scale vector resident in shared memory
would reduce the producer from 32 transactions to 17 and its issue floor from
about 7.94 us to 4.22 us.

## Candidate

- Reserve a 2048-byte resident-scale region at shared-memory offset 65536.
- Move the barrier pool from offset 66048 to 67584; total dynamic shared memory
  becomes 67712 bytes.
- Use a named transaction barrier after the GatedUp full/empty rings.
- Have the weight producer issue one complete scale copy before its stage loop,
  then immediately continue issuing weight TMA so the scale prologue overlaps
  the first activation/weight fills.
- Have all math warps wait once before the mainloop and read stage slices from
  the resident vector.
- Keep BK64, depth4, N64, wait1, activation transport, and DownResidual
  unchanged.

The task contract was also audited: the kernel launch supplies one global
`S[1024]` pointer, and the current planner assigns exactly one GatedUp task to
each active CTA. The scale/barrier region safely aliases later DownResidual
storage because task slots end with a CTA-wide synchronization.

## Result

H100 job 551835, CUDA 13.1, torch 2.13.0+cu130, 30 CUDA-graph samples over
three cold weight sets:

| Path | PR #3 baseline | Resident scale |
| --- | ---: | ---: |
| GatedUp | 15.64 us | 15.72 us |
| fused | 25.25 us | 25.30 us |
| DownResidual | 12.38 us | 12.22 us |
| TileLang composition | 22.84 us | 22.89 us |

The candidate minimum fused latency was 25.00 us, also indistinguishable from
the baseline 25.01 us. `cuobjdump` reported 110 registers, zero stack, and zero
local memory. The minimal GatedUp parity gate passed with cosine 1.0.

Jobs 551791 and 551813 did not execute the kernel: the first exposed an
uninitialized FlashMLA submodule in the new worktree, and the second caught a
compile-time scale-slice index omission. Neither is included in the performance
decision.

## Decision

Reject the resident-scale implementation and retain the PR #3 production
kernel. Reducing 15 scale issue operations did not move either median or
minimum latency, so those operations are overlapped and are not on the current
critical path. A combined weight-plus-scale payload is therefore also rejected:
it adds a prepack/layout contract without evidence that further scale transport
work can improve the fused path. Scale-folded weights are not pursued because
transport removal was neutral and folding would additionally change BF16
rounding.

The next experiment is the BK256 dataflow upper bound, which changes the
weight-stage cadence and activation layout rather than optimizing an already
hidden scale transaction.
