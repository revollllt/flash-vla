# Agent Note: Pi0's expert stays on TileLang — the CUDA/PDL chain exists to undo a cost Pi0 never pays

Status: rejected — priced before building (job 599747). Pi0.5's chain costs
40.8 to 41.0 us per layer-step where Pi0's shipped TileLang route costs 38.0
to 38.4 us, on one node in one process, so porting the chain would add 0.44 to
0.54 ms to a 7.05 ms stage. The only shape-specific credit available to the
port is bounded at 1.24 us per layer-step and does not close the gap.

Date: 2026-09-07

## Problem

The [optimization campaign](../../proposed/performance/2026-09-06-optimization-campaign-plan.md)
opened lane D0 to move Pi0.5's action-expert CUDA/PDL chain onto Pi0, on the
premise that the chain measures 7.39 ms of `action_expert` against Pi0's
7.70 ms, so a port worth 0.3 ms was there for the taking once the buffers were
padded and the geometry parameterized. Both Targets run the same 18-layer,
1024-wide expert with a 4096 FFN, 8 query heads over 1 KV head at head dim
256, so the kernels transfer; the question was only whether they should.

The premise mixed instruments. The 7.39 ms is a same-process A/B/A of the
Pi0.5 stage; the 7.70 ms is `benchmarks floor`, which runs under kernel
attribution and reports a larger stage minimum than the latency harness does.
Measured the same way on the same node, the two stages are 7.399 and 7.049 ms.

## Decision

Do not route Pi0's action expert to the CUDA chain. Pi0's shipped plan stays
on `tilelang-fused`. The chain remains a Pi0.5 route and now lives in the
shared [`gemma_expert` package](../../implemented/architecture/2026-09-07-gemma-expert-package.md),
so this is a routing decision, not a reason to keep the kernels private.

Job 599747, ACD1-58, one process per Target, reps 100, clocks unlocked, read
`min`, tree `a02bc64` (the routes measured are unchanged by the package move,
which job 599785 proves bit-identical). `action_expert` stage minimum, A/B/A with the shipped
route on both control legs:

| Target | reference (all TileLang) | shipped | control legs |
|---|---:|---:|---|
| Pi0 (`tilelang-fused`) | 8.259 | **7.049** | 7.049 / 7.004 |
| Pi0.5 (`cuda-pdl`) | 8.392 | **7.399** | 7.399 / 7.436 |

Subtracting each Target's non-chain expert sites (Pi0 also runs a state
projection, a second action MLP and a per-step row copy) leaves the five chain
call sites over 180 layer-steps:

| route | per layer-step |
|---|---:|
| Pi0 shipped, TileLang lazy pre-norm | 38.0 - 38.4 us |
| Pi0.5 shipped, CUDA/PDL chain | 40.8 - 41.0 us |

The ranges span both control legs and the uncertainty in the non-chain
subtraction (Pi0's per-step row copies are not separately timed, a 20 us band
over the stage). Both are an order below the 2.4 to 3.0 us gap.

## Why the chain loses on Pi0 and wins on Pi0.5

Pi0's shipped route folds the RMS normalization into the gated FFN's GEMM: the
row sum of squares accumulates from the same shared tile the GEMM already
consumes and the factor scales the fp32 accumulator in the epilogue, which
removes a launch and a serializing bf16 scale at once. Pi0.5 cannot do it,
because AdaRMSNorm needs that tile unscaled for the norm and scaled for the
GEMM, which is recorded in
[the Pi0.5 Target's decisions](../../implemented/architecture/2026-09-06-pi05-target-decisions.md).
The persistent FFN task loop and the PDL chain are how Pi0.5 recovers the
ground AdaRMS costs it.

What that fold is worth, comparing implementations of one call site in one
regime (cupti, cold weights, standalone, same job):

| implementation of `action_expert_norm_gated_ffn` | per invocation |
|---|---:|
| Pi0 shipped, RMS folded into the GEMM | 11.20 us |
| Pi0 reference, factor kernel then GEMM | 19.23 us |
| Pi0.5 reference, AdaRMS factor kernel then GEMM | 22.62 us |

Those numbers compare implementations, not stages: standalone per-site deltas
over-predict the stage delta badly, because outside the graph every case pays
a cold L2 and overlaps with nothing. On Pi0 they sum to 3.35 ms of predicted
saving against a measured 1.21 ms. The stage-level ordering is what decides,
and it is unambiguous: the chain is worth 0.99 ms to Pi0.5 against its own
reference route, Pi0's fusions are worth 1.21 ms against its own, and Pi0's
route is still the faster of the two per layer-step. The chain arrives on Pi0
with nothing left to recover and brings its own overheads instead.

The port would also have added attention work rather than removed it. The CUDA
attention walks a key extent padded to 1024 because its eight splits each walk
two 64-key ring stages; Pi0's cache holds 819 real keys and its FlashDecoding
route walks exactly those. The port keeps the 1024.

## Alternatives considered

- **Shorten the key pad for the Pi0 profile.** `ATTN_SPLIT=7` gives
  `KEYS_PAD = 896`, the smallest legal extent covering 819, cutting attention
  key traffic 12.5%. Bounding the chain's attention share by Pi0's own
  attention cost (9.92 us per layer-step, and the CUDA one does more key work,
  so it is at least that) caps the credit at 1.24 us per layer-step against a
  2.4 to 3.0 us gap. Not built.
- **Drop the AdaRMS terms from the Pi0 build.** Pi0 has no scale, shift or
  gate at any expert norm, and every one of them is a multiply by 1 or an add
  of 0 at the point the kernel applies it, so constant vectors are
  bit-identical to removing them. The one place removal could pay is the QKV
  mainloop, where the scale is indexed by the contraction axis and is re-applied
  per output tile; the kernel is issue-latency bound there (91% no-eligible
  warp at the geometry its header records), so the multiply most likely hides
  in an existing stall. Unquantified, and reachable only through the port.
- **Port only the FFN half.** The route constraints permit it, and the
  persistent FFN is the half whose advantage is least tied to AdaRMS. Not
  measured: it needs the port to exist, and the lane's decision rule stopped
  before that. This is the first thing to try if the question is reopened.
- **Read the standalone per-kernel numbers as the verdict.** Rejected as an
  instrument. Launched outside the pipeline, the chain's atomic groups measure
  76.35 and 55.01 us per layer-step, 2.6x its in-graph cost and worse than its
  own reference route, because every PDL wait blocks on an unrelated
  predecessor and the persistent consumer pays a full ramp with nothing to
  overlap into. The kernel-design wiki records both halves of this already:
  `measure-in-the-graph` says to compare whole-stage wall time per layer
  rather than the sum of kernel durations whenever the decision is whether to
  fuse, and `measuring-persistent-kernels` says an L2 flush before each replay
  delays a persistent grid's co-residency and grows a tail the kernel does not
  have. The same-process A/B/A is the instrument; it agrees with this verdict
  by a smaller margin.

## Consequences

- Pi0's `action_expert` headroom stays where the floor report puts it, and it
  is not a launch-structure problem: the stage is 7.05 ms against a 4.46 ms
  measured ceiling, and the chain's launch and dependency structure is not the
  lever that closes it.
- Pi0's expert buffers are not padded and Pi0 declares no `mask_bias`. Lane C
  wants both for the backbone attention it is porting; the change is written
  and kept as a patch in this task's workspace rather than committed here, so
  whoever needs it owns it.
- The `gemma_expert` package ships with one registered consumer. Registering
  the second still requires the row count and the prefix length to become
  per-profile compile-time constants, which nothing here has done.
- Lane D1 prices the expert megakernel on Pi0.5 and does not depend on this.

## Verification

Job 599747, ACD1-58, tree `a02bc64`, torch 2.13.0+cu130, CUDA 13.1, clocks
unlocked. `python -m benchmarks latency --target <t> --plan shipped --plan
reference --plan shipped --reps 100` on both Targets, and `python -m
benchmarks kernels --target <t> --plan {shipped,reference} --segment
action_expert --timer cupti` on both. Control spread on the chunk minimum was
0.012 ms (Pi0) and 0.072 ms (Pi0.5), both inside the registry's 0.10 ms
validity limit, and the minimum detectable effect on the `action_expert`
minimum was 0.044 and 0.037 ms, an order below the 0.44 to 0.54 ms this note
attributes. No candidate was built, so there is no parity result and no gate
verdict.

## Related notes

- [the Gemma expert package](../../implemented/architecture/2026-09-07-gemma-expert-package.md):
  the move that made this a routing question.
- [decoder PDL chain](../../implemented/architecture/2026-09-02-decoder-pdl-chain.md):
  what the chain buys on the Target that needs it.
- [the Pi0.5 Target](../../implemented/architecture/2026-09-06-pi05-target-decisions.md):
  why AdaRMS forbids the fold Pi0 uses.
