# Agent Note: Persistent task loop for the Pi0.5 action-expert attention block

Status: rejected — built, correct, and after the C7518 fix (below) measured
at parity with the TileLang composition it was meant to replace (fused
25.0 us event-graph vs 25.3, 1.01x; span 25.3 vs 23.4); the 10 us target is
not reachable with split joins through global memory at M = 50. The same
task bodies as ordinary per-op kernels (`standalone` mode) beat the TileLang
kernels for qkv (1.16x) and attention (1.05x) and lose on o_proj (0.72x).
Those two winners are what shipped into the pipeline: backend `cuda`
routes `decoder_norm_qkv_rope` and `decoder_attention` per call site, o_proj
stays on TileLang, for -0.3 ms on the 8.3 ms decoder stage (below). The
standalone-first line continues in the plan; the kernel, planner, harness
and contract stay in the tree.

Extends the task-loop idiom of
[`2026-08-26-cute-ffn-kernel-structure`](../../proposed/architecture/2026-08-26-cute-ffn-kernel-structure.md)
to the attention half of the decoder layer. Interface, numerical gate and
timer: [`specs/tile/attention_block_contract.md`](../../../../specs/tile/attention_block_contract.md).
Per-revision measurements and the checklist:
[`specs/tile/attention_block_plan.md`](../../../../specs/tile/attention_block_plan.md).

## Problem

The attention half of the Pi0.5 action expert -- QKV projection, multi-query
attention over the prefix KV cache, gated output projection -- is the largest
single item of the decoder stage. Re-measured through the contract's harness
(job 555047, same process, cold L2, CUPTI over a CUDA graph) the four-kernel
TileLang composition spans **23.3 us** per layer-step: `ada_qkv_gemm_rope`
8.1, `fd_flat_split_mask` 7.9, `fd_flat_combine` 1.6, `matmul_gated_res`
4.9, gaps 0.8. The previously recorded 26.11 us predates the toolchain.

The proposal priced the tile floors at 6.5 us from measured constants and
attributed the gap to grid ramp (four nodes) and TMA issue rate, predicting
that one persistent launch would reach 10 us.

## Decision as built

One persistent 132-CTA task loop per layer-step, static task table, gmem
counters, no scheduler: 80 qkv tasks (40 n-tiles x split-K 2), 64 attention
tasks (8 heads x 8 kv-splits), 64 combine tasks (8 heads x 8 row groups, the
`fd_combine` shape, dealt to the CTAs idle during attention), 128 o_proj
tasks (16 n-tiles x 8 heads), one task per slot, 192 threads (one math
warpgroup, two TMA producer warps), one 160 KB shared pool. Split 0 reduces
the projection partials; every attention split publishes and the combine
tasks fold. Attention tasks are dealt to the CTAs that are idle during qkv
(52 in slot 0, 12 in slot 1 of the split-1 qkv CTAs) and issue every K/V
frame inside the prefix at t = 0; only the frame overlapping the cache suffix
waits on the KV counter, and Q on its head's counter. Every activation row-major, weights consumed as stored, Q and O
head-major, prefix length compile-time.

Four findings changed the design during bring-up and hold for any SM90
kernel here:

- **A 3-D TMA box `{64, rows, k/64}` with the 64-element chunk as the outer
  dimension loads a row-major (rows x 256) bf16 tile in one copy**, landing
  as `[chunk][row][64]`, the CuTe SW128 K-major image. BK = 256 therefore
  costs one 32 KB TMA, not four, and needs no M-major transpose (the FFN's
  BK256 experiments used four 2-D boxes). qkv mainloop 4.7 -> 3.3 us, o_proj
  1.5 -> 0.8 us.
- **`S = Q K^T` must run at >= 64 keys per wgmma.** At N = 32 the
  `m64n32k16` instruction re-reads the 2 KB Q A-tile per 1 KB of K and runs
  at ~75 cycles, 3x [wgmma.issue.wg.ss], shared-memory-bound against the landing TMA
  frames; the ablation (job 555193) put 2.5 of the 5.1 us attention mainloop
  on it. 64 keys per stage: 5.1 -> 3.7 us.
- **A reduction belongs to its own task kind, spread across idle CTAs, not
  to split 0.** The proposal's "split 0 reduces" put 224 KB through one
  CTA's TMA ceiling after the slowest sibling (2.8 us join + 6.3 us
  combine); 64 combine tasks of 8 x 4 KB each do it in ~2 us: 30.9 -> 28.9
  us.
- **A partial must be published with one bulk store from a staged
  shared-memory image.** Thousands of 4-16 B generic stores make the release
  fence wait for every one of them (3.9 us per attention task); one
  `cp.async.bulk` group is one completion: 28.9 -> 27.8 us.
- **Read the whole ptxas output.** Every revision from v1 carried
  `C7518: wgmma.mma_async instructions are serialized` -- the compile filter
  showed only errors, registers and spills -- and the serialised wgmma was
  the "3x over wgmma.issue.wg.ss" term in every ablation. Trigger: register (RS)
  operands refilled across a runtime loop that also contains a divergent
  watchdog exit. Fix: two A fragments alternated by stage parity and a fully
  unrolled stage loop. qkv mainloop 6.85 -> 2.69 us; the task-loop kernel
  went from 27.8 to 25.0 us event-graph without any other change.
- **An in-smem scale pass is not free.** The per-K AdaRMS scale as an RMW
  over the landed frame plus proxy fence and warpgroup barrier cost 0.33 us
  per stage; applying it to the ldmatrix'd register fragment (RS wgmma)
  removes it and halves the wgmma's smem operand reads.
- **Prefetch across the dependency, not across the slot.** Dealing attention
  to CTAs idle during qkv, with frame-level dependencies, lands K/V before Q
  is ready without any cross-slot ring lifetime: 27.8 -> 26.3 us. The
  remaining attention wait is the head's slowest qkv tile. Dropping the qkv
  split to free CTAs instead (job 556008) cost 4.7 us of exposed ring
  refills and was reverted.

## Why it is rejected

The per-task globaltimer timeline (`--timeline`, five stamps per task) and
the per-kind stage tables give the critical path of the final revision (v11,
job 556021, ACD1-2):

| term on the critical path | us |
|---|---:|
| qkv (slowest tile of a head): first frame 1.7, mainloop 3.2, split join 2.0-2.5, epilogue 1.0 | ~9-10 |
| attention: Q load ~1, mainloop 3.7, publish 2.5 (K/V already resident) | 7.2 |
| combine: counter hop, 8 x 4 KB loads 1.4, store 0.5 | ~2.5 |
| o_proj: o_buf wait + first frame 1.2-2.2, mainloop 0.8, join 2.4-3.7, epilogue 1.4 | 6-8 |
| grid ramp, counter memset | ~1.5 |

The four mainloops sum to ~9 us; the dependency hops (a release/acquire
round trip plus one TMA first-frame latency at every task start) and the
partial publishes/joins sum to ~15 us in series. ncu on v4 (job 555126)
shows what that means: 11% DRAM, 4% SM, 91% of scheduler cycles with no
eligible warp. The block is a latency chain, and its links are the
synchronisation points the design chose, not the tiles.

Every measured lever was pulled and is recorded in the plan: thread-major
vector partials (47.6 -> 35.1 us), shared-memory staging by the idle producer
warp (fold compute 5-10 us -> 2-6 us, but the join wait absorbed it), ring
depth 8 for the projections (no change: not refill-bound), the mask slice in
shared memory (0.8 us), BK = 256 (2.5 us), 64-key stages (1.4 us), per-split
staging flags and release-reduction counters (no change), the combine task
kind (2.0 us), bulk-store publishing (1.1 us), attention prefetch on idle
CTAs (1.5 us); no-split qkv rejected (+4.7 us). What is left is the shape of
the task graph itself: three dependency hops and three partial exchanges in
series, each 2-4 us on this machine, against a composition whose kernel
boundaries cost ~1.2 us each and give it a free grid-wide barrier and a free
re-partitioning of the work.

What fusion buys here, priced from these measurements against an equally
tuned set of separate kernels: three boundaries (~4 us) minus three in-kernel
hops (~2 us each), plus cross-op prefetch of the next task's frames (~1-2 us
per kind; now exploited for K/V and the o_proj weights, and worth ~1.5 us
because the remaining wait is the producer's own tail). That is a few
microseconds at best; the larger gains -- BK = 256
via the 3-D box, 64-key S stages -- are op-level and apply to unfused
kernels too.

Standalone-first (owner's direction, 2026-08-27): per op, the same bodies
as grid kernels against the TileLang kernel for that op, CUPTI-over-graph
span of exactly that op's launches, two launches allowed: qkv 7.9 vs 9.3 us
(1.16x, one kernel: BK=128 x depth 4, RS, no split), attention 10.0 vs 10.6
(1.05x, split + 2-row combine), o_proj 8.2 vs 5.9 (0.72x, split-8 + reduce;
a single-launch last-arriver variant measured 8.7 and was rejected). The
projection split-8 pays ~2 us of fixed per-CTA latency plus a partial
exchange for a 64 KB stage; TileLang's o_proj streams full K per CTA with
BM=16 and no reduction. Next lever for o_proj: a full-K or split-2 form with
RS operands (priced in the plan).

Pipeline integration is per call site, not per block: the pipeline's
decoder-side buffers are allocated padded (64 rows, 1024 keys) and exposed
as the old views (contract 3.4), the `cuda` wrappers recover the padded base
and refuse anything else, Q crosses head-major in backend scratch, and the
attention combine writes token-major into `decoder_q_buf` so the TileLang
o_proj is untouched. Rejected alternative: a `decoder_attention_block` call
site -- it would have forced the o_proj choice into the block and hidden the
per-op comparison the standalone-first line is about.

What would change the fused answer is a combine that does not round-trip
through global memory. DSMEM within an 8-CTA cluster is the natural one and is
blocked: a 132-CTA persistent grid with cluster size 8 is not co-resident on
this machine ([`h100-cluster-placement-limits`]), and a persistent counter
protocol deadlocks on a non-resident cluster.

## Alternatives considered

- **M-major activations** (FFN rev 2): superseded by the 3-D box, which
  reaches BK = 256 row-major in one copy.
- **Flash attention with no KV split**: per-CTA ingest of the whole cache
  puts an ~8 us floor on the stage [tma.bw.cta.dram]; still rejected.
- **Cluster / DSMEM combine**: the only route that removes the join hop;
  blocked by cluster placement, see above.
- **A second math warpgroup**: not needed for registers (242, no spills) and
  buys no tensor throughput [wgmma.ratio.sm.wg2].
- **Full-decoder scope** (interleaving FFN tasks into the idle slots): fills
  the idle CTAs but leaves every join and its hop on the critical path;
  not pursued.

## Verification

Kernel `backends/cuda/kernels/attn_taskloop.cu` (header
`sm90_attn_task_desc.cuh`), planner `backends/cuda/attn_taskloop.py`,
harness `eval/correctness/pi05/attention_block_parity.py`. Every revision
passed, in the same invocation as its number: parity of every truncated
table (`qkv`, `attn`, `oproj`, `qkv_attn`, `full`) in both aliasing modes
against `attn_reference` / `attn_block_reference` at cosine > 0.999 (worst
0.9999738 on `out`), prefix and pad cache rows bit-unchanged, replay x3
bit-identical.

Final numbers (job 556329, ACD1-32, CUDA 13.1, torch 2.13.0+cu130, clocks not
pinnable, n = 90 CUPTI samples over 3 A/B rounds): fused 25.3 us span
(single replay), 25.0 us event-graph median; standalone block 22.2 us span,
25.9 us event-graph; composition 23.4 us span, 25.3 us event-graph median.
Per op (standalone vs TileLang, op-bench): qkv 7.9 / 9.2, attention 10.0 /
10.6, o_proj 8.2 / 5.9 (sa5, job 556291). Standalone per-task timeline:
qkv first frame 0.7, mainloop 2.7 (8 stages), epilogue 0.8; attention first
1.6, mainloop 2.6, publish 2.2; o_proj first 1.25, mainloop 0.4, publish 0.5. CUPTI-graph medians for the
fused launch carry a heavy tail (min 32.9, p99 73 us) caused by the 2xL2
flush kernel that precedes each replay delaying co-residency of the persistent
grid; the event-graph timer with rotating sets is the number to compare.

Mixed plan end to end (job 556407, ACD1-13 shared, clocks unpinned,
`sbatch sbatch/plan_e2e.sh`): `plan_parity` against the all-TileLang
decoder -- 1 step x 1 layer actions cosine 0.9999995 with the cache suffix
bit-identical, 1 x 18 layers 0.99993 (gate 0.999 passed), 10 x 18 reported
0.9983 (chaotic regime, not gated); replay bit-identical. Decoder stage
A/B/A in one process, median 8.293 / 8.060 / 8.618 ms, min 8.235 / 7.972 /
8.355 ms: -0.26 .. -0.38 ms by min, matching 180 x (1.3 + 0.6) us from the
per-op numbers. Whole forward wall (min) 18.99 / 18.49 / 18.82 ms.

Commands: `sbatch --job-name=attn-vN sbatch/pi05_cuda.sh
eval/correctness/pi05/attention_block_parity.py --impl both --timeline
--stage-bench --bench --reps 30 --rounds 3`; ncu via
`sbatch -w ACD1-21 sbatch/profile_attn.sh` (report
`profiles/attn/ncu_555126.*`); ablations via `sbatch sbatch/attn_ablation.sh`.

<!-- retired-tags-ok: this note records or predates the tag rename; old spellings here are deliberate. -->
