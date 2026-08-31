# Pi0.5 action-expert attention block: optimization plan and checklist

Status: closed 2026-08-27 (Phase 3 gate not met; see Outcome).
Contract: [`attention_block_contract.md`](attention_block_contract.md) (the
interface, parity gate and timer every item below is measured under).
Decision record (rejected 2026-08-27 with the measured attribution):
[`2026-08-27-attention-block-taskloop`](../../.agents/notes/rejected/architecture/2026-08-27-attention-block-taskloop.md).

**Outcome.** Phases 0-4 ran; the kernel is correct on every gate and, after
eleven measured revisions, 26.3 us per layer-step against the composition's
23.4 us (0.88x; event-graph 28.0 vs 25.2). Phase 3's gate ("target 10 us") is not met and the note records the
reason: ~16 us of split-join latency in series on the critical path. Phase 5
integrates the standalone-first winners instead of the fused kernel: the
`cuda` backend routes `decoder_norm_qkv_rope` and `decoder_attention`, o_proj
stays on TileLang (see Standalone-first, sa7, and Phase 5).
Sibling plan for the other half of the layer:
[`ffn_megakernel_optimization_plan.md`](ffn_megakernel_optimization_plan.md).

## Goal, target, invariant

Replace the four-node TileLang composition of one layer-step
(`ada_qkv_gemm_rope -> fd_split -> fd_combine -> matmul_gated_res`, with the
RMS factor an input on both sides) with one persistent 132-CTA task-loop
launch behind the same call signature -- the `attn_taskloop_launch` ABI
mirrored by `attn_block_reference.py`.

- Target **10 us** per layer-step under the contract's timer, against a
  **7.7 us** floor (6.5 us of tile floors from measured constants plus one
  `launch.lat.dev.ramp`) and against the composition **re-measured in the same
  process** (the recorded 26.11 us is a stale-toolchain number and is not a
  baseline).
- Invariant, as for the FFN: static per-CTA task table, `blockIdx.x` indexes
  its own row, gmem counters for ordering, no device-side scheduler, no work
  stealing, no cooperative launch. Parity and replay are hard gates before any
  number is reported.

## Working rules (from `AGENTS.md`)

1. **Screen before you build.** Every candidate change is priced first with
   the analysis scripts in [`attention_block_analysis/`](attention_block_analysis/)
   (`wave.py` floors, `layout.py` transaction counts, `budget.py` TMA
   legality and smem/register union). A candidate whose predicted gain is
   below the ~6% noise floor is not run; it is recorded as screened out.
2. **One variable per experiment**, control rebuilt in the same job, A/B
   interleaved when the delta is small (contract §6.4).
3. **Rule out confounders before attributing**: clocks unpinned, node
   differs, TileLang recompiled, a different `torch`, a warm L2 — each is
   checked and named in the report before a kernel change is credited.

## Phase 0 — contract, harness, baseline

Nothing in this phase touches kernel code.

- [x] Rebase the working branch onto `main` (2026-08-27: fast-forwarded
      `fix-third-party-submodules` 429e459 -> c4e7b48 with a one-file stash;
      the FFN retirement change is now present).
- [ ] Split the pending working-tree changes into separate PRs (skill/doc
      refactor, Agent Notes infrastructure, attention proposal + contract +
      plan + kernel). Owner's call: the plan commits nothing.
- [ ] `specs/tile/attention_block_contract.md` reviewed and accepted by the
      task owner. Settled there, not in code: the interface is the
      `attn_taskloop_launch` ABI (`rms_factor` an input, `out` aliasable with
      `x`, head-major `q_buf` / `o_buf` as caller-allocated observable
      scratch, `KEYS_PAD = 1024`, pad rows unspecified under the
      row-independence invariant).
- [x] `attn_block_reference.py` cross-checks against `models/pi05/reference`
      on CPU (2026-08-27: out 0.9999286, k/v suffix 0.999996; run with
      `OMP_NUM_THREADS=1` on the login node).
- [x] Analysis scripts committed under `specs/tile/attention_block_analysis/`
      and re-run once; their printed tables match the proposal's floor table.
- [x] `eval/correctness/pi05/attention_block_parity.py` (job 555044: all fused modes pass):
  - [x] Inputs from `attn_block_reference.make_inputs(device="cuda", seed,
        alias_out)`; both aliasing modes are exercised.
  - [x] Block gate: `AttnBlockReference.forward` on cloned inputs, compared on
        real rows and cache suffix per contract §4; prefix and pad cache rows
        bit-unchanged; replay bit-identity.
  - [x] Stage bisection: `attn_reference` pre-fills `q_buf` / caches /
        `o_buf` and supplies the per-kind expected outputs for the truncated
        tables (`qkv`, `attn`, `oproj`).
  - [x] TileLang **control adapter** with the same call signature: compiled
        `tl_ada_qkv_gemm_rope` fed `rms_factor[:M]` directly (no
        `tl_rms_factor` node), `decoder_attention` on the `KEYS` rows of the
        caches with a private token-major Q scratch, `tl_matmul_gated_res`
        into `out[:M]`; `q_buf` / `o_buf` left untouched.
  - [x] `--bench`: CUPTI-over-graph timer (§6.2), per-kernel record dump with
        the gap term (§6.3), event-graph cross-check with enough rotating sets
        to exceed L2, JSON report, refusing to run unless parity passed in
        the same invocation.
  - [x] `--timer` is recorded in the report; a CUPTI fallback to events is an
        error, not a warning.
- [x] **Baseline job** 555047 (ACD1-59, torch 2.13.0+cu130, clocks NOT
      pinned): TileLang composition through the harness = **23.33 us span**
      (qkv 8.10 + fd_split 7.90 + fd_combine 1.57 + o_proj 4.93, gap 0.83);
      CUPTI-graph median 26.37 us (min 25.44, p99 27.26, n=90); event-graph
      cross-check 25.53 us (10 rotating sets). This replaces 26.11 us.
- [x] Harness soundness check: CUPTI median 26.4 vs event-graph 25.5 us on
      the composition (3.4%, inside the noise floor). On the fused launch the
      CUPTI samples have a heavy tail (min 54, median 56, p99 117 us) that the
      event-graph timer does not show (51.4 median / 51.0 min): the 2xL2
      flush kernel that precedes each CUPTI replay delays co-residency of the
      132 persistent CTAs. Read the fused number from the event-graph timer
      or the single-replay records until the flush is replaced by rotation.

Gate: composition parity passes; baseline recorded with all §6.3 metadata.

## Phase 1 — planner and table validation (host only)

- [x] `backends/cuda/attn_taskloop.py` mirroring `taskloop.py`: `build()`
      (nvcc, hashed cache), `build_table(mode)`, `validate_table()`,
      `AttnTaskloop.launch()` with on-stream counter zeroing, watchdog buffer.
- [x] Table modes: `qkv`, `attn`, `oproj`, `full`, plus `qkv+attn` for the
      first dependency chain. Sentinel rows for idle CTAs; the grid stays 132.
- [x] Validator proves, offline: every task has one owner; every counter's
      arrive count equals its producer count (`kQArrive=4`, `kKvArrive=8`,
      `kAttnArrive=7`, `kOutArrive=7`); the splits of one tile are on
      consecutive CTAs (co-residency makes split 0's wait a poll); each slot
      is monomorphic; `QKV_TASKS + ATTN_TASKS + OUT_TASKS <= 132 x 3`.
- [ ] `check.cpp` static asserts folded into the header or a host unit test
      so a geometry edit that breaks the table fails at compile time.
- [x] Watchdog site table (`WATCHDOG_SITES`) defined for every wait in the
      three bodies before the bodies exist.

Gate: validator passes on all modes; a deliberately corrupted table is
rejected.

## Phase 2 — task bodies, one kind at a time

Measured revisions (all on the full 132-CTA table, parity + replay passing
every time; fused kernel duration from the single-replay CUPTI record,
composition span in the same job):

| rev | job | node | change | fused | composition | notes |
|---|---|---|---|---:|---:|---|
| v1 | 555047 | ACD1-59 | first correct kernel | 47.6 us | 23.3 us | joins from global in (row, col) order, 255 regs + spills |
| v2 | 555092 | ACD1-59 | thread-major vector joins, hoisted epilogue loads, in-place ex2 softmax | 35.1 us | 23.6 us | timeline: qkv mainloop 4.9, attn mainloop 6.2, split-0 joins still 5-10 us |
| v3 | 555106 | ACD1-11 | projection ring depth 8, mask slice in smem, join-wait stamp | 36.8 us | 23.4 us | depth 8 changed nothing (qkv mainloop 4.96 -> 4.96): not ring-refill bound; mask in smem: attn mainloop 6.2 -> 5.4; fold compute after the join still up to 4.7 / 9.8 / 3.5 us (register-starved global loads) |
| v4 | 555124 | ACD1-1 | split partials bulk-copied into the freed pool by the act producer (drain barrier), folded from smem; 230 regs, 0 spills | 35.2 us | 23.4 us | fold compute now 3.4 / 5.9 / 2.2 us max, join+staging 1.7 / 2.7 / 2.4 us; per-body times unchanged: qkv mainloop 4.7 (HBM floor ~2), attn 1.3 us/stage; ncu capture job 555126 |
| v5 | 555137 | ACD1-9 | BK=256 projections: x and o_buf as one 3-D box {64,64,4} (32 KB), weights as one {64,256} box; qkv 2 stages, o_proj 1 | 32.7 us | 23.1 us | ncu (555126) on v4: DRAM 11% / SM 4% / no-eligible 91% -> TMA-count bound [tma.issue.warp]; qkv mainloop 4.7 -> 3.3, o_proj 1.5 -> 0.8; qkv tail (end 5.3 median, 11.7 max) unchanged and sets the critical path |
| abl | 555193 | ACD1-52 | attention mainloop ablations on v5 (attn-only table, 4 stages): baseline 5.12 us, no-softmax 5.06, no-PV 4.29, no-S 2.62 | - | - | S = Q K^T carries 2.5 us: m64n32k16 runs ~75 cycles/instr (3x [wgmma.issue.wg.ss]) because N=32 re-reads the 2 KB Q A-tile per 1 KB of K -- shared-memory-bound against the landing TMA frames |
| v6 | 555209 | ACD1-9 | attention 64 keys/stage (m64n64k16 S gemm, 32 KB K/V frames, depth 2, 2 stages/split) | 30.9 us | 23.0 us | attn mainloop 5.1 -> 3.7; slowest-task print: mainloops sum to ~11 us, split-join waits + folds to ~16 us (qkv split-0 join 1.6 + epilogue 2.0; attn join 2.3-3.6 + two-round combine 5.5; o_proj join 2-4) |
| v7 | 555226 | ACD1-9 | per-(head,split) staging flags (stage each sibling as it publishes), `red.release.gpu` counters, rope loads hoisted above the join | 30.9 us | 23.0 us | qkv split-0 epilogue 2.0 -> 1.0; attention combine unchanged (6.3 us: slowest sibling + 224 KB at the per-CTA TMA ceiling); total unchanged. Final. |
| v8 | 555286 | ACD1-52 | `kCombine` task kind (fd_combine shape): every split publishes; 64 (head, 8-row) combine tasks on the CTAs idle during attention; o_proj waits 8 combines | 28.9 us | 23.5 us | attention combine 6.3 -> ~2 us (1.1 load + 0.7 store per task, parallel); next largest: publishing a partial with scattered stores + release = 3.9 us per attention task, o_proj join 3.0 |
| v9 | 555300 | ACD1-50 | every partial staged in the freed pool and published with ONE `cp.async.bulk` store + `wait_group 0` + proxy fence before the release | 27.8 us | 23.4 us | attention publish 3.9 -> 2.5 us, attention stage-only 17.3 -> 14.3; event-graph 29.1 vs 25.2 (0.87x). Remaining chain: 3 dependency hops (~2 us each: counter RTT + first TMA frame), 3 publishes/joins (2.0 / 2.5 / 2.3-4.6), mainloops 3.2 + 3.7 + 0.8 + 1.3 |
| v10 | 556008 | ACD1-2 | `QKV_SPLIT=1` (40 tasks x 4 stages) AND attention dealt to the slot-0-idle CTAs with frame-level K/V dependencies (prefix frames issued at t=0, only the suffix frame waits on kKv, Q waits on its head) | 30.6 us | 23.0 us | two effects in opposite directions: no-split qkv mainloop 3.2 -> 7.9 us (two exposed refills on a depth-2 ring) = REJECTED; attention prefetch cut the post-qkv chain from 9.4 to 7.2 us = kept |
| v11 | 556021 | ACD1-2 | split-K 2 restored; attention dealt to the 52 slot-0-idle CTAs plus slot 1 of the 12 split-1 qkv CTAs (free ~3 us before any Q counter flips); combine on CTAs 24..87 slot 1 | **26.3 us** (span 27.0) | 23.4 us | event-graph 28.0 vs 25.2 (0.88x). Attention dep+first is now purely the Q counter (8.8 us median = the head's slowest qkv tile); K/V prefetch bought ~1 us. Final. |


Each body lands only after its stage gate (contract §5) passes on the
truncated table, then its stage-only time is compared with its floor. Order
follows the dependency chain so each step's inputs can be pre-filled from the
reference.

### 2a `kQkvProj`

- [x] Producer warps: row-major `x` ring (8 KB, SW128) and as-stored `w_qkv`
      ring (BN=64 box, 128 B rows); `ada_scale` slice via 1-D bulk copy.
- [x] Math: in-smem `bf16(x*s)` scale (FFN pattern), N=64 wgmma, `wait_group`
      at the measured knee, release-on-retirement.
- [x] Epilogue: split-K join (split 0 reduces the fp32 partial in fixed
      order), `rstd` then bias in fp32, adjacent-pair RoPE on Q and K columns
      only, head-major store into `q_buf`, cache suffix rows for K/V, **no
      store to rows `>= M` of the cache**, counter release per head / KV.
- [x] Stage gate on `q_buf`, `k_cache[968:1018]`, `v_cache[968:1018]`.
- [x] Stage time vs floor 1.98 us: 12.4 us stage-only (v7); mainloop 3.2 us
      after BK=256, the rest is first-frame latency, the split join and the
      epilogue (timeline, job 555226).

### 2b `kAttention`

- [x] Q slab resident (32 KB) loaded once; K and V rings (depth 4, 16 KB
      frames); per-split trip count is compile-time (`ATTN_TRIP = 4`).
- [x] Online softmax in fp32 with the additive mask; pad keys `[1018, 1024)`
      and the prompt hole are masked, never skipped; running max initialized
      finite so a fully masked tile cannot produce NaN.
- [x] Split join: splits 1..7 write bf16 partial + fp32 (m, l), release the
      head counter; split 0 combines in fp32 and stores head-major `o_buf`,
      then releases the "head combined" counter.
- [x] Stage gate on `o_buf` with `q_buf` and caches pre-filled from the
      reference and the Q/KV counters pre-satisfied.
- [x] Stage time vs floor 1.62 us: 18.4 us stage-only (v7); 242 registers, no
      spills after the shared-memory folds (v1 had 255 + 232 B spills).

### 2c `kOutProj`

- [x] Split = head: task `(n, h)` reads `o_buf[h]` as its k-slice and
      `w_o[h*256:(h+1)*256, n*64:(n+1)*64]` as stored.
- [x] Split join into split 0; epilogue `out + acc * g` in fp32, bf16 store;
      correct whether or not `out` aliases `x` (each thread reads its element
      before writing it, and `x` is not re-read after the qkv stage).
- [x] Stage gate on `out` with `o_buf` pre-filled, both aliasing modes.
- [x] Stage time vs floor 1.51 us: 12.2 us stage-only (v7); mainloop 0.8 us.

### 2d Full table

- [x] `qkv+attn` chain passes, then `full`.
- [x] Replay check (fresh counters, fresh `x`, identical outputs) x3.
- [x] Truncated-table bisection retained as harness modes, not deleted.

Gate: full parity + replay pass; every wait has a watchdog site.

## Phase 3 — acceptance measurement

- [x] Same job, same process: fused block vs composition adapter, contract
      timer, `n >= 30`, A/B interleaved x3.
- [x] Report: median/min/p99, gap term, per-kernel records, environment.
- [x] Classify the gap to the 7.7 us floor by binding term (TMA issue, per-CTA
      ceiling, HBM, L2 re-read, counter RTT, ramp) using the constants tags;
      confirm with one `nsys` timeline (TMA gaps) and one `ncu` capture
      through `sbatch/profile_ffn.sh`'s pattern.
- [x] Update the Agent Note: `implemented` with verification, or `rejected`
      with the measured reason. Either outcome is a result.

Gate: parity passed in the same invocation as the reported number; the note
names the harness command, job id, node, timer backend and `n`.

## Phase 4 — isolated experiments (after Phase 3, one at a time)

Each item is screened with the analysis scripts first; predicted gain, the
term it moves, and the noise floor are written down before the job is
submitted. Suggested order by expected value:

- [x] `QKV_SPLIT` 2 -> 1 (v10): rejected, +4.7 us on the qkv mainloop (refill latency on a depth-2 ring); 2 -> 4 screened out after v5.
- [x] Combine as its own task kind (v8): -2.0 us, see the table. Reverses the proposal's "split 0 reduces" decision for attention.
- [ ] `ATTN_SPLIT` 8 -> 4 and 16: screened out -- 4 doubles the 3.7 us mainloop for a 0.7 us smaller combine; 16 doubles the partials.
- [x] `ATTN_BKK` 32 -> 64 with depth 2 (v6): -1.4 us, the S gemm was smem-bound at N=32.
- [x] Task-table order (v11): attention on the CTAs idle during qkv with
      frame-level K/V dependencies, ~-1.5 us; L2 reuse ordering not needed
      (the cache is L2-resident for every split).
- [ ] Fold `rms_factor` into the fused launch (removes one node; needs a
      full-row reduction the qkv split does not have — price it first).
- [ ] Counter self-reset on the device (removes the memset node; replay-safe
      by construction). Screened out: the memset is 0.7 us and the gap is 8 us.
- [x] Second math warpgroup **only** if 2b measured spills: no spills after v4; not taken.
- [ ] Cluster/DSMEM combine: blocked on a DSMEM bandwidth probe in
      `hardware-unit-test` before it can be costed.

Anything that improves a stage-only time but loses the full block, or
violates parity/replay, is rejected and recorded.

## Standalone-first (2026-08-27, after the task loop plateaued)

Decision (owner): make each op beat its TileLang kernel as an ordinary grid
kernel before spending anything on launch overhead; two launches per op
(split + reduce/combine) are allowed. The same task bodies run in a
`standalone` mode (no counters, dependencies by launch order, every split
publishes) behind the contract's call signature, so the per-op number, the
task-loop number and the TileLang number all measure the same code path.
Per-op timer: CUPTI-over-graph span of exactly that op's launches
(`--op-bench`).

| rev | job | qkv (sa / tl) | attention (sa / tl) | o_proj (sa / tl) | notes |
|---|---|---:|---:|---:|---|
| sa1 | 556125 | 11.6 / 9.3 (0.80x) | 10.6 / 10.5 (0.99x) | 12.3 / 5.9 (0.48x) | per kernel: qkv split 6.2 + reduce 2.7; attn split 7.3 + combine 2.4; o_proj split 3.4 + reduce **6.1** (16 CTAs x 128 KB at one SM's bandwidth). qkv split alone already beats the whole TileLang qkv kernel (8.0). |
| sa2 | 556141 | 11.7 / 9.3 (0.80x) | **10.0 / 10.6 (1.05x)** | 8.1 / 5.9 (0.72x) | reduce kernels spread over 8 fragment groups per tile (o_proj reduce 6.1 -> 1.9), combine at 2 rows per CTA (2.4 -> 1.8), standalone publishes wait only for the smem read. qkv reduce unchanged (2.7: launch ramp + dependent loads, not bandwidth) -> qkv must be one kernel. Standalone block 28.5 = fused 28.7. |
| sa3 | 556156 | 15.4 / 9.3 (0.60x) | 10.0 / 10.5 (1.05x) | 8.2 / 5.9 (0.72x) | single-kernel qkv at BK=128 x depth 4 (40 CTAs x 8 stages): the qkv kernel alone 10.9 us, worse than the 80-CTA split kernel (6.2). Per-CTA delivery sits at ~30-40 GB/s whatever the frame size or depth (BK=64/256, depth 2/4/8) -- far under [tma.bw.cta.dram]; suspect: every CTA reads the same x tile at the same time (L2 hot-spot). Ablation job 556163 (no-x / no-W). |
| abl | 556169, 556195 | - | - | - | qkv standalone timeline + ablations (40 CTAs, 8 stages, no dependencies): mainloop 8.2 us; no x/W traffic 7.5; also no smem scale 4.9; also no wgmma 3.8; both off 1.4. Per stage: traffic 0.09, in-smem scale 0.33, SS wgmma 0.46 (110 cycles per m64n64k16, 3.3x wgmma.issue.wg.ss), barriers 0.18 -- shared-memory bandwidth, not DRAM, binds the projection mainloop. The x-broadcast hot-spot hypothesis is refuted (no-x saves 0.7 us). |
| sa4 | 556213 | 11.4 / 9.3 (0.82x) | 9.9 / 10.7 (1.08x) | 8.1 / 5.9 (0.73x) | qkv A operand via ldmatrix into registers, AdaRMS scale applied there, RS wgmma; single-launch qkv (no reduce). qkv kernel 10.9 -> 10.05 us, mainloop 8.2 -> 6.85 (0.86 us/stage): the RMW/fence/barrier went away but the wgmma term did not. ncu on the standalone kernels: job 556221. |
| wt | 556225 | 11.5 vs 11.7 | - | - | B-major experiment: W_qkv pre-transposed (K-major B, 3-D box) vs as stored (MN-major B): qkv mainloop 6.78 vs 6.85 -- no difference. The contract's "weights as stored" clause costs nothing; hypothesis refuted. |
| C7518 | (compile) | - | - | - | **ptxas `C7518: wgmma.mma_async instructions are serialized`** on the qkv standalone kernel AND the whole task-loop kernel (every revision since v1 with split-K, and the RS qkv): the compile filter only showed errors/registers. Cause: RS register operands refilled across a runtime loop with a divergent watchdog exit; fix: two A fragments alternated by stage parity + fully unrolled stage loop. This is the 3x-over-`wgmma.issue.wg.ss` term measured in every ablation. |
| sa5 | 556291 | **7.9 / 9.3 (1.17x)** | **9.9 / 10.7 (1.08x)** | 8.2 / 5.9 (0.72x) | C7518 fixed (two register A fragments, unrolled stage loop): qkv mainloop 6.85 -> 2.69 us (0.34 us/stage), qkv kernel 6.1 us; the task-loop kernel un-serialised too (attention mainloop 3.65 -> 2.50): fused 25.6 us span / 25.0 event-graph = **parity with TileLang (0.99x)**; standalone block 21.7 span / 25.2 event-graph. |
| sa6a | 556317 | - | - | FAIL | single-launch o_proj, last arriver folds the other seven: parity passed but replay was not bit-identical (fold order depended on who arrived last) and the counters were shared with the task-loop path (left at 7). Fixed in sa6: dedicated `kSaOutBegin` counters, fold all eight in head order. |
| sa6 | 556329 | 7.9 / 9.2 (1.16x) | 10.0 / 10.6 (1.05x) | 8.7 / 5.9 (0.68x) | deterministic last-arriver o_proj (all eight folded in head order, own counters): correct and replay-safe but slower than split + reduce (the last CTA folds 128 KB alone, 2.2 us) -> reverted to split + reduce (sa5 numbers stand). Fused 25.0 event-graph = 1.01x. |
| sa7 | 556407 | e2e | e2e | (TileLang) | **mixed plan end to end** (`--plan attn-cuda`: qkv + attention on `cuda`, o_proj on TileLang; combine writes token-major into `decoder_q_buf`). Plan parity vs all-TileLang: 1 step x 1 layer actions cosine 0.9999995, cache suffix bit-identical; 1 x 18 layers 0.99993 (passed); 10 x 18 reported 0.9983 (chaotic regime). Decoder stage, same process A/B/A, ACD1-13 shared, clocks unpinned: median 8.293 / **8.060** / 8.618 ms, min 8.235 / **7.972** / 8.355 -- about -0.3 ms by min (-0.26 .. -0.38), the predicted 180 x (1.3 + 0.6) us = -0.34 ms. Whole forward (wall, min) 18.99 / 18.49 / 18.82 ms. |

## Phase 5 — pipeline integration (mixed plan, 2026-08-27)

Integrated per call site rather than as a block: the fused kernel did not
earn a `decoder_attention_block` call site, the standalone winners did.

- [x] `buffers.py`: the decoder-side buffers are allocated padded
      (`ROW_PAD = 64`: 64 query rows, 1024 cache keys) and exposed under the
      old keys as leading-row views, so every TileLang consumer keeps its
      shapes; pad mask entries `MASK_NEG`, every other pad row zero, nothing
      writes a pad row (contract 3.4).
- [x] `backends/cuda/wrappers.py`: `decoder_norm_qkv_rope` and
      `decoder_attention` with the TileLang signatures, registered as backend
      `cuda`; Q crosses head-major in backend scratch (the attention wrapper
      checks its qkv ran), the combine writes token-major into the
      pipeline's `decoder_q_buf` (standalone op 6) so the TileLang o_proj is
      unchanged. `Pi05Inference(plan={"decoder_norm_qkv_rope": "cuda",
      "decoder_attention": "cuda"})`; `benchmarks.e2e_pi05 --plan attn-cuda`.
- [x] `eval/correctness/pi05/plan_parity.py`: two engines, one checkpoint and
      input, actions + cache-suffix metrics, replay check; gate at
      `--steps 1` (cosine > 0.999), deep run reported.
- [x] End-to-end before/after in one process (A/B/A), `sbatch
      sbatch/plan_e2e.sh`: sa7 above, -0.3 ms on the 8.3 ms decoder stage.
- [ ] o_proj on `cuda` once its standalone kernel beats TileLang (0.72x today).
- [ ] Only then: the full-layer scope (attention + FFN slots interleaved),
      as a new Agent Note.

## Evidence checklist for every PR in this line

- [ ] Command lines that ran (build, parity, bench), job ids, node.
- [ ] Parity JSON with worst cosine; replay result.
- [ ] Timer backend named; `n`; median/min/p99; gap term.
- [ ] Control measured in the same process; floor tag named.
- [ ] Agent Note updated in the same PR; this checklist's boxes updated.
