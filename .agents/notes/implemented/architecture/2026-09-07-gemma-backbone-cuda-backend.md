# Agent Note: the shared Gemma backbone CUDA backend

Status: implemented

Date: 2026-09-07

## Problem

Both H100 Targets run the same Gemma backbone -- 18 layers, width 2048, FFN
16384, eight query heads over one KV head at head_dim 256 -- and only the
prefix row count differs (968 with Pi0.5's 200-token prompt, 768 with Pi0's
empty one). They ran it with different implementations anyway. Pi0.5's prefix
attention was a fused CUDA kernel living in that Target's `backends/cuda/`;
Pi0's was a four-launch torch chain that also copied its result. Pi0's output
projection was a TileLang body where Pi0.5's was cuBLAS. A kernel written for
one Target could not be reached by the other.

The segment is also the campaign's second-largest headroom
([campaign plan](../../proposed/performance/2026-09-06-optimization-campaign-plan.md)):
6.75 ms on Pi0.5 and 5.95 on Pi0 against measured ceilings of 4.46 and 3.54.

## Decision

1. **A shared component package**, `hardware/nvidia/h100/gemma_backbone/`,
   holding the backbone's CUDA backend, its ABI-mirror references and its
   kernels, registered by both Targets as `gemma-cuda`. It imports `models/`,
   `runtime/` and the shared tile library and no Target, per
   [device component packages](2026-09-07-device-component-packages.md).
2. **No route constraint, and no `eval/smoke.py` change.** Every wrapper reads
   and writes the graph's own buffers in the layout the TileLang route uses --
   in particular Q stays token-major rather than the head-major scratch the
   action expert's CUDA pair keeps -- so each call site routes independently
   and the route check needs no per-Target oracle for this backend.
3. **Pi0's shipped plan takes the prefix attention and the output projection.**
   The attention is the kernel Pi0.5 already shipped, whose source moved here
   byte-identically; the output projection is cuBLAS `addmm_`, which Pi0.5 has
   used since [prefix GEMM epilogues](2026-09-03-prefix-gemm-epilogue.md).
4. **Pi0's FFN down projection stays on TileLang.** cuBLAS is slower there.
5. **Pi0.5's shipped plan takes its backbone attention from the package**, so
   the Target's own copy of `enc_attn` is now unreferenced and can be retired.
6. **A persistent warp-specialized gated FFN was built and is not shipped.**
   `kernels/gated_ffn.cu` stays in the tree, correct and unrouted, with its
   ablation switches; the rejection and what a successor should start from are
   below.

## Alternatives considered

- **cuBLAS for Pi0's FFN down projection.** Rejected on measurement, against
  an explicit prediction of 0.415 ms. See Consequences: it is 6.61 us per call
  slower in the graph, and two isolated timers got the sign wrong.
- **A cuBLAS route for the gated FFN**, which the campaign named as the branch
  to take if the library reached 90 % of the ceiling. Rejected: at the Pi0.5
  shape two GEMMs plus a fused multiply measure 767 us and the cuBLASLt
  GELU form 397 us, against 224 for the TileLang fused body.
- **Fusing the RMSNorm into the projection's A operand.** Rejected: whole-row
  statistics over K=2048 exceed shared memory, and the normalized activation
  must be rounded to bf16 before the projection, an ordering
  [the QKV epilogue note](../feature/2026-09-02-encoder-qkv-rope-epilogue.md)
  records as not interchangeable. The separate pass already runs at ~83 % of
  `[ld.bw.dev.dram]`, so the whole prize is one launch ramp.
- **Fusing the gated FFN with the down projection.** Rejected: the hidden
  buffer is 31.7 MB and cannot stay on chip, and Pi0.5's down projection
  already measures 110 % of its wgmma ceiling.

## Consequences

- Both Targets reach one backbone implementation. Pi0's backbone segment loses
  51 launches per forward (192 to 141) and all 178 us of its device copies.
- **Wave quantization is what makes a per-row prediction wrong across these two
  Targets, and it is worth stating as a rule.** At a 128x128 tile the N=2048
  sites make 8 x 16 = 128 tiles on Pi0.5, 0.97 of a 132-SM wave, and 6 x 16 =
  96 on Pi0, 0.73 of a wave with 36 SMs idle. Measured efficiency follows
  exactly: Pi0.5's cuBLAS down projection reaches 714 TFLOP/s and Pi0's 568,
  where 0.84 x 0.73/0.97 predicts 0.63 against 0.67 measured. Pi0.5 lands on
  one wave by accident of its prompt length. Carrying its nanoseconds-per-row
  to Pi0 predicted a gain the machine cannot give, and no tile with BN a
  multiple of 64 lands Pi0's count near 132 or 264. The tile library has no
  split-K primitive, so this is a named blocker for Pi0's two N=2048 sites.
- **An isolated timer can get the sign of a route swap wrong, not just its
  magnitude.** For Pi0's FFN down projection, `--timer cupti` read +0.4 us per
  call and `--timer cudagraph` read -5.1, while the in-graph subtraction read
  **+6.61**. In the pipeline the 25 MB hidden buffer this reads was written by
  the gated FFN one kernel earlier and is L2-resident; the TileLang body
  carries `SWIZZLE=8` to rasterize for that reuse and cuBLAS chooses its own
  order. This extends [measure in the graph](../../../.claude/skills/kernel-design/references/wiki/measure-in-the-graph.md),
  which warns about magnitude. Every route decision in this lane after that
  point was made by subtracting `benchmarks profile` on both plans in one job.
- **`benchmarks kernels` could not run a Pi0 segment at all** until the fix
  now on main: Pi0's `llm_backbone_embed_prompt` copies a zero-row tensor at
  `prompt_len=0`, issues no kernel, and made the CUPTI timer raise.
- The floor report's segment `measured` column is an attributed number and is
  not the latency harness's. Read one run's `wall_us` against its own
  `total_us`: the backbone's inter-kernel gap is 67 us on Pi0.5 and 103 on
  Pi0, 1 to 2 % of the segment and near the launch ramp. An earlier reading of
  620 us came from crossing two instruments and is withdrawn.
- `benchmarks/kernels.py` needs no built-in case: it derives its cases from the
  recorded invocations of whatever the plan routes. No new tile primitive was
  added, so `eval/tile_sm90` needs no case.

### The gated FFN kernel, and why it is not shipped

The call site is 56.6 % of Pi0.5's backbone kernel time at 74 % of
`[wgmma.clock.sm]`, and unlike the N=2048 sites it is not wave-quantized
(1024 tiles on Pi0.5, 768 on Pi0, a 3 % tail). A persistent warp-specialized
dual-GEMM was built for it: 128x128x128 tile, two math warpgroups stacking
along M because two live f32 accumulators are 256 registers per thread over
one, three 32 KB rings filling 224 KB of the 227 KB maximum, one 3-D TMA box
per operand per K-step, 168 registers and no spills.

**Its mainloop succeeded and its epilogue sank it.** Ablation columns, one
rebuild each in one job, isolated cudagraph timer at the Pi0.5 shape:

| column | us/call |
|---|---:|
| copy ring + math, no epilogue | 157 |
| everything | 318 |
| incumbent TileLang body | 205 |

157 us against a 152.85 us wgmma floor is 97 % of the tensor core, where the
incumbent sits at 74 %: 3-D boxes and persistent scheduling did exactly what
they were meant to. The epilogue then adds 161 us, half the kernel. Two store
mechanisms -- a staged shared tile with a TMA store, and bf16 pairs written
straight from the accumulator -- land within 10 us of each other, so the store
mechanism is not the cause.

What a successor should start from: the mainloop, unchanged, and the
observation that a persistent CTA pays its epilogue 7.76 times where a
tile-per-CTA kernel pays it once. The "stage the C tile through shared memory"
finding of the two GEMM epilogue notes was established on non-persistent
kernels and does not carry over. The remaining structural answer is to overlap
tile *t*'s epilogue with tile *t+1*'s mainloop, which needs either a second
accumulator set (no registers are left) or a dedicated epilogue warpgroup,
whose cost `ext-fa3-pingpong` prices as smem staging plus a sync.

## Verification

Login node: `python -m eval.smoke` passes for both Targets, eleven candidate
plans and both route oracles;
`grep -rn "h100\.pi0\b\|h100\.pi05" src/flash_vla/hardware/nvidia/h100/gemma_backbone`
is empty.

GPU, all jobs from this lane's worktree, clocks unlocked:

- **Kernel parity** (T2 ABI mirrors, job 599788, ACD1-58): attention rel_rms
  2.05e-3 / cosine 0.9999979 on Pi0 and 2.20e-3 / 0.9999976 on Pi0.5; output
  projection 6.64e-4 / 0.9999998 and 5.96e-4 / 0.9999998; against gates of
  6.6e-2 and 0.99978.
- **Bit identity of the kernel move** (job 599788): `lab.stage_dump compare`,
  Pi0.5 shipped against the package route, one node one seed, every declared
  stage output `torch.equal`.
- **In-graph attribution**, both plans in one job (599832, ACD1-8): Pi0's
  attention 24.92 to 17.74 us per call, output projection 18.96 to 11.88,
  down projection 91.98 against 85.37 (the swap that was dropped), segment
  wall 5726.7 to 5552.3 us, copies 178.3 to 0, launches 192 to 141.
- **Same-process A/B/A against shipped**, 100 reps: chunk `min` -0.096 ms on
  ACD1-58 with a 0.010 ms control spread (job 599788) and -0.111 ms on ACD1-8
  with 0.052 (job 599832), both before the down projection was dropped.
- **Promotion gate** (job 600111): `eval.gate --baseline --reps 100` returns
  `pass` on both Targets. Pi0 under `improve`, chunk `min` -1.574 ms against
  the reference route with a 0.026 ms control spread, tail 0.276 ms inside the
  0.5 ms bound, `baseline_layer0` passed. Pi0.5 under `no-regression`, which
  is the right mode for a change proven bit-identical, no regressions, tail
  0.228 ms, `baseline_layer0` passed. Evidence records are named in the log.
- **Not obtained**: the Nsight Compute capture of the incumbent gated FFN.
  Job 599809 passed `--launch-count 2` with no name filter, which counts from
  process start and profiled torch's weight-initialisation kernels instead;
  the corrected capture was still queued when this lane stopped.

## Related notes

- [device component packages](2026-09-07-device-component-packages.md): the
  contract this package satisfies and the dependency rule it keeps.
- [prefix GEMM epilogues](2026-09-03-prefix-gemm-epilogue.md): why the two
  residual sites are cuBLAS, and the staged-store finding that does not carry
  into a persistent kernel.
- [a hand-written SM90 GEMM for the short-K call sites](../../rejected/architecture/2026-09-03-sm90-short-k-gemm.md):
  the one-wave finding this note measures across two Targets of one kernel,
  and the persistent multi-tile successor it named, now built and priced.
- [fused encoder MQA attention](../feature/2026-09-03-encoder-mqa-cuda-attention.md):
  the kernel whose source this package now owns.
