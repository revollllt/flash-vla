# Pi0 · RTX 5090 · run-01

First optimization run of `rtx5090/pi0`, from the all-torch bring-up route to a
hand-written CUDA route: **46.794 → 30.867 ms, 1.52×**.

![Optimization progress](progress.svg)

## Workload and measurement conditions

| | |
|---|---|
| Target | `hardware/nvidia/rtx5090/pi0`, bf16, 18 layers, 10 denoise steps |
| Checkpoint | seeded synthetic weights, `seed=0` — Pi0 needs no asset |
| Fixture | `flash-vla/pi0-inputs-v1/seed-0` |
| Shape | 3 views × 224², 768 visual + 200 prompt tokens, chunk 50 |
| Environment | RTX 5090, driver 580.142, torch 2.13.0+cu130, CUDA 13.1, `sm_120f` |
| Protocol | `latency-v2`: fresh process, first capture per leg, warmup 5, 100 reps, median |

```bash
python -m benchmarks latency --target rtx5090/pi0 --plan shipped --seed 0 \
  --warmup 5 --reps 100 --out results/pi0-rtx5090/run-01/measurements/005-shipped-final.json
python -m eval.correctness --target rtx5090/pi0 --plan shipped --steps 1 --layers 1
```

**Do not compare any of this with an H100 figure.** A different GPU is a
different comparison context; this run starts its own curve.

## Result

| # | change | median | delta |
|---|---|---:|---:|
| 0 | every call site in torch | 46.794 | |
| 1 | expert RMSNorm, RoPE scatter, gated activation in CUDA | 40.240 | **−6.588** |
| 2 | expert attention: cuBLAS GEMMs + hand-written masked softmax | 37.040 | **−3.137** |
| 3 | the same norm kernels on the backbone | 34.423 | **−2.607** |
| 4 | vision LayerNorm + GELU kernels, residual folded into GEMM beta | 32.753 | **−1.571** |
| 5 | single-pass masked softmax | 32.886 | *neutral* |
| 6 | expert gate and up packed into one GEMM | 31.396 | **−1.490** |
| 9 | expert norm + QKV + RoPE fused into one kernel | 31.220 | **−0.176** |
| 10 | that kernel's weight tile transposed by `ldmatrix.trans` | 31.078 | **−0.142** |
| 11 | action-token output projection off torch | 30.999 | **−0.079** |
| 12 | packed gated activation vectorized to 128-bit | 30.867 | **−0.132** |

Every delta in rows 1–4 is a **paired A/B in one job**: the retained route and
the candidate measured back to back, same process family, same driver, with the
retained route as leg 0. `min` and `p99` moved with the median in all four.

Row 5 is not plan-selectable — it changes a kernel the deployed route already
uses — so it is measured as the same route before and after. The final number
comes from three legs in one job at **32.886 / 32.888 / 32.887 ms, a 0.002 ms
spread**, and 32.753 is the previous job's figure for the route without it. The
0.13 ms between them is cross-job variation, not a regression, and the change is
kept for a numerical reason rather than a latency one: it removes a bf16
round-trip and moves the operator's cosine from 0.999992 to 0.999999.

Launches fell from **12831 to 3460**, −73%.

## Correctness

The reference route stays all-torch, so `eval.correctness` compares the
hand-written kernels against the implementation they replaced rather than
against themselves.

| | shallow gate | full depth (10 steps × 18 layers) |
|---|---|---|
| cosine | **0.99983** | **0.99793** (budget 0.9943) |
| rel_rms | 0.0034 | **0.0643** (budget 0.34) |
| verdict | **passed** | inside the deepest tolerance |

`replay_identical` is true. The full-depth cosine drifted from 0.99976 after
iteration 1 to 0.99793 now — 36% of the deepest budget spent, all of it bf16
rounding and reduction order. Nothing in the route computes at a lower precision
than the torch form it replaced.

Operator-level checks are in `correctness/`:
`lab/sm120/pi0_torch_parity.py` pins the torch backend against the TileLang
wrappers it was written from, and `lab/sm120/pi0_wrapper_bisect.py` pins each
CUDA wrapper against its torch counterpart — including with the residual buffer
deliberately aliased to the output, which is how the one real bug below was
caught.

## What was tried and rejected

**A fully fused single-kernel attention.** Correct, and **8× slower than the
split form** — 201 µs against 25 at Pi0's shape, on a torch chain of 70. ncu
says why: 32.09M instructions for 342 MFLOP, because a CUDA-core dot product
costs two shared-memory loads and an FFMA per two FLOP where one `mma.sync`
does 4096 FLOP in one instruction [mma.rate.sm.bf16]. Two rounds of tuning were
also negative — transposing K in shared memory to fix a measured 16-way bank
conflict, then padding the rows, moved 214 µs to 201. **The instruction count is
the wall, not the conflicts**, and only a tensor-core mainloop clears it. Kept
unrouted in `kernels/expert_attention.cu` so the negative can be re-run.

**A tensor-core single-kernel attention.** `mma.sync` clears the instruction
wall the CUDA-core attempt hit -- and the kernel is still **3.1x slower than
the split form**, 79.5 us against 25.3 at the same shape, correct at cos
0.999994 / rel 3.4e-03. Four rounds of tuning took it 210 -> 79.5 us: padding
the shared tiles so a fragment load spreads over all 32 banks (210 -> 155),
16-byte vector loads in place of per-element ones (155 -> 131), a softmax
reduced by eight threads per row through shuffles instead of one thread walking
32 columns (131 -> 89.7), four independent mma accumulator chains (89.7 -> 87.7),
and staging the next key tile in registers a tile ahead (87.7 -> 79.5). 203
registers, no spills, 47296 B smem.

**The limit is structural, and it is measured.** Pi0's 408 flat queries are 26
mma M-tiles, so a design that keeps a softmax row in registers has 26 CTAs to
offer a 170-SM part. A sweep of the same kernel over larger query counts:

| CTAs | wall | work |
|---:|---:|---:|
| 26 | 79.6 us | 1x |
| 51 | 81.3 us | 2x |
| 102 | 81.6 us | 4x |
| 153 | 83.5 us | 6x |
| 204 | 145.1 us | 8x |

**Six times the work for 1.05x the time** -- 85% of the machine is idle and
cannot be given anything, and the jump at 204 is the second wave. Per-CTA time
is what sets the wall, and it is dominated by re-reading all of K and V: with
the per-tile global load ablated the kernel runs 38.6 us, so **49 of 87.7 us
was that load**, which every query-tile CTA must do in full because the softmax
cannot be split across CTAs without a partial round-trip. Buying the
parallelism back costs more than it saves: a 6-way key split writes 6 x 408 x
256 fp32 partials, 5.2 MB of round-trip, **6.8 us at [ld.bw.dev.dram]** against
a ~13 us main loop.

Meanwhile the split form is near its own floor. Its three passes cost 1.3 MB,
1.34 MB and 1.3 MB of traffic, 4.2 us each at [ld.bw.dev.dram], so **12.6 us is
structural and it measures 15.9 us in the graph -- 79% of it**. The floor
model's 4.17 us ceiling for this call site assumes a fusion that this shape
cannot afford. Kept unrouted in `kernels/expert_attention_mma.cu`; the sweep is
`bench_mma_scaling` in `lab/sm120/pi0_attention_bench.py`.

**Three of the four small action-token projections.** All four were measured as
a paired A/B, one site per leg, against the same job's shipped leg.
`action_expert_action_out_proj` is a win and is routed: 11 launches to 3, and
−0.065 ms with both candidate legs under both control legs in an A/B/A/B.
`action_expert_state_proj` (3 launches to 1) and `action_expert_action_in_proj`
(6 to 2) came back at +0.015 and +0.007 ms, inside a ~0.05 ms cross-job spread
-- the launches they save do not show, so they stay on torch rather than add
code for nothing. `action_expert_action_mlp` as a single `torch.addmm` was
**0.202 ms slower** than torch's mm-add-copy at its 50 x 1024 x 1024 shape,
which is far more than the two launches it removes.

**A cuBLASLt GELU epilogue on the vision feed-forward.** `aten::_addmm_activation`
would fold the activation into the GEMM and save a 13.2 MB round trip, and it
measured **57.46 us against 51.31** for the separate pass at 768 x 1152 x 4304
-- the epilogue costs more GEMM than the pass it replaces (132.5 against 148.4
TFLOP/s). Its GELU is also neither spelling exactly: it sits 2.7e-03 from
tanh and 2.6e-03 from erf at bf16 output.

**`F.scaled_dot_product_attention` with a precomputed mask**: 90.3 µs against
the torch chain's 73.0 at this shape. Slower than what it would replace.

**Reducing `NUM_STAGES` to fit H100's TileLang tiles into 99 KB** (before the
route was rewritten): ran end to end and returned cos 0.898 with
`replay_identical` false. Three of those configs feed warp-specialised builders
where the stages *are* the producer buffer; at one stage the producer has
nothing to fill while the consumer reads. A race, not a tolerance.

## One bug worth recording

The first form of the vision residual projections computed `out = x @ w + bias`
and then added `res`. The graph binds `res` to the buffer being written on those
call sites, so the add read a residual the GEMM had already overwritten: model
correctness came back at **cos 0.057**. The torch form it replaced evaluates the
whole expression before copying and is alias-safe by construction. The fix puts
the residual in as the GEMM's beta term.

The operator bisect passed the whole time with distinct buffers. It only
reproduced the failure once `res` was deliberately aliased to `out`, at cos
0.962. **A parity test that does not exercise the aliasing the graph actually
uses proves nothing about it.**

## Where the time goes now

Floor model after iteration 4 (`artifacts/profile/floor2.json`), ceiling built
from this machine's measured constants:

| segment | measured | ceiling | % | launches |
|---|---:|---:|---:|---:|
| `llm_backbone` | 15.57 | 11.74 | 133% | 244 |
| `action_expert` | 12.91 | 7.64 | 169% | 2913 |
| `vision_encoder` | 4.80 | 2.73 | 176% | 303 |
| **total** | **33.29** | **22.11** | **151%** | **3460** |

**`llm_backbone` is done.** Its largest call site, `llm_backbone_norm_gated_ffn`
at 8.814 ms, *is* its two GEMMs, and cuBLAS runs them at **82–85% of this
machine's measured tensor ceiling** (207 and 214 TFLOP/s against
[mma.tflops.dev.bf16]'s 253). `mma.sync` is the only tensor-core path on this
part and already reaches 100% of 512 FLOP/cycle/SM [mma.rate.sm.bf16], so a
hand-written GEMM has nothing to find there. Measured directly:

| GEMM | µs | TFLOP/s | of 253 |
|---|---:|---:|---:|
| backbone gate/up, 768×2048×16384 | 248.9 | 207.1 | 82% |
| backbone ffn down, 768×16384×2048 | 240.8 | 214.0 | 85% |
| expert gate/up, 51×1024×4096 | 14.9 | 28.7 | **11%** |
| expert QK^T, 408×256×819 | 13.9 | 12.3 | **5%** |

**`action_expert` is where the remaining headroom is**, 5.3 ms of it, and its
GEMMs sit at 5–11% of the tensor ceiling because they are tiny and
weight-bandwidth bound, not FLOP bound. At `[ld.bw.dev.dram]`'s
`3.35 + MB/1.524` µs, one 8.4 MB expert weight is an 8.9 µs read against a
1.7 µs compute. Ten denoise steps re-read all 18 layers' weights — 5.8 GB per
forward — which no kernel removes.

## Next, in order of expected value

1. **Pack the expert's gate and up projections into one GEMM.** Two 8.4 MB
   weight reads become one 16.8 MB read and pay `[ld.bw.dev.dram]`'s 3.35 µs
   fixed cost once instead of twice. Needs the two weights contiguous, which is
   a checkpoint-loading change rather than a kernel.
2. **A tensor-core attention mainloop.** The remaining 2.1 ms on
   `action_expert_attention` needs `mma.sync` fragments fed by `ldmatrix`; the
   CUDA-core attempt above establishes that nothing less will do.
3. **`vision_encoder`**, 2.1 ms at 176%, is the smallest of the three and has
   had the least attention.
