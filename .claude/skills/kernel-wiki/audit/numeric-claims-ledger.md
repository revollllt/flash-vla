# Numeric claims ledger

Scan date: 2026-09-07, `scripts/scan_numbers.py --sm90`: unit-bearing numbers
in the prose of sm90 pages, outside fenced blocks and bracketed tags. Each is
`supported` (a locator backs it), `rule` (a threshold or magnitude stated as
experience, per the page's `confidence`), or `upstream` (an upstream figure
kept as upstream's with its locator).

| Page | Candidate | Status | Evidence |
| --- | --- | --- | --- |
| `wiki/hardware/wgmma.md` | 3x | rule | shared-memory-bound below N=64, relative to `[wgmma.issue.wg.ss]`; measured, E003 |
| `wiki/kernels/deepgemm.md` | 1550 TFLOPS | upstream | README 2025-04-18 news entry, shape unstated; `performance_claims` row |
| `wiki/kernels/flash-attention-3.md` | 740 TFLOPS, 1.2 PFLOPS | upstream | arXiv 2407.08608 abstract; `performance_claims` rows, E006 |
| `wiki/kernels/flashmla.md` | 1460 TFLOPS, 1450 TFLOPS | upstream | KernelWiki page, README benchmark locators |
| `wiki/kernels/fp8-block-scale-gemm.md` | 1550 TFLOPS | upstream | as `kernel-deepgemm` |
| `wiki/kernels/megakernel-forms.md` | 1.5x, 30 us | supported | template 43 STATUS (jit vs aot); template 45 STATUS (10-30 us kernels), E004 |
| `wiki/kernels/sparse-mla.md` | 1450 TFLOPS | upstream | KernelWiki page |
| `wiki/patterns/cold-burst-ceiling.md` | 30 MB, 100 MB | rule | the phase size and the ramp length of `[tma.bw.dev.burst]`, E003 |
| `wiki/patterns/epilogue-bound-short-k.md` | 10x | rule | output bytes per FLOP at short vs long K; campaign note, E005 |
| `wiki/patterns/fusion-latency-chain.md` | 2 us | rule | first TMA frame per task kind; campaign note, E005 |
| `wiki/patterns/isolated-timer-overstates-fusion.md` | 11 us, 8.5 us | supported | gemma backbone note (cold vs in-place library attention), E005 |
| `wiki/patterns/pdl-primary-is-a-resource.md` | 3 us | rule | the size of a launch-bound primary; campaign note, E005 |
| `wiki/patterns/persistent-kernel-timing-artifacts.md` | 2x | rule | isolated vs in-graph overstatement; campaign note, E005 |
| `wiki/patterns/serial-epilogue-owner.md` | 0.6 us, 10 us, 0.2 us | supported | ffn-dr-wide-tile rejected note, E005 |
| `wiki/patterns/serialized-wgmma.md` | 3x, 110 cycles | supported | campaign note; relative to `[wgmma.issue.wg.ss]`, E003/E005 |
| `wiki/patterns/wgmma-tile-n-floor.md` | 3x | rule | as `hw-wgmma` |
| `wiki/techniques/pipeline-stages.md` | 695.43 TFLOPS, 939.61 TFLOPS | upstream | KernelWiki page, tcgen05 tutorial rows |
| `wiki/techniques/producer-fusion-pdl.md` | 1.3x | supported | `[coop.ratio.dev.relaunch]`, E003 |
| `wiki/techniques/reduction-own-task-kind.md` | 128 KB | rule | the fold size at which a parallel reduce wins; campaign note, E005 |

Unresolved: 0.
