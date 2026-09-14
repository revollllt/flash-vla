# Corrections caught from outside the loop

Cases where a reader asked about a number and the conclusion changed. Grouped by
failure mode; see [README](README.md) for what earns a place here and
[the workflow](../optimization.md) for the review list these produced.

All of these come from one engagement — the RTX 5090 bring-up and its first
optimization run, 46.794 → 27.556 ms. Six conclusions were wrong when they were
written and none was caught by the loop. Two had already been filed as settled,
one of them a measured negative on what was then the largest remaining call
site.

## A number was carried, never measured

**LingBot's best was quoted as 64.804 ms.** It was read off the top table of
`results/README.md`, which rendered the retired Campaign controller's legacy
entries with a column called "Current best ms" and no marker that they were
historical. The real figure was 23.497 ms
([lingbot-h100/run-02](../../results/lingbot-h100/run-02/README.md)) — a 2.8x
error, in the direction that made the existing work look worse.

The generator has since been fixed so the caveat cannot be lost again: current
runs come first and the legacy table carries its own heading and disclaimer
(`b314280`). The note that used to say this was hand-written, and the rebuild
overwrote it.

**A smaller instance.** `run-01`'s shape line claimed "768 visual + 200 prompt
tokens" for a route whose every measurement records `prompt_len0`. Nobody had
measured the 200 (`d67659a`).

## A ratio was taken against an unchecked denominator

**`mma.sync` was reported at 60% of peak**, i.e. 40% left on the table. Two
independent errors: the peak came from `spec.py`'s fp16-*accumulate* figure
while the measurement ran fp32 accumulate, and it was computed at the 2.407 GHz
marketed boost while the part runs ~2.89 GHz under load.

The tell was already on the page. A measured 253 TFLOP/s against a "peak" of
209.5 is impossible, and an impossible ratio was reported rather than
investigated. Re-measured clock-free as FLOP/cycle/SM it is 99.9% of 512
(`measured/unit-mma.md`, `40816f3`); there was no 40%.

## An estimate assumed the wrong memory

**Split-KV for the action expert's attention was dismissed on a combine cost of
6.8 µs**, computed by running a 5.2 MB partial buffer through
`[ld.bw.dev.dram]`'s 1524 GB/s. That buffer is L2-resident, and this machine's
own `copy_` measures 4879 GB/s in `lab/sm120/pi0_bandwidth_bench.py` — the bench
that produced the correcting number had been written by the same loop and was
not consulted.

Three times too expensive, and used to reject a design by comparing it against
the main loop it was supposed to shorten.

## A negative outlived its toolbox

**A fused tensor-core attention was measured at 79.5 µs and recorded as
rejected**, with a scaling sweep and a mechanism, and the call site written off.

That kernel predated the fused QKV projection and used none of what it had
established: `ldmatrix` instead of twelve scalar shared loads, one job per warp
instead of one warp per scheduler, and operands staged in their natural order
instead of a scatter-transposed V carrying the exact 8-way bank conflict
`ldmatrix.trans` exists to remove. Rebuilt with all three and numerically
unchanged: **32.3 µs, 2.46x**. With the key axis split eight ways: **18.94 µs
against the split form's 25.2**, and **−0.847 ms deployed** (`c76fe4b`).

What survived was narrower and still true: 408 flat queries is 26 mma M tiles,
and no scheduler invents tiles. The error was concluding from that that the site
had no room, while the key axis sat unused.

**This is the mode worth generalizing.** When the toolbox gains a technique
measured to be worth a factor, every negative recorded before it is provisional
until re-run.

## A limit was declared from one attempt

**`measured/ncu-metrics.md` said profiling was "blocked entirely"** after one
`ERR_NVGPUCTRPERM`. That error gates *counters*; CUPTI activity tracing was
never blocked, and `sudo ncu` works with no module change and no reboot
(`47ee0d5`). The whole timeline method was declared impossible on one error
string.

**The same mode in the source hierarchy.** The sm_120 ISA facts were first taken
from a widely-mirrored community wiki, and both of its load-bearing claims were
false — it says sm_120 keeps `wgmma`, and that clusters are limited to one CTA.
`ptxas` rejects `wgmma` for every sm_120 target, and this part launches an
8-CTA cluster with working distributed shared memory (`[cluster.count.max]`).
The method that replaced it is in `measured/isa-support.md`: ptxas is the oracle
because it is the implementation.

## A dependency was treated as a property of the world

**Programmatic dependent launch was probed, found available on sm_120 and
measured at 1.166x under graph replay** (`lab/sm120/pdl_unit.cu`), then set
aside with an accurate reason: PDL needs the producer *and* the consumer to
carry `griddepcontrol`, and a cuBLAS call between two hand-written kernels
breaks the chain — which here was nearly every pair.

That reason went into the next-steps list as *"it grows as more GEMMs become
hand-written"*, with no step proposed to make that happen. The blocker was named
precisely and left standing.

One instruction then spanned what the loop had filed as two separate items — a
GEMM problem it judged too large, and a technique it judged blocked — by asking
for every cuBLAS call site to be rewritten with CuTe/CUTLASS and PDL built on
top. **−1.785 ms across two iterations, the largest single intervention of the
run** (`fdfe281`, `624ab17`).

Nothing in that analysis was measured wrongly. A correctly identified dependency
was recorded as a property of the world rather than as the next thing to attack.

## A case that produced no check

**Scope was inherited from the previous hardware.** The first plan ported H100
TileLang configurations by reducing `NUM_STAGES` to fit 99 KB of shared memory.
It ran end to end at cos 0.898 with `replay_identical` false, because three of
those configs feed warp-specialised builders where the stages *are* the producer
buffer: at one stage the producer has nothing to fill. A race, not a tolerance.

The instruction that unblocked it — each hardware axis gets its own
implementation and the kernels may be rewritten entirely, torch first for an
end-to-end number, then optimized toward SOL — is now the route's premise rather
than a check, which is why it appears here without one.
