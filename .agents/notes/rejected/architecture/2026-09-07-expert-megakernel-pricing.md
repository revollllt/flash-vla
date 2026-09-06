# Agent Note: the expert megakernel is priced below its entry condition — the chain's remainder is hops, not bytes

Status: rejected — priced below the entry condition (0.76 ms upper bound
against 1 ms), no prototype built. The only mechanism a persistent layer
megakernel adds over the shipped PDL chain is weight-stream continuity, and
this pipeline has already measured that its weight streams are ~83% hidden.

Date: 2026-09-07

## Problem

The [optimization campaign](../../proposed/performance/2026-09-06-optimization-campaign-plan.md)
opened lane D1 to price a Pi0.5 action-expert megakernel and, if the price
justified it, prototype one layer of it against the shipped
[`cuda-pdl` chain](../../implemented/architecture/2026-09-02-decoder-pdl-chain.md).
The lane's entry condition for building was written into the plan: a priced
upper bound of at least 1 ms on the Pi0.5 chunk.

The stage has the headroom to make the question worth asking. One expert
layer-step measures 40.8 to 41.0 us in the graph against a 24.28 us delivery
ceiling summed over its five call sites, so 16.62 us per layer-step and 2.99 ms
per chunk sit above what this machine charges for the bytes. The question is
which mechanism can reach that remainder.

Fusion's answer is narrower than it looks here, because the chain is not an
unfused pipeline. Its boundaries already carry programmatic dependent launch,
and two of its six launches are already persistent task loops.

## Decision

Do not build the megakernel. The priced upper bound is 0.76 ms per chunk
against a 1 ms entry condition, and the two arithmetic framings that clear 1 ms
are each refuted by a job already on record.

### What fusion can add, and only that

Weight prefetch across a dependency is the mechanism. The PDL chain already
performs it at four of its six boundaries, with the overlaps measured in the
chain's own note: 0.92 us for rms to qkv, 6.90 for qkv to attention, 11.70 for
attention to combine, and a role-split wait that releases the FFN's weight
loader. Two edges are uncovered, and they are the whole of the megakernel's
case:

- **combine to the out-projection producer.** The
  [cooperative TileLang producer](../../implemented/architecture/2026-08-28-cooperative-xfs-pdl.md)
  cannot carry a programmatic wait, so its 4.194 MB weight stream starts only
  after the combine grid drains.
- **the FFN to the next layer's qkv.** Nothing prefetches across it.

Launch removal is not a credit. `coop.ratio.dev.relaunch` prices a device-side
relaunch at 1.29x one grid barrier, so the substitution is worth 0.31 us, and
the megakernel wiki records it measuring nothing at 10-30 us kernels.

### The pricing

Weight delivery per layer-step is four separate bursts today. Bursts are priced
at `ld.bw.dev.dram` (1.85 us + MB/2.77) except gate+up, which carries the
declared `tma.bw.dev.burst` ceiling of 9.41 us from job 591174. One continuous
stream is priced at the marginal rate, which is where the cold-burst curve
lands by 268 MB.

| stream | MB | separate burst | inside one stream | continuity credit |
|---|---:|---:|---:|---:|
| qkv | 5.243 | 3.743 | 1.893 | 1.850 |
| out-projection | 4.194 | 3.364 | 1.514 | 1.850 |
| gate+up | 16.777 | 9.410 | 6.057 | 3.353 |
| down | 8.389 | 4.878 | 3.028 | 1.850 |
| total | 34.603 | 21.395 | 12.492 | 8.903 |

| # | basis | us/layer-step | ms/chunk |
|---|---|---:|---:|
| P1 | the table above: every burst's fixed cost removed | 8.90 | 1.60 |
| P2 | P1 with the FFN half capped by its construction bound (job 589178) | 6.09 | 1.10 |
| P3 | P2 with qkv struck, already pre-wait under PDL | 4.24 | **0.76** |
| P4 | the mechanism as built on four of these five stages (job 556021) | ~1.5 | 0.27 |

P1 is an arithmetic ceiling and `tma.bw.dev.burst`'s own rule forbids using it
without first bounding the stream's exposed share. That bound exists for 73% of
the bytes:
[the FFN stream-continuity note](2026-09-03-ffn-stream-continuity.md) freed the
entire GatedUp and DownResidual weight streams by construction and measured
2.39 us per layer against the 14.29 us those two sites' ceilings claim, an
exposed share of 16.7%. Its own alternatives section closes this lane's
mechanism by name: cross-launch continuity across layers has the same ceiling.
P2 applies that. P3 additionally strikes the qkv stream, whose weight mainloop
the PDL chain already runs entirely before its wait.

P3 remains generous twice over. It charges the out-projection stream at 100%
exposure against the pipeline's measured 16.7%, and it grants the FFN half its
whole 2.39 us bound when continuity reaches only the burst-ramp share of it,
5.20 of 14.29 us of delivery ceiling, which is 0.87 us at uniform exposure.

### The debit

Per the fusion-economics rule, four internalized boundaries cost four counter
round trips at `atom.lat.dev.hop` 651 ns, so 2.60 us per layer-step, plus a
first TMA frame per task kind, which the attention task loop's own timeline
measured at 1.2 to 2.2 us. The offsetting launch credit is at most 1.55 us and
`coop.ratio.dev.relaunch` says not to take it. The ledger is a 4.24 us credit
whose components the bounds say are mostly hidden, against a 2.60 us debit that
is not.

### The last unpriced item

The decoder attention split, todo#8 in the Pi0.5 latency loop's queue at
7.65 us for roughly 1.05 MB, was deferred as megakernel scope and is the only
item that was never priced. It is priced here at **zero for continuity**. The
chain already pre-issues the mask and every pure-prefix key and value frame
above its wait, and
[the attention task loop](2026-08-27-attention-block-taskloop.md) already deals
attention onto the CTAs idle during qkv with frame-level dependencies. Its
5.4 us above the delivery ceiling is the true dependency of Q on its head's
slowest qkv tile, plus the split publish and the combine join. That is
dependency-chain latency, which is what this note attributes the stage's
remainder to.

## Alternatives considered

- **Build the five-stage persistent kernel anyway.** Four of the five stages
  already exist in that form in the shared package and were rejected at 1.01x
  against the composition (jobs 556021 and 556329,
  [attention block task loop](2026-08-27-attention-block-taskloop.md)). That
  note's own pricing of what fusion bought there was "a few microseconds at
  best", with cross-op prefetch of the next task's frames worth about 1.5 us
  because the remaining wait is the producer's own tail. The megakernel adds
  the two FFN stages to that, and those two are the shipped persistent loop
  whose streams the construction bound already closed.
- **Cross-layer weight continuity, the half that is genuinely new.** Already
  built through a different surface: candidate c2 of
  [the decoder launch fusions](2026-09-03-decoder-launch-fusions.md) made the
  FFN trigger at entry so the successor grids become resident during
  DownResidual and prefetch their weights and prefix keys early. It measured
  +0.56 ms on the chunk on two nodes, jobs 588825 and 588826. The mechanism is
  a resource collision, and
  [the GatedUp copy column](../../proposed/architecture/2026-09-03-ffn-gu-copy-column.md)
  explains it: that phase runs within 1.6% of this machine's cold-DRAM delivery
  rate for its geometry (job 591174), so bytes injected into it are added
  rather than hidden. Cross-layer continuity's entire plan is to stream the
  next layer's weights during exactly that phase.
- **The interpreter-style megakernel** (reference template 40). Already
  measured losing to plain launches at every depth. Not repeated.
- **Fuse only the FFN half**, producer and task loop into one launch. It is the
  half whose route constraints permit it, but it is also the half the
  construction bound covers: its two streams free by construction are 2.39 us,
  and the producer-to-FFN edge is the one boundary that already has a
  role-split wait. Not built.
- **Read the stage's 2.99 ms of headroom as the megakernel's upper bound.**
  Rejected as a category error. That headroom is measured against a delivery
  ceiling; a megakernel changes when bytes are fetched, not how many, so it may
  claim only the exposed share of the delivery it moves.

## Consequences

- **The Pi0.5 action expert's remaining headroom is dependency-chain latency,
  not delivery and not launch structure.** Of 40.9 us per layer-step, 24.28 is
  the summed delivery ceiling and 16.62 us is above it. The items that make up
  that remainder are each priced and each small: the qkv split join, the
  attention split publish and the combine join, and DownResidual's readiness
  poll, join wait, fold reads and residual read-modify-write at 1.05, 0.96,
  1.37 and 0.69 us. None is a delivery problem and none has yielded to
  reordering.
- **The megakernel direction is closed for this component**, and with it the
  last item the Pi0.5 latency loop deferred to it. Every optimization family
  named for the expert is now either shipped or rejected with a mechanism.
- **The reopening condition is a number.** Continuity's arithmetic credit is
  8.903 us per layer-step; clearing 1 ms on the chunk needs 5.56 us of it, so
  it needs the weight streams' exposed share to exceed **62%** against a
  measured 16.7%. Two things could produce that. A change that makes the FFN's
  weight bytes cheaper would lift the phase off the cold-DRAM floor and let its
  roughly 4 us ring-protocol floor bind instead, which is the entry condition
  the copy-column note already wrote; that is a numerics or model decision, not
  a kernel one. Or a mechanism that removes hops rather than bytes, for which
  the megakernel is the wrong instrument, since it adds 2.60 us of them.
- The shared [`gemma_expert` package](../../implemented/architecture/2026-09-07-gemma-expert-package.md)
  is unchanged. No route, plan, kernel or contract moves because of this note.

## Verification

No candidate was built and no GPU job was spent by this lane, so there is no
parity result, no benchmark leg and no gate verdict. Every number cited is from
a job already recorded elsewhere:

| number | job | node | owning note |
|---|---|---|---|
| chain 40.8-41.0 us per layer-step, stage 7.399 ms | 599747 | ACD1-58 | [Pi0's expert chain priced out](2026-09-07-pi0-expert-cuda-chain.md) |
| per-site delivery ceilings, 24.28 us per layer-step | 598964 | ACD1-55 | the floor report |
| both FFN weight streams free = 2.39 us per layer | 589178 | ACD1-11 | [FFN stream continuity](2026-09-03-ffn-stream-continuity.md) |
| GatedUp within 1.6% of the cold-DRAM delivery rate | 591174 | ACD1-32 | [GatedUp copy column](../../proposed/architecture/2026-09-03-ffn-gu-copy-column.md) |
| four of five stages fused = 1.01x | 556021, 556329 | ACD1-2, ACD1-32 | [attention block task loop](2026-08-27-attention-block-taskloop.md) |
| early successor grids = +0.56 ms on the chunk | 588825, 588826 | ACD1-1 | [decoder launch fusions](2026-09-03-decoder-launch-fusions.md) |
| PDL boundary overlaps 0.92 / 6.90 / 11.70 us | 583761 | ACD1-58 | [decoder PDL chain](../../implemented/architecture/2026-09-02-decoder-pdl-chain.md) |

Machine constants are the sm90 measured table's `ld.bw.dev.dram`,
`tma.bw.dev.burst`, `atom.lat.dev.hop`, `coop.lat.dev.sync`,
`coop.ratio.dev.relaunch` and `launch.lat.dev.ramp`. The full pricing, the
tensor table and the four-row ladder with its assumptions are in the lane's
workspace contract at `artifacts/ktasks/expert-megakernel/`.

## Related notes

- [decoder PDL chain](../../implemented/architecture/2026-09-02-decoder-pdl-chain.md):
  the incumbent, and the reason three of the four prefetch edges are already
  covered.
- [cooperative XFS producer and PDL](../../implemented/architecture/2026-08-28-cooperative-xfs-pdl.md):
  the producer whose launcher cannot carry a wait, which is one of the two
  uncovered edges.
- [FFN stream continuity](2026-09-03-ffn-stream-continuity.md) and
  [the GatedUp copy column](../../proposed/architecture/2026-09-03-ffn-gu-copy-column.md):
  the construction bound and the machine pairing that cap this lane's credit.
- [attention block task loop](2026-08-27-attention-block-taskloop.md) and
  [decoder launch fusions](2026-09-03-decoder-launch-fusions.md): the two
  halves of this experiment, already run.
- [the Gemma expert package](../../implemented/architecture/2026-09-07-gemma-expert-package.md)
  and [Pi0's expert chain priced out](2026-09-07-pi0-expert-cuda-chain.md):
  lane D0, which supplied the per-layer-step number this note prices against.
- [the optimization campaign plan](../../proposed/performance/2026-09-06-optimization-campaign-plan.md):
  lane D1's charter and its 1 ms entry condition.
