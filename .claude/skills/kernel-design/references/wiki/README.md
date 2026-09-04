# sm90 Kernel Wiki — distilled optimization experience, routed by symptom

Consultable optimization knowledge for sm90 kernel work: one idea per
entry, stated as portable experience. The candidate loop in `kernel-design`
consults this table before proposing a move; a human reads it the same way.

## Route by symptom

| You see | Read |
|---|---|
| wgmma mainloop ~3x over `[wgmma.issue.wg.ss]`, no memory-side cause | [c7518-wgmma-serialization](c7518-wgmma-serialization.md) |
| build log filtered to errors only; perf mystery | [c7518-wgmma-serialization](c7518-wgmma-serialization.md) |
| BK > 64 on a row-major tile seems to need many boxes or a transpose | [tma-3d-box-row-major](tma-3d-box-row-major.md) |
| a gemm at tile N < 64 is smem-bound; S = QK^T slow | [wgmma-tile-n-floor](wgmma-tile-n-floor.md) |
| one CTA folds all partials; join waits on the slowest sibling | [reduction-own-task-kind](reduction-own-task-kind.md) |
| per-K scale/dequant done as an in-smem RMW | [scale-on-register-fragment](scale-on-register-fragment.md) |
| partial publish slow; release fence behind scattered stores | [bulk-store-publish](bulk-store-publish.md) |
| deciding whether to fuse; fused kernel not beating the composition | [fusion-economics](fusion-economics.md) |
| ncu: mostly no-eligible-warp, single-digit DRAM+SM on a persistent kernel | [fusion-economics](fusion-economics.md) |
| CTAs idle while a dependency completes | [prefetch-across-dependency](prefetch-across-dependency.md) |
| producers stall on frame reuse; deeper ring does not help | [release-on-retirement](release-on-retirement.md) |
| small kernels chained between heavy stages; launch ramp adds up | [producer-fusion-pdl](producer-fusion-pdl.md) |
| a pre-arranged layout would speed the kernel but must be produced | [layout-production-budget](layout-production-budget.md) |
| TMA reads stale data after a counter wait; smem swizzle offset bugs | [layout-production-budget](layout-production-budget.md) caveats |
| cluster_sync placement, DSMEM reduction, mbarrier scope | [cluster-barrier-placement](cluster-barrier-placement.md) |
| persistent-kernel timing looks absurd; cross-job deltas unstable | [measuring-persistent-kernels](measuring-persistent-kernels.md) |
| short-K GEMM at ~30% MFU; tile sweep flat; cuBLAS faster at the same shape | [epilogue-staging-short-k](epilogue-staging-short-k.md) |
| cold weight stream at ~40-60% of DRAM; ring depth / box / CTA count all null | [cold-burst-ceiling](cold-burst-ceiling.md) |
| a tiny launch-bound kernel sits on a PDL chain and fusing it away regresses | [pdl-primary-is-a-resource](pdl-primary-is-a-resource.md) |
| a fusion wins on a cold isolated timer and is neutral or worse in the graph | [measure-in-the-graph](measure-in-the-graph.md) |
| widening a split-K tile to cut joins made the phase slower | [serial-epilogue-owner](serial-epilogue-owner.md) |
| an ablation says removing a stage/warp/tile costs a lot; you plan to add one | [price-the-direction](price-the-direction.md) |
| every lever in one phase measures null; the last gradient looks like a pole | [stacked-floors](stacked-floors.md) |
| unfamiliar archetype: small-M GEMM / decode attention / megakernel | `ext-deepgemm-sm90`, `ext-flashmla-sm90`, `ext-fa3-pingpong`, `ext-mpk-megakernel` |
| you know the move but not how the choreography is spelled | [`../templates/`](../templates/README.md) — compilable sm90 skeletons |

## Entry format

```yaml
---
id: <file stem>
type: pattern | technique | case | external   # diagnosis / move / worked trade-off / distillation
arch: sm90
tags: [..]
confidence: measured | source-reported | inferred
---
```

Body sections, in order: `Context` (when the entry applies), `Move` (the
rule), then optionally `Why it works` and `Caveats`. External entries add a
`Source` section with original upstream links.

## Rules

- **Entries are distilled experience, never experiment records.** No job
  ids, revision histories, project file citations, or "our kernel X"
  narration. The experiments that established an entry live in Agent Notes
  and per-task workspaces; a project refactor must never cascade into this
  wiki.
- **A number appears only when it is the rule** — a threshold ("tile
  N < 64"), or a typical magnitude stated as experience ("~3x", "worth
  trying when the fold reaches ~128 KB").
- **Machine constants are cited by tag** (`[wgmma.issue.wg.ss]`) and never
  restated — `python3 .claude/skills/hardware-unit-test/scripts/constants.py
  --tag <tag>` is the authority, with the validity range the tag carries.
  The `hardware-unit-test` skill is a portable sibling designed for this
  cross-reference.
- **One idea per entry.** Corrections edit the entry in place
  (current-state only); a reversal gets a new cross-linked entry.
- **Entries come only from established results** — a promoted or rejected
  candidate loop, a measured probe, or an authoritative upstream source —
  never from proposals.

## Origin

The organizational method — symptom-indexed routing, typed one-idea entries,
experience decoupled from any one project's history — is adapted from
[mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki) (MIT), the
knowledge base behind
[kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents).
No content is copied from it: it is Blackwell-first, so its instruction
vocabulary (tcgen05, TMEM, CLC, 2-SM MMA) has no sm90 counterpart. Every entry
here is written for sm90 from first-hand established results.

Two deliberate departures:

- **Its snippets are short verbatim excerpts from upstream files**, checked for
  provenance (byte-identity against a pinned upstream SHA) rather than for
  compilability. `../templates/` instead carries whole skeletons that a
  checker compiles and inspects, because a fragment referencing undeclared
  state teaches structure but cannot be run or trusted.
- **Its full-file corpus is vendored upstream code.** Here that layer already
  exists as `third_party/` submodules, so it is not duplicated.
