# Corrections caught by the loop, and independently confirmed

Cases the loop found in its own earlier conclusions. Same shape as
[human.md](human.md), same failure modes, one extra requirement: the reasoning
that produced the wrong conclusion also produces the retrospective about it, so
nothing is admitted on the loop's own word.

**No entries yet.** The bar below has not been cleared, and an empty file is the
honest state — the alternative is a list of self-assessments with nothing
holding them up.

## Before adding one

Read [README](README.md) for the full rule. In short, an entry needs a
falsifiable artifact rather than a narrative, that artifact has to contradict
the earlier conclusion rather than merely differ from it, and a reader that
produced neither the conclusion nor the retrospective has to confirm both, in a
fresh context, following
[correction-review](../../.agents/skills/correction-review/SKILL.md).

Record the confirmation with the entry: who checked it, what they re-ran, and
what they found. A confirmation that only restates the claim is not one.

## Candidates that have not been through this

Named so they are not lost, and so nobody mistakes their absence for absence of
evidence. Each is a real self-correction from a later session; none has been
independently checked, so none is an entry.

- **Tile sweeps were run against a cache.** Every sweep up to `3950103` reused
  one weight buffer, so after the first iteration it sat in L2 and each tile was
  chosen against a cache the deployed route does not have — the action expert
  walks 18 layers ten times a forward. Re-swept cold, three of nine tiles moved.
  Same mode as *an estimate assumed the wrong memory*, and if it survives review
  it belongs beside that case.
- **The floor model's ceiling was optimistic.** It divides every compute-bound
  site by 100% of the measured tensor peak, which nothing in the route reaches;
  re-derived at the 90% the stack delivers, the floor moves 22.07 → 23.76 ms
  (`2d1b18d`). Same mode as *a ratio was taken against an unchecked
  denominator*.
- **A paired A/B was measured on a contaminated GPU.** A change was committed on
  a comparison that fell inside a window when a stale process held 7.4 GB of the
  card; re-run clean, the comparison reversed and the change was withdrawn
  (`76d90e9`). This one is the most interesting of the three, because the loop
  caught it only by re-running for an unrelated reason.
