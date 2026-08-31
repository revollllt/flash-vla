# Parity — metrics, tolerances, gates

## One metrics module

Five metrics, one implementation: `max_abs`, `mean_abs`, `rms_error`,
`p99_abs`, `cosine_similarity`. They live in one shared module under the
correctness suite (`eval/correctness/metrics.py`; the first task that needs it
creates it by lifting the implementation the existing scripts duplicate).
Parity scripts import it, never re-declare it.

## Inputs

- Both sides consume the SAME tensors. Random weights are deliberate: any
  difference is then an implementation difference. A trained checkpoint adds
  coverage of weight conversion only; use it for that claim, not as the
  default.
- Prefer input scales that exercise the dtype's edges on the kernel's own path
  — softmax and norm reductions are where bf16 surprises hide.

## Tolerance shape

- **Tight where nothing accumulated.** The single-step / single-layer /
  layer-0 comparison is the structural gate: a wrong layout, mask, rotation,
  or weight fold shows up there at full size.
- **Loose but smooth at depth.** Rounding drift compounds per layer, so the
  criterion at depth is a floor plus a maximum per-layer step — a step change
  between consecutive layers is a real bug at that layer, which one aggregate
  number would hide.
- **Padded regions: finiteness only.** `0 * NaN = NaN` survives masks; a NaN
  parked in a padded row will eventually reach real data.

## Gate vs report

A parity result gates promotion only where the comparison is stable:
single-step, single-layer, and full-depth single-pass checks. Multi-step
chaotic maps — a flow loop iterated on random weights — are REPORTED, never
gated: a healthy implementation can sit visibly far from cosine 1.0 in that
slot, so a "failure" there is not evidence of a regression. The contract names
which checks gate and which report. When a number is in dispute, the parity
script that owns the rounding contract is the authority.
