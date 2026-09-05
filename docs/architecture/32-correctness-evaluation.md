# Numerical Correctness Evaluation

Status: partial. The five metrics live in one module
(`eval/correctness/metrics.py`) that every parity script imports; thresholds
come from the acceptance registry keyed by precision policy; engine-level
reports carry the identity block. The in-engine tier has a generic runner
(`python -m eval.correctness.in_engine`) that compares any candidate plan or
table option against the Target's reference, segment by segment over the
outputs the Target declares, with oracle injection, per-layer profiles,
padded-allocation finiteness and replay determinism. The official-baseline
tier stays per model (its adapters are model-specific) and is registered per
Target in the acceptance registry.

## Levels

1. **Tensor level**: named stage outputs and the final action chunk, compared
   against an oracle on identical inputs.
2. **Policy quality**: task success in an environment. Out of scope this
   phase; the acceptance registry keeps its slot empty.

Correctness is independent of latency and of profiling. It is measured on the
captured engine, not on an eager re-implementation, so that what is validated
is what is deployed.

## Metrics

One shared module implements five metrics; parity scripts import it and
never re-declare them:

`max_abs`, `mean_abs`, `rms_error`, `p99_abs`, `cosine_similarity`.

All five are reported for every comparison; a threshold may be set on any.
Thresholds are framework defaults keyed by precision policy and live in the
acceptance registry; a Target overrides one only with a stated reason.
Root-mean-square error is the reported form of mean squared error. Cosine and
RMS are dominated by the largest-magnitude channels, and an action chunk mixes
channels of different scale, so `max_abs` and `p99_abs` are always reported
beside them.

## Oracle tiers

| Tier | Oracle | Inputs | Catches |
|---|---|---|---|
| in-engine reference route | the same engine on the Target's reference plan | random weights, seeded inputs | kernel errors, wiring, aliasing, route constraint violations |
| official baseline | the upstream implementation through an adapter under `eval/baselines/` | random weights by default; a trained checkpoint only to validate weight conversion | layout, RoPE, masks, folding, conversion |
| policy quality | environment task suite | trained checkpoint | whether a numerical change alters behaviour; out of scope this phase |

Random weights are the default on purpose: both sides consume the same
tensors, so any difference is an implementation difference. A trained
checkpoint adds coverage of conversion only and is used for that claim.

## Comparison structure

A harness that omits any of these produces a wrong verdict, not a weaker
one.

1. **Shallow gates, deep reports.** The structural gate is the single-step,
   single-layer (or layer-0) comparison, where nothing has accumulated and a
   wrong layout, mask, rotation or fold shows at full size. Deeper runs are
   reported. A multi-step denoising loop on random weights is a chaotic map:
   any two implementations that are not bit-identical separate, so it is never
   gated.
2. **Truncate both sides.** When depth is cut for bisection, the oracle is cut
   to the same depth.
3. **Stage isolation by oracle injection.** To measure one stage, its upstream
   inputs are taken from the oracle and written into the Target's declared
   stage outputs through the engine protocol, so upstream drift does not mix
   into the number. The full end-to-end comparison is read after the isolated
   one passes, never instead of it. A Target declares each segment's outputs
   by buffer name with the axis that indexes layers where one exists; a
   contract region of a larger buffer (the prefix rows of a KV cache) is
   declared as a named alias view so the harness never learns the layout.
4. **Per-layer smoothness.** Where a stage repeats a layer, the metric profile
   is reported per layer: tight at layer 0, a floor at depth, and a maximum
   step between consecutive layers. A step change is a bug at that layer,
   which one aggregate number hides.
5. **Padded regions: finiteness only.** Padded rows are compared for
   finiteness, never for value; a NaN parked in padding survives masks and
   reaches real data later.
6. **Replay determinism.** Two replays of the candidate on the same inputs are
   bit-identical. This is a gate; it is the cheapest detector of races in
   persistent and cooperative kernels.
7. **Layout transforms belong to the comparison.** Where the Target and the
   oracle store a tensor in different but equivalent layouts, the parity
   script that owns the rounding contract applies the transform and documents
   it; that script is the authority when a number is in dispute.
8. **Same inputs, seeded, at the dtype's edges.** Both sides consume identical
   tensors from a stated seed; input scales exercise the reductions where the
   dtype's rounding hides.

## Report schema

One JSON document per comparison:

- `identity`: as in the latency report, plus the oracle tier and the oracle's
  version or checkpoint.
- `config`: steps, depth, injection point, weights (random seed or
  checkpoint), prompt.
- `comparisons`: per named tensor, the five metrics; per-layer profiles where
  applicable; finiteness of padded regions; replay determinism.
- `mode`: which comparisons are gates and which are reports, copied from the
  acceptance registry so the report is self-describing.
- `verdict`: absent; produced by the promotion gate.

## Ownership

- Parity scripts live under `eval/correctness/<model>/` and belong to the
  Target; each names the rounding contract it owns.
- Baseline adapters live under `eval/baselines/` and expose the canonical
  inputs, the final output and named stage tensors without becoming a
  dependency of production code.
- The generic in-engine runner executes the in-engine checks of the
  acceptance registry through the engine protocol; the Target contributes
  its factory, its declared stage outputs and, for the baseline tier, its
  adapters and scripts. The promotion gate (planned) sequences both tiers.
