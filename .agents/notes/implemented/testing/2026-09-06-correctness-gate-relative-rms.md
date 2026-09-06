# Agent Note: the correctness gate is relative RMS and cosine, calibrated

Status: implemented

## Problem

Every correctness gate read one number, the cosine similarity, against a
threshold someone had typed. Cosine is a direction: it is dominated by the
largest channels and blind to a uniform scale error, and an action chunk mixes
channels of different scale. The thresholds had no source: 0.999 for a shallow
run, 0.9999 for layer 0, 0.99 at full depth, chosen by feel.

Two defects sat underneath. The metrics were computed in float32, and the
float32 cosine of a million-element tensor with itself reads 0.99994, below
the layer-0 threshold, so a gate at that size would have failed identical
tensors. And the tolerances were read for `bf16` by name in three scripts
rather than for the precision policy the runner reports, so a second policy
could not have been gated at all.

## Decision

1. **Two gates, both must hold.** `rel_rms` (root-mean-square error over the
   reference's root-mean-square: scale-free, every channel weighted by its
   error) beside `cosine_similarity`. `eval/metrics.py` reports six metrics;
   `rel_rms_max` and `cosine_min` gate, the rest are reported beside them.
2. **Float64.** Both tensors are compared in float64 after flattening, so a
   threshold measures the kernel and not the accumulator.
3. **Tolerances are calibrated.** `eval/calibrate.py` reads the natural
   dispersion between the two promoted routes of a Target (the shipped plan
   against the reference plan: bf16 reduction order is the only difference)
   from `eval.correctness` reports at the gate's own depth, and proposes ten
   times the worst dispersion. The registry records the run its numbers came
   from (`CALIBRATION_SOURCE`). A new Target calibrates the same way before
   it can be gated.
4. **Tolerances are keyed by depth and precision.** `tolerances(precision)`
   returns `shallow`, `layer0` and `deepest` pairs plus `max_cosine_step`; a
   registry check names the depth key; `eval/correctness.py` and both
   official-baseline scripts read the pair for the precision policy the
   runner's identity reports.
5. **Every baseline script has a verdict.** Pi0's official-baseline comparison
   (a full forward on the checkpoint) reports `passed` under the `deepest`
   pair; Pi0.5's backbone and expert comparisons gate on both metrics.

## Alternatives considered

- Keep cosine alone with tighter numbers: rejected; a tighter direction gate
  still passes a uniform scale error.
- Mean squared error in absolute units: rejected; the number depends on the
  scale of the tensor, so one threshold cannot serve the KV cache and the
  action chunk.
- Calibrate from the full-depth PR1 dumps: rejected; at ten diffusion steps
  and eighteen layers the two Pi0.5 routes disperse to `rel_rms` 0.25 on the
  suffix cache, which is the chaotic regime the registry reports but does not
  gate. The gate's own depth is what calibrates the gate.
- Leave the samples' dtype edges (subnormals, saturation) to the gate:
  deferred; recorded in the runtime note's open items.

## Consequences

- A kernel that scales its output wrongly fails the gate; before, it passed
  when the direction was right.
- The float32 cosine artifact cannot mask or fake a failure.
- Reports carry `max_rel_rms`, `min_cosine`, the tolerance pair and its key;
  older reports carry `threshold` and are not re-read.
- Changing a tolerance means re-running `eval/calibrate.py` and citing the
  run, not editing a number.

## Verification

- Login node: `eval.calibrate` on the PR1 full-depth dumps reproduces the
  float64 numbers (Pi0.5 shipped vs reference `rel_rms` 0.097 on the chunk,
  Pi0 0.0098); `python -m eval.smoke`; every changed module imports.
- GPU, the calibration input (job 599011, ACD1-50): `eval.correctness`
  shipped vs reference, both Targets, seeds 0 and 1 at 1x1 and seed 0 at
  1x18 and 10x18. Shallow: Pi0.5 `rel_rms` 1.1e-3 / 1.8e-3, cosine
  0.9999994 / 0.9999983; Pi0 6.2e-3 / 6.6e-3, 0.999981 / 0.999978; every
  stage output but the action chunk bit-identical at that depth. Full depth,
  one step: Pi0.5 prefix cache `rel_rms` 3.4e-2, cosine 0.99943; Pi0 1.7e-2,
  0.99985. Ten steps (reported, not gated): Pi0.5 suffix cache 0.25 / 0.970.
  The registry's pairs are ten times the worst shallow and the worst
  full-depth dispersion.
- GPU verification of the pairs (job 599021, ACD1-33): 1x1 on both Targets
  passes both gates. `eval.gate --baseline` on Pi0.5: every correctness
  gate passed with both numbers printed (shallow `rel_rms` 1.06e-3, cosine
  0.9999994; deep 3.4e-2 / 0.99943 and ten-step 0.25 / 0.970 reported),
  the baseline tier passed under its pairs, and the verdict was `fail` on
  the tail bound alone (the open finding of the acceptance note: 10.9 ms
  on the candidate leg, 0.14 on the reference). `eval.pi05.reference`
  under the OpenPI interpreter: backbone layer 0 `rel_rms` 0.011, cosine
  0.99994; deepest 0.075 / 0.9972; worst per-layer step 0.00025; the
  expert's one step 0.0042 / 0.999991; both passed. The `layer0` pair
  sits six times above the measured layer-0 dispersion against OpenPI.

## Related notes

- [acceptance is deployability](../architecture/2026-09-06-acceptance-is-deployability.md):
  the registry these tolerances live in; its Decision §7 made the threshold a
  key, this note makes the key a pair.
- [runtime/Target boundary and acceptance first](../architecture/2026-09-06-runtime-target-boundary-and-acceptance-first.md):
  Decision §5, the correctness structure, now gated on two metrics.
