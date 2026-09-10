> Historical record. Current instructions: [optimization workflow](../optimization.md).

# Flash-VLA optimization checklist

Source: optimization plan v0.2 (2026-09-07), PR01–PR15. This is the execution
checklist requested by the owner. Check an item only after its stated evidence
exists; implementation, CPU tests, GPU checks and conclusions are distinct.

Owner constraints override the draft: no new hashes, frozen contracts,
baselines or acceptance gates by default. Reuse existing acceptance and
incumbent; use explicit run IDs, source copies and actual loaded paths when
isolation is needed. Do not hash unchanged files or introduce a general
framework before its current use is exercised. Preserve unrelated changes.

## Existing evidence

- [x] Inspect actual checkout and preserve unrelated changes (`cc2289d`, existing `.gitignore` edit).
- [x] Retain supplied plan/starter under `artifacts/optimization_inputs/`.
- [x] Reject unknown route keys before backend construction (PR02 slice; five CPU tests).
- [x] Check all 28 existing Pi0/Pi0.5 plans against their CPU graphs.
- [x] Integrate completed-range exporter and 16 CPU tests (PR13 slice).
- [x] Compile CUDA demo, check parity for 1/64/512 CTAs, memcheck, SASS off removal (PR14 slice; job 602410).

## PR01 — Experiment inputs and evidence

Evidence / applicability: `eval/tests/test_optimize.py`; acceptance is read-only. The owner constraints explicitly replace the plan's default hash/frozen-contract proposals.

- [x] Validate minimal experiment inputs: target, hypothesis, competing explanation, cheapest probe, scope, stages and budget.
- [x] Read acceptance without changing policy; reject missing fields, unsupported versions, scope escape and excessive budgets on CPU.
- [x] Separate execution, validity, correctness, conclusion and promotion status in persistent evidence.

## PR02 — Actual implementation and routing

Evidence / applicability: `component-preflight.json`, `source.json`/`loaded.json` in the component run, five binding tests and the additional shallow graph checks. Identity remains unknown outside declared source coverage; equal routes do not prove no-op.

- [x] Capture only declared source/dependency inputs and effective routes with explicit run IDs; record dirty changes and loaded paths, without new hashes.
- [x] Verify changed source/header invalidates reuse while notes-only edits do not trigger rebuilding.
- [x] Prevent an uncertain implementation identity from being reported as a proven no-op.

## PR03 — Two implementation versions

Evidence / applicability: Job 602496: A/A-prime/B are separately loaded libraries; geometry 4K/2V versus 3K/3V; both deployed shapes pass parity. Unsupported loaders fail explicitly.

- [x] Load two isolated versions of one existing component in one process and record actual artifacts.
- [x] Verify both versions execute separately; reject aliasing and unsupported loading modes.
- [x] Run A/A′ with separately built/loaded/captured instances and retain raw measurements.

## PR04 — Timeline and floor semantics

Evidence / applicability: Eight timeline tests and job 602515 (`profile_602515/profile.json`). Same-trace union is distinct from the separately measured stage wall.

- [x] Retain start/end, stream, invocation and available graph-node identity from one replay.
- [x] Compute duration sum, interval union and makespan separately, including occurrence-local repeated groups.
- [x] Keep ambiguous mapping unattributed; never label differences between separate captures as overlap.
- [x] Remove automatic floor stopping conclusions for unsupported overlapping group models.
- [x] Test serial/overlap/gap/repeated-group/multi-stream/missing-mapping cases on CPU.
- [x] Capture a representative real GPU chain and report profiled intervals separately from unprofiled stage timing.

## PR05 — Existing evaluation early exit

Evidence / applicability: Four early-exit tests and `workflow-comparison.json`; existing acceptance values are unchanged. No official evidence reuse is inferred from indirect comparisons.

- [x] Separate numerical oracle from the existing performance incumbent in gate arguments/evidence.
- [x] Stop after required correctness failure or unavailable required official evidence, before ordinary performance/floor work.
- [x] Record actual correctness coverage edges; do not imply candidate→official from two indirect comparisons.
- [x] Verify early exit and missing-evidence behavior with CPU mocks; preserve all acceptance thresholds.

## PR06 — Recoverable experiment loop

Evidence / applicability: `component-isolation-001/evidence.json`, jobs 602483/602496, CPU interruption/recovery and budget tests. Slurm accounting additionally retains setup failures outside stage time.

- [x] Implement preflight/probe/check/measure/qualify by calling existing tools; persist commands, failures, elapsed costs and stage outputs.
- [x] Resume interrupted runs without silently rerunning uncertain stages or double-counting costs.
- [x] Enforce acceptance task budgets, reserve confirmation resources and stop on invalid/correctness-failed evidence.
- [x] Exercise success, early failure and interrupted recovery on CPU and one bounded GPU task.

## PR07 — Fixtures and confirmation

Evidence / applicability: Actual fixture in `async-attention-001/run_602528`, independent jobs 602547/602559; GPU alias check 602589; QKV weight rotation and confirmation jobs 602586/602588.

- [x] Export one real region fixture with copied values, ABI/layout/alias metadata and production cache-context limitations.
- [x] Keep correctness snapshots outside timing; preserve weight rotation for performance fixtures.
- [x] Fix paired order and sample counts before execution; retain raw A/A′ and confirmation results.
- [x] Verify fixture values/aliasing/replay behavior on GPU.

## PR08 — Evidence context

Evidence / applicability: `evidence-context.json`, `attention-query.json`, existing NCU query and scoped wiki queries. The NCU FFN action is inapplicable to attention; missing conditions/counters remain unknown.

- [x] Query existing NCU/wiki/templates/notes and kernel-trace summaries with architecture/precision/layout/synchronization applicability checks.
- [x] Separate observations, competing hypotheses and the next discriminating probe; unknown conditions remain unknown.
- [x] Test wrong-architecture/layout inputs and missing counters on CPU.

## PR09 — Experiment memory

Evidence / applicability: Small index and recovery tests cover duplicate delivery, missing large artifacts, changed conditions, deliberate replication and invalid versus no-benefit classification. Current decisions are in the owning Agent Notes.

- [x] Persist small evidence index, failure classification, source links and reopen conditions independently of large artifacts.
- [x] Detect duplicate experiments without permanently excluding changed conditions or deliberate replication.
- [x] Test invalid-versus-no-benefit classification, reopening and missing large-artifact behavior.

## PR10 — Resource coordination

Evidence / applicability: Isolated component/source/output directories; Slurm job/GPU UUID recording. CPU fault injection covers live owners, worker loss, contention and concurrent budget reservations; no actual shared GPU mixing is used as a test.

- [x] Isolate trial source/build/output directories and declare dependency/write conflicts.
- [x] Use the existing Slurm scheduler for GPU ownership; track job/GPU identity and actual terminal state.
- [x] Test worker loss, expired ownership while job runs, duplicate delivery and external contention; never assume expiry means idle.

## PR11 — Latest-incumbent qualification

Evidence / applicability: Source applicability and post-gate mutation tests; shared component GPU checks cover Pi0/Pi0.5. No promotable candidates exist, so combined-candidate GPU qualification and actual promotion/rollback are not triggered. Source-version qualification remains explicitly unsupported; supported route comparisons require the same checked source tree.

- [x] Check whether incumbent/dependencies changed before reusing qualification; request only affected remeasurement.
- [x] Record cross-Target impact, candidate combinations and rollback location without modifying shipped automatically.
- [x] Test stale evidence, duplicate results and conflicting candidates; verify relevant combined GPU behavior.

## PR13 — Trace semantics and query

Evidence / applicability: 17 exporter plus five query tests; actual capture/stage metadata, TMA/WGMMA token/generation validation and filtered wait/tail summaries.

- [x] Record capture/stage dictionary/coverage/clock and validation metadata with raw outputs.
- [x] Support marker/range identity and async token/generation pairing, with incomplete/overflow data explicit.
- [x] Query ranges by SM/launch/task/stage/role/iteration with waits/tails/coverage summaries.
- [x] Extend CPU cases for replay identity, unknown clocks, missing endpoints, token reuse and partial captures.

## PR14 — Device calibration and visualization

Evidence / applicability: Jobs 602410/602503: timer observation, fixed sampling, SASS/resources, parity and memcheck. Perfetto v58.3 visibly imports the real one-CTA trace and reports WMMA scope 2.752 us.

- [x] Measure observed timer granularity on H100; distinguish it from guaranteed clock accuracy.
- [x] Compare fixed off/coarse/focused samples and resource/SASS changes; report perturbation without subtracting a constant overhead.
- [x] Import an actual GPU trace in Perfetto and verify event units/tracks.

## PR15 — Hopper async and real task

Evidence / applicability: Real existing attention source copy, job 602528; jobs 602547/602559 pass parity, memcheck/synccheck and replay isolation. 602559 validates 32 TMA plus 32 WGMMA observation pairs. Off/coarse/focused results falsify negligible instrumentation, so quantitative bottleneck attribution is withheld.

- [x] Instrument an existing TMA/WGMMA primitive at its original issue/wait boundaries without adding synchronization.
- [x] Validate operation/group/generation pairing and independent CUDA Graph replay storage.
- [x] Run numerical/synchronization checks and sampling perturbation comparisons on H100.
- [x] Use trace on one actual production region to distinguish competing explanations; recheck with tracing off.

## PR12 — Real pilots and effectiveness

Evidence / applicability: QKV jobs 602586/602588, TMA jobs 602552/602553, retained failed jobs, fixed CPU old/new workflow comparison and controller cost comparison. No performance candidate is promoted; E2E gate is therefore not triggered.

- [x] QKV pilot: confirm current route/shape/evidence, price competing mechanisms with the cheapest discriminating experiment.
- [x] Vision short-K/TMA pilot: control transaction bytes/layout/occupancy or state remaining confounders.
- [x] Retain every trial, including failures, resource cost and independent confirmation; no unsupported speedup attribution.
- [x] Compare a bounded existing-workflow versus new-workflow task with fixed inputs/budget; state carryover and small-sample limitations.
- [x] Report framework overhead and information gain separately from kernel/E2E benefit; complete relevant existing qualification if promoting.
- [x] Close this checklist only when all applicable evidence is present; leave unresolved items visibly unchecked.

## Closure and support boundary

The checklist's applicable first-version items have evidence; conditional
promotion work is not triggered because the measured candidates are not eligible.
This does not claim that all possible backends or cross-source-version evaluator
protocols are implemented. See `optimization-results.md` for the explicit
unsupported qualification path, unresolved performance attribution, raw costs
and reopening conditions. No E2E speedup or full-model official qualification is
claimed. The owner's unrelated `.gitignore` edit is retained.
