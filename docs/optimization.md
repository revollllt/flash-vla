# Model optimization workflow

This is the default workflow for agents optimizing an existing Target. Use the
existing runtime, kernels and tools directly. Campaign initialization, context
activation, re-anchor, qualification and publication are not prerequisites.

1. Identify the model, checkpoint, fixed inputs and environment; read the current
   implementation and relevant prior results. Measure the current version, or reuse
   a measurement from this work session with the same code and conditions.
2. Profile top down: capture the complete forward first to locate the dominant
   module, CPU/GPU work, synchronization and gaps. Then profile the selected module
   at call-site/kernel level. Estimate recoverable end-to-end time before choosing
   an optimization. Reuse a representative profile while the bottleneck is unchanged.
3. Look for existing implementations first, especially Pi0/Pi0.5 and other Target
   kernels. Make one small change based on the observed bottleneck.
4. Check correctness relevant to that change, then measure the candidate in a
   separate process. Confirm that local gains reach the complete forward.
5. Keep a correct improvement; otherwise revert only the candidate's changes or
   record uncertainty. Append a short result and continue. Refresh the overall
   profile when the bottleneck or execution path changes, or local gains fail to
   explain end-to-end results. Respect the user's task scope and time budget.

## Commands

Run GPU commands inside a Slurm allocation on lab-H100. Use the Target's existing
runtime and asset options; apply the same options and seed to every command below.
These examples use registered Pi0.5 plans; they do not select a real checkpoint
unless its asset options are supplied.

```bash
# A and B can be measured separately; each invocation uses a fresh worker.
python -m benchmarks latency --target h100/pi05 --plan reference --out artifacts/A.json
python -m benchmarks latency --target h100/pi05 --plan shipped --out artifacts/B.json
# Or run exactly two independent workers, sequentially on the same GPU.
python -m benchmarks latency --target h100/pi05 --plan reference --plan shipped --out artifacts/AB.json

# First the entire forward timeline; then only the module identified as costly.
python -m benchmarks profile --target h100/pi05 --plan shipped --overview --trace-dir artifacts/profile/overview
python -m benchmarks profile --target h100/pi05 --plan shipped --segment action_expert --trace-dir artifacts/profile/detail
# If hardware counters are needed, use ncu-report for selected kernels only.

# A shallow model check is one available check, not sufficient for every change.
python -m eval.correctness --target h100/pi05 --plan shipped --steps 1 --layers 1
```

`--overview` records the real forward with its first captured graphs, including
host work. Its CPU segment labels are launch scopes, not GPU execution durations;
inspect the GPU timeline. `--segment` attributes one selected module without
running attribution for every module. Profiling must use the benchmark's inputs,
compile settings and graph regime. Profiler numbers are diagnostic; benchmark
latency in a separate uninstrumented process.

## Measurement and correctness

- Compare the same physical GPU, driver/runtime, checkpoint, inputs, shapes,
  precision and timing scope. Record these with the command. A new environment
  starts new measurements; no context activation is needed, and cross-environment
  differences must not be counted as code improvements.
- Each version runs in its own process using only the initial capture. Every graph
  stays on its capture stream. Defaults: five forward warmups, 100 wall samples,
  no soak, no attribution, chunk latency only. Loading/capture are outside timing;
  input staging, host work, replay and the final CUDA synchronization are included.
- Median is the primary comparison. Keep min, p99 and ordered samples. At 100
  samples, p99 is descriptive, not a daily promotion gate. A/B/A or `--calibrate`
  is optional when drift is suspected; without controls, noise is unknown, not zero.
  Results close to observed variation need a focused repeat, not automatic promotion.
- Use `latency --breakdown` only when device/host/segment diagnostics help answer
  the current question; `--attribution` opts into CPU/GPU telemetry. Neither is
  required for normal measurement. An unchanged version need not be remeasured
  for every candidate while the recorded conditions remain applicable.
- Reuse existing numerical references and tolerances. Check relevant real shapes,
  finiteness and outputs after a kernel change. Shared runtime/synchronization
  changes need the affected GPU integration check. Semantic or approximate changes
  need appropriate model/task quality evidence. Fewer checks do not mean looser
  tolerances; do not create new oracle files when an existing reference suffices.

## Record and continue

Use Git, raw benchmark JSON and one append-only experiment log. A log entry needs
only the hypothesis/change, code revision (and diff for uncommitted work), command
and environment, correctness result, latency and keep/reject/uncertain conclusion.
No new hashes, frozen contracts, receipts or schemas. An unsuccessful experiment
is a normal result. Source edits can be tested before commit; preserve the diff
needed to identify the tested code.

Historical Campaign commands and `eval.gate` remain explicit tools for old
records or complete qualification. Their state machine, three-leg evidence,
publication and reconciliation rules do not govern daily optimization. Do not
fabricate missing historical evidence to fit a new measurement. Plot/publish at
useful milestones; a presentation failure does not stop the next experiment.
