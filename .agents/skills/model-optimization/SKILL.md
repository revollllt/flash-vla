---
name: model-optimization
description: Cut an existing Target's deployed end-to-end latency on its GPU. Owns the model and kernel loops, measurement conditions, delegation and the exit condition. Use for an optimization round; bring-up of a new model belongs to target-onboarding.
---

# Optimization workflow

Approach the hardware's SOL at the model's real shapes with hand-written
kernels, and cut deployed end-to-end latency. Two loops:

- **Kernel loop**, parallel across agents: Profile → Analyze → Design →
  Implement → Validate, iterating on the result.
- **Model loop**, serial: Profile → Analyze → Design → Implement → Validate →
  Deploy → Profile.

The loop runs until step 7's exit condition. Having a result to report is not a
terminal state; a human intervention is normally a stop, not a restart. Each
attempt states one falsifiable hypothesis, the observation behind it, the time it
should recover, and the cheapest experiment that separates it from the
alternative; a failed or ambiguous result revises the hypothesis rather than
adding another candidate on top.

1. **Prepare the model and the environment.** Run from the project root on the
   target GPU, with the model's own Python environment, real checkpoint, fixed
   inputs and run parameters. Use an existing Target; onboard a new model with
   [target-onboarding](../target-onboarding/SKILL.md). The examples below use
   LingBot, whose checkpoint and fixture resolve through `FLASH_VLA_ASSETS` — a
   JSON file mapping logical asset IDs to local paths, with relative entries
   resolved beside it. Targets on synthetic weights, such as `rtx5090/pi0`, need
   no asset map.

2. **Measure the current model and check its output.** The current `shipped`
   version is the comparison point. Reuse the existing numerical references and
   tolerances, and any measurement from this round taken on the same code under
   the same conditions.

   ```bash
   python -m benchmarks latency --target h100/lingbot_vla --plan shipped --seed 42 --out results/lingbot-h100/run-name/measurements/000.json
   ```

   Hold the same physical GPU, driver and runtime, checkpoint, input, shape,
   precision and timing scope across versions. Each version runs in its own
   process, first capture, one graph per stream; 5 warmup and 100 measured
   iterations, compared on the median, raw samples kept. Timing covers input
   transfer, host work, replay and the final sync, and excludes load and
   capture. Measure A and B separately; re-measure only on evidence of drift.

3. **Read the model.** Follow `vision_encoder → llm_backbone → action_expert`
   against [ARCHITECTURE.md](../../../ARCHITECTURE.md) for shapes, call counts,
   existing optimizations, and repeated work inside the denoising loop.

4. **Profile top down and pick a bottleneck worth the work.** Take a whole
   forward's GPU timeline with
   [gpu-profiler-analysis](../gpu-profiler-analysis/SKILL.md), locate the costly
   module or the host/sync gap, then descend to its call sites and kernels. Use
   [ncu-report](../ncu-report/SKILL.md) where counters can settle whether a
   kernel is compute-, memory- or pipeline-bound. Size the remaining headroom
   from the existing `tools.profiling.floor` report and the measured constants
   in [hardware-unit-test](../hardware-unit-test/SKILL.md), together with the
   call count, and measure only what those cannot answer.

   ```bash
   python -m tools.profiling.model --target h100/lingbot_vla --plan shipped --seed 42 --overview --trace-dir artifacts/profile/overview
   # Descend only when the overview points at action_expert.
   python -m tools.profiling.model --target h100/lingbot_vla --plan shipped --seed 42 --segment action_expert --trace-dir artifacts/profile/detail
   ```

   Profile with the same inputs and execution config as the benchmark. Read GPU
   time; a CPU segment label only marks submission range. Measure deployed
   latency in a separate process with no profiler attached.

5. **Design and implement kernels, in parallel.** Give each kernel or
   independent fusion chain its own agent and worktree, iterating with
   [kernel-design](../kernel-design/SKILL.md), which owns the search and the
   implementation choices. A [kernel-wiki](../kernel-wiki/SKILL.md) miss on a
   problem with a known solution makes writing that page part of this round's
   output: "the corpus does not have it" is not "it has to be written from
   scratch." A delegated prompt carries the target and the evidence for it, the
   call site's shapes and layout, what may be changed, the hypothesis to test,
   how to validate it and what to return — not a copy of this session.

6. **Validate the kernel: correctness, then local gain.** Check the relevant
   shapes and inputs against the existing references and tolerances, and compare
   the kernel or the whole fusion chain with
   [benchmark-kernel](../benchmark-kernel/SKILL.md) under the real data layout,
   cache state and execution conditions. A failure or an uncertain gain stays in
   the kernel loop. **Hand a candidate to the model loop once it is correct and
   its local gain is credible** — it does not have to be finished first.
   Performance measurements on one GPU are serial; with several GPUs each can
   run its own before/after on one card, and nothing else should contend during
   a model timing run.

7. **Integrate, validate, deploy and measure, one candidate at a time.** Bring
   one winning candidate into the current best model and plan, and run the
   correctness checks it affects; an approximating change also needs a task
   quality check. Deploy to the real inference path, then run step 2's command
   against the plan actually loaded, saving the result under the same
   `measurements/` directory by iteration number (`001.json`). Keep it on an
   end-to-end gain; otherwise revert or record it as uncertain and move to the
   next candidate. After K1 is accepted, K2 compares `M+K1` against `M+K1+K2`.

   When a local gain does not reach the model, return to step 4 and feed the
   analysis back to the kernel loop. Otherwise pick the next hotspot from the
   deployed version. Adapt candidates to the current model, re-validate only
   what the change touches, and reuse a profile whose bottleneck has not moved.

   Before ending, give each leading hotspot its share of reachable capability
   and the reason no candidate remains; without both, it is not finished.
   Replacing a deployed implementation is this loop's normal operation and needs
   no approval. Three things do: the numerical contract (precision policy and
   its tolerances), a change of measurement conditions, and an exhausted budget.

**Review the conclusions.** These six test a *conclusion*, not a process, and
each is answerable from numbers already written down. The cases behind them are
in [corrections](../../../docs/corrections/README.md).

- **A compared number comes from this round, or states its provenance.** A row
  in a summary table is not the current baseline.
- **A ratio names its denominator.** An impossible ratio is a bug in the
  denominator before it is a finding: a measurement above "peak" means the peak
  was computed wrong — wrong precision tier, wrong clock, wrong unit.
- **An estimate names the memory tier it assumes**, against this machine's
  measured constants. A workspace that fits L2 cannot be priced at DRAM
  bandwidth.
- **A negative carries the toolbox it was measured with, and expires when that
  changes.** If a technique later measured to be worth a factor was not in it,
  the conclusion is deferred, not settled, until it is re-run.
- **"Unavailable" needs the boundary probed, not one error message.** Probe with
  [hardware-unit-test](../hardware-unit-test/SKILL.md); an implementation such
  as `ptxas` outranks a second-hand document.
- **A blocker named precisely is the next target, not a property of the world.**
  "It holds once X changes" has to come with the cost of changing X, or it
  parks a proven technique inside your own conclusion.

Each round saves its change, code revision or diff, command and environment,
correctness result and raw benchmark JSON under `results/<target>/<run>/`, and
updates `iterations.csv` and `progress.svg`. Start from the current version's
measured point. Record the deployed end-to-end median for every model candidate,
keep the ones that did not gain and the uncertain and failed attempts, and let
only the retained versions advance the main curve. A kernel's local gain goes in
the experiment note; it never stands in for model latency on that curve. Changed
conditions start a separate record — data taken under different conditions is
never joined into one speedup curve. See [the results
tools](../../../lab/results/README.md) for the short version.

A round is reported on the deployed version and its comparable end-to-end
change, the correctness evidence, the failed and uncertain conclusions, the
remaining bottlenecks and the next hypothesis worth testing, each linked to its
change and its record. A gain that was not measured is not reported.

Load a skill when the step needs it; none of them is required every round. Do
not add hashes, frozen contracts or gates, and do not run unrelated sweeps.
`python -m lab.results rebuild` redraws saved results. The Campaign state machine
is retired.
