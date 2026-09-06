# Agent Note: one deployment configuration, and a tracked lab workspace

Status: implemented

## Problem

The repository could not tell a deployment configuration apart from an
experiment. Nine named call-site plans sat beside the code that ships, only one
of which was ever deployed and one of which is the correctness oracle. The
per-kernel and per-chain trial scripts lived under the correctness tree with
names shaped by how they were written (`plan_parity`, `in_engine`,
`promotion_gate`, `fold_equivalence`, `xfs_real_chain`) rather than by what
they check, so `eval/` read as a pile of history rather than as the checks that
gate a release. The architecture was a nine-document tree written ahead of the
code, whose stale sections had already begun to contradict the source.

The cost is not tidiness. An agent asked to optimize this repository could not
answer "which configuration ships?" from the layout, and a reader could not
tell which document was authoritative.

## Decision

1. **One shipped plan and one reference plan per Target, in the deployment
   path.** `src/`, `eval/` and `benchmarks/` carry exactly those two per
   Target: the plan that deploys, and the correctness oracle route the shipped
   one is compared against. They are declared on the Target itself, so
   `--plan shipped` and `--plan reference` mean something without a lookup
   table.
2. **`lab/` is the optimization workspace, and it is tracked.** Candidate
   plans, per-kernel trials, ablations and their Slurm wrappers live there:
   `lab/plans/` (passed to any harness as `--plan lab/plans/<name>.json`),
   `lab/pi05/` (the Pi0.5 experiment scripts, moved out of the correctness
   tree), `lab/sbatch/` (their wrappers, submitted from the repository root).
   The dependency runs one way: `lab/` may import `src/`, `eval/` and
   `benchmarks/`; the deployment path never imports `lab/`. A promoted
   candidate ships its kernel and its check into the deployment path in the
   same change, and its trial script stays in `lab/` as the record of how it
   was measured.
3. **`eval/` uses direct names.** `acceptance.py` (the registry),
   `correctness.py`, `gate.py`, `smoke.py`, `metrics.py`, `pi05/reference.py`,
   `pi05/fold.py`, `pi05/tokenize.py`, `pi0/reference.py`, `tile_sm90/` and
   `baselines/`. A file is named for what it checks, not for the comparison
   technique it happens to use.
4. **Documentation collapses to one page.** `ARCHITECTURE.md` owns the premise,
   the boundary, the template, the dependency direction, the runtime interface
   as locators, the invariants, what a Target consists of, the optimization
   loop (the human's role, the flywheel, the order of evidence, the exclusion
   rule before attribution, the stop condition, the skill table), and the
   evaluation entry points. The nine-document tree and the Pi0.5 bring-up plan are
   deleted. Decisions, their alternatives and their evidence live in
   `.agents/notes/`; source is authoritative for implementation.

## Alternatives considered

- **Make the workspace a gitignored scratch directory.** Rejected: the trial
  scripts are the evidence behind recorded findings — which tile configuration
  won, which candidate was rejected and on what measurement. Ignoring them
  would delete the reproduction path for every number in the notes on the next
  clean checkout.
- **Delete the trial scripts outright, since they gate nothing.** Rejected for
  the same reason from the other side: they reproduce findings the notes cite,
  and several of them are the only way to localize a failure to one kernel
  rather than to the whole pass. Keeping them in the deployment path was the
  problem; keeping them is not.
- **Keep all nine plans in the deployment path and mark one "shipped".**
  Rejected: only one configuration ships, and a directory of near-identical
  alternatives is exactly the ambiguity this change removes. A candidate plan
  is an experiment and belongs with the experiments.
- **Keep the architecture tree and only fix its stale sections.** Rejected: the
  tree duplicated implementation that the explicit graph now states once in
  source, and every section that was not duplication was a decision, which
  belongs in a note.
- **Move the trial scripts under `benchmarks/`.** Rejected: `benchmarks/` is
  the generic harness surface every Target is measured through, and it must
  stay model- and experiment-free.

## Consequences

- "Which configuration ships?" is answerable from a Target's `target.py`, and
  a harness invocation names its route in one word.
- `eval/` is now readable as the gate: the registry, the checks it runs, and
  the per-model official-baseline tier. Nothing in it is a trial.
- Scripts that moved to `lab/` import `eval.metrics` and the current call-site
  names; the docstrings and READMEs that pointed at their old locations point
  at the new ones. The `kernel-design` skill's candidate loop works from
  `lab/`.
- One page can go stale in one place. The architecture no longer carries a
  per-document status line, because a page that needs one is a page that should
  have been a note.

## Verification

No GPU is required to falsify most of this; the checks are:

- **Login node.** `python -m eval.smoke` (declares both Targets and checks
  every plan under `lab/plans/` plus `shipped` and `reference` route the whole
  graph); `python -c "import flash_vla"`; `--help` on `eval.correctness`,
  `eval.gate`, `eval.pi05.reference`, `eval.pi05.fold`, `eval.pi05.tokenize`,
  `eval.pi0.reference` and each script under `lab/pi05/`, which is what proves
  the moved modules still import.
- **The dependency direction, by grep.** `grep -rn "lab/" src eval benchmarks`
  finds nothing: the deployment path does not reference the workspace.
- **No dangling documentation, by grep.**
  `grep -rn "docs/architecture\|PLAN.md\|promotion_gate\|in_engine"
  --include=*.md --include=*.py --include=*.sh` finds nothing outside
  `third_party/` and archived notes.
- **GPU** (job 598879, ACD1-15, the PR's tree). `eval.correctness` 1 step x 1
  layer: Pi0.5 shipped vs reference min cosine 0.9999994, Pi0 0.9999812, both
  replay-identical and finite. `eval.gate` on Pi0.5 shipped vs reference: every
  in-engine gate passed, the A/B/A read -0.995 ms on chunk min, verdict
  `blocked` on the baseline tier not run (the interpreter fix is the next
  note's). `lab.pi05.kernels` and `lab.pi05.ffn_taskloop --modes gu,dr,full`
  passed their own checks. `eval.tile_sm90` failed to compile: after its move
  it derived the repository root one level too high; fixed, re-run as job 598942.

## Related notes

- [explicit graph and ModelRunner](../architecture/2026-09-06-explicit-graph-and-model-runner.md):
  where the shipped and reference plans are declared, and why `Identity` has no
  table-option axis any more.
- [runtime/Target boundary and acceptance first](../architecture/2026-09-06-runtime-target-boundary-and-acceptance-first.md):
  the acceptance registry `eval/` is organized around; its Decision §7 is
  replaced by the one-page rule above.
- [the Pi0.5 Target](../architecture/2026-09-06-pi05-target-decisions.md): the
  bring-up decisions the deleted plan file carried.
