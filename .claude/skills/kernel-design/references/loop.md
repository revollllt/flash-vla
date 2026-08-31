# The Candidate Loop — workspace, evidence, stop, promotion

## Workspace — never committed

`artifacts/ktasks/<task>/` (`artifacts/` is gitignored):

```text
contract.md          the filled contract; its content travels into the note/PR on promotion
docs/draft.md        first plan draft — REQUIRED before any code exists
docs/plan.md         the executable plan (human mode: signed before candidate 1)
candidates.jsonl     the ledger
benchmark.csv        tabular results, one row per measurement
profile/             NCU / nsys artifacts and their summaries
runs/                build products, job logs
```

## Draft -> plan

The draft states: the measured baseline and how it was validated; the risks
and unknowns; candidate directions ranked by expected value; the first
concrete steps; the exact validation and evaluation commands; and the evidence
that will promote or reject a candidate. Convert it into `plan.md` before
editing code. Directives given mid-loop (human mode) are folded into
`plan.md`, so the plan stays the source of truth rather than the transcript.

## The ledger

One JSON line per candidate: `id`, `parent`, `backend`, `thesis` (one line —
what this candidate believes will be faster), `status`
(`kept | revised | rejected` + reason), `parity`, `min_ms` / `median_ms`,
artifact paths. A candidate is backend x strategy — TileLang candidates are
as legitimate as hand-written CUDA, and different call sites may promote
different backends. Rejected candidates keep their reason; that is half the
value of the ledger.

## Measurement discipline

- Baselines are measured before candidate 1, on the exact production shape.
- `benchmark-kernel` owns method. Compare plans same-process A/B/A; with
  unpinned clocks read `min`, not `median`.
- Capture profiles via `gpu-profiler-analysis`; interpret them via
  `ncu-report`; pick the responding move via `wiki/README.md`.
- Never edit kernel source while a compile or profile job is in flight.

## Stop conditions

Stop at the first of:

1. promotion criteria met;
2. the remaining blockers are explicit — a named constant GAP, a floor shown
   unreachable with evidence, a dependency outside the task;
3. the contract's budget is exhausted (default: 6 candidates, or 3
   consecutive without improvement; the contract may override).

Auto mode returns only at a stop condition, with the evidence pack: the
ledger, the best candidate's parity and benchmark results, and what it would
try next with more budget.

## Promotion checklist

- parity gates pass (the gate set named in the contract);
- benchmark evidence at the stated reproducible config, meeting the criteria;
- wired into the op table; the e2e plan run shows no regression;
- a built-in case added to `benchmarks/kernels.py` when the kernel is
  production;
- Agent Note added or updated; the evidence summary copied from the workspace
  into the note/PR — the workspace is disposable after this.

Promote only on evidence; record every rejection's reason.
