# The Candidate Loop — workspace, evidence, stop, promotion

## Workspace — never committed

`artifacts/ktasks/<task>/` (`artifacts/` is gitignored):

```text
contract.md          the filled contract; its content travels into the note/PR on promotion
docs/draft.md        first plan draft — REQUIRED before any code exists
docs/plan.md         the executable plan (human mode: signed before candidate 1)
candidates.jsonl     the ledger
benchmark.csv        tabular results, one row per measurement
profile/             each candidate's ncu REPORT.md, or a pointer to its
                     artifacts/profile/<run>/ run directory; a torch / nsys
                     timeline only for a stream-level question (gaps, overlap)
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
- Profiling has two levels. Kernel level — why this candidate is slow — is
  `ncu` first: `ncu-report` runs the pass (capture on an ncu-capable node,
  the six dimensions, the playbook, `REPORT.md`) and names the symptom;
  `wiki/README.md` maps the symptom to the move. Pipeline / stream level —
  which stage, launch, gap or overlap — is the torch profiler and `nsys`
  first, through `gpu-profiler-analysis`; it localizes before a task exists
  and is the Profile step after Deploy, and it does not replace an ncu
  report inside the loop. A bottleneck without metric values behind it is a
  hypothesis, not a finding.
- Never edit kernel source while a compile or profile job is in flight.

## Stop conditions

Stop at the first of:

1. promotion criteria met: the acceptance registry's candidate rule
   (`eval/acceptance.py`, `promotion_bar_ms` on chunk `min` in `improve`
   mode) and its deployment tail bound, decided by `python -m eval.gate`;
2. the remaining blockers are explicit — a named constant GAP, a ceiling
   shown reached with evidence, a dependency outside the task;
3. the Target's budget in the registry is exhausted (the contract may
   narrow it, never widen it).

Auto mode returns only at a stop condition, with the evidence pack: the
ledger, the best candidate's parity and benchmark results, and what it would
try next with more budget.

## Promotion checklist

- parity gates pass (the gate set named in the contract);
- benchmark evidence at the stated reproducible config, meeting the criteria;
- wired into the op table; `python -m eval.gate --candidate <plan>` on the
  Target returns `pass` (a `blocked` run is rerun, never read as a pass);
- a built-in case added to `benchmarks/kernels.py` when the kernel is
  production;
- Agent Note added or updated; the evidence summary copied from the workspace
  into the note/PR — the workspace is disposable after this.

Promote only on evidence; record every rejection's reason.
