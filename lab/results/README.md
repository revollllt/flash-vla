# Saved results

For a new optimization run, keep its command, workload/environment and conclusions
in `results/<target>/<run>/README.md`, raw benchmark JSON in `measurements/`, and
append each model trial to `iterations.csv`. The agent updates this table after
each experiment; `latency_ms` comes from that trial's deployed end-to-end median.
Keep code revisions or diff references and numerical-check results beside the
corresponding measurement. Kernel-only trials belong in the notes.

The following numbers are illustrative, not project measurements:

```csv
iteration,change,latency_ms,decision,report,revision
0,Current version,100.0,start,measurements/000.json,commit-a
1,Fused projection,94.0,keep,measurements/001.json,commit-b
2,Alternate tile,95.0,revert,measurements/002.json,commit-c
3,Failed numerical check,,failed,,commit-d
```

Use `start` for the measured starting version, `keep` for an accepted model,
`revert` for a rejected trial, `uncertain` when the evidence is inconclusive,
and `failed` for an invalid or numerically incorrect trial. Missing valid latency
stays blank. Only `start` and `keep` advance the retained-model line. Different
checkpoint/input/shape/device or timing conditions use separate run directories.

```bash
python -m lab.results curve results/pi05-h100/run-name/iterations.csv \
  --out results/pi05-h100/run-name/progress.svg --title "Pi0.5 · H100"
```

Embed `![Optimization progress](progress.svg)` in the run README and link that
README from the results overview. The curve command plots the recorded decisions;
it does not qualify a candidate or run measurements.

## Historical results

Read the existing [results dashboard](../../results/README.md) or its `index.json`.
These tools render saved `trace.json` files without recreating an experiment,
loading a model or importing a Campaign controller. Run from the project root
with Matplotlib installed:

```bash
python -m lab.results plot path/to/trace.json --out artifacts/progress.svg
python -m lab.results rebuild
```

`plot` renders one trace. `rebuild` regenerates the SVG, summaries and dashboard
for legacy entries under `results/targets/` (use `--root` for another checkout).
It also discovers links to new run READMEs through their `iterations.csv`. Existing trace and resume
files remain historical data; neither command executes their recorded commands.
Context boundaries and unsuccessful trials stay visible, and each context's
speedup uses its own recorded anchor. Rebuilding is a presentation operation,
not a new validation of historical performance.

Campaign creation, resume, qualification and automatic publication have been
retired. New optimization work follows [the workflow](../../docs/optimization.md).
