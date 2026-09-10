# Saved results

Read the existing [results dashboard](../../results/README.md) or its `index.json`.
These tools render saved `trace.json` files without recreating an experiment,
loading a model or importing a Campaign controller. Run from the project root
with Matplotlib installed:

```bash
python -m lab.results plot path/to/trace.json --out artifacts/progress.svg
python -m lab.results rebuild
```

`plot` renders one trace. `rebuild` regenerates the SVG, summaries and dashboard
under `results/` (use `--root` for another checkout). Existing trace and resume
files remain historical data; neither command executes their recorded commands.
Context boundaries and unsuccessful trials stay visible, and each context's
speedup uses its own recorded anchor. Rebuilding is a presentation operation,
not a new validation of historical performance.

Campaign creation, resume, qualification and automatic publication have been
retired. New optimization work follows [the workflow](../../docs/optimization.md).
