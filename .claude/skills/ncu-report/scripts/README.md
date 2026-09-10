# NCU helpers

Run from the project root. Report analysis needs Python and the Nsight Compute
`ncu_report` module, but no GPU. Set `NCU_PYTHON_DIR` to the matching installation's
`extras/python` directory when it is outside standard install locations.

| Helper | Use |
|---|---|
| `report_query.py` | `summary`, `rules`, `hotspots`, or `compare`; `--action N` selects a launch |
| `analyze_reports.py` | Export metrics/rules for one or more tagged reports |
| `extract_stall_hotspots.py` | Aggregate PC/source stalls; `--sass` shows instructions |
| `plot_timeline.py` | Inspect sampling series with `--list`, or render an ASCII timeline |

```bash
python .claude/skills/ncu-report/scripts/report_query.py summary artifacts/profile/kernel.ncu-rep
python .claude/skills/ncu-report/scripts/report_query.py hotspots artifacts/profile/kernel.ncu-rep --top 10
python .claude/skills/ncu-report/scripts/analyze_reports.py \
  --run-dir artifacts/profile/compare \
  --report artifacts/profile/a.ncu-rep --tag a \
  --report artifacts/profile/b.ncu-rep --tag b
```

For a standalone harness, copy `harness_template.cu` and its included
`safetensors_loader.h` into an ignored experiment directory. Fill the template's
TODOs before compiling; an existing project driver is often enough.

```bash
mkdir -p artifacts/profile/harness
cp .claude/skills/ncu-report/scripts/harness_template.cu artifacts/profile/harness/kernel.cu
cp .claude/skills/ncu-report/scripts/safetensors_loader.h artifacts/profile/harness/
# After filling the template, compile for the intended GPU (this example: Hopper).
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo \
  -I third_party/cutlass/include artifacts/profile/harness/kernel.cu \
  -o artifacts/profile/harness/kernel -lcuda
```

See [collection](../references/03-collection.md) for capture selection and
[Python API](../references/04-python-api.md) for custom analysis.
