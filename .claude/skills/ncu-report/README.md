# ncu-report

A Claude Code skill for profiling CUDA kernels with Nsight Compute on this repo's target, **NVIDIA H100 (sm90)**, and turning the report into a named bottleneck and a ranked optimization plan. It covers the whole loop — profiled process, Slurm capture on an ncu-capable node, login-node parsing with the `ncu_report` Python API, six analysis dimensions, a diagnosis playbook, and the final report — with the sm90 metric-name and stall-reason vocabulary verified on this cluster's Nsight Compute 2025.4.1.

## Provenance

The structure, the generic reference text and the helper scripts derive from
[mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill)
(MIT, `LICENSE` retained), which targets B200 / sm100. This copy:

- makes H100 / sm90 the primary target and adds what the upstream never had:
  a verified sm90 vocabulary (`references/08-sm90-metric-names.md`), the Hopper
  companion (`hopper-cuda-programming.md`), the sm90 playbook patterns O–V, the
  wgmma / TMA / cluster / spill / bank-conflict metric families in the helpers,
  and the 2025.4.1 API shapes (rule dicts, `sass_by_pc(pc)`, the
  `warpsampling:` / `*.TriageCompute.*` timeline names);
- binds the workflow to this host: ncu-capable node list, `--clock-control=none`,
  `sbatch/_common.sh`, the `-lineinfo` env hooks, the engine's own drivers as the
  profiled process, `artifacts/profile/<run>/` as the run-directory home;
- keeps the upstream B200 material as an appendix (`references/08-b200-metric-names.md`,
  `blackwell-cuda-programming.md`) for reports from that GPU;
- drops the flashinfer-trace dataset browser (no such dataset here); the
  harness guide keeps the generic advice on choosing representative workloads.

## Layout

```
.
├── SKILL.md                          ← entry point (YAML frontmatter + workflow)
├── hopper-cuda-programming.md        ← companion: H100 architecture, Hopper features, principles ↔ NCU signals
├── blackwell-cuda-programming.md     ← companion (upstream, Chinese): B200
├── references/
│   ├── 00-directory-layout.md        ← run-directory convention (read first)
│   ├── 01-workflow.md                ← end-to-end checklist
│   ├── 02-harness-guide.md           ← profiled process, -lineinfo hooks, standalone harness
│   ├── 03-collection.md              ← ncu recipes and the Slurm job
│   ├── 04-python-api.md              ← ncu_report patterns (2025.4.1 shapes)
│   ├── 05-analysis-dimensions.md     ← six dimensions, sm90 names
│   ├── 06-diagnosis-playbook.md      ← pattern → cause → fix (A–N generic, O–V sm90)
│   ├── 07-report-template.md         ← REPORT.md structure
│   ├── 08-sm90-metric-names.md       ← verified sm90 vocabulary, diff vs B200
│   ├── 08-b200-metric-names.md       ← upstream sm100 reference
│   └── 09-common-issues.md           ← permissions, clocks, sampling, JIT, versions
├── scripts/
│   ├── report_query.py               ← summary / rules / hotspots / compare
│   ├── ncu_utils.py                  ← shared helpers, key-metric lists
│   ├── analyze_reports.py            ← archive + compare
│   ├── extract_stall_hotspots.py     ← per-line stall aggregation
│   ├── plot_timeline.py              ← ASCII timelines
│   ├── harness_template.cu           ← standalone harness (sm_90a)
│   ├── safetensors_loader.h
│   └── README.md
└── LICENSE                           ← MIT (upstream)
```

## Requirements

- Repo venv (`.venv/bin/python`, 3.12) on the login node for parsing.
- Nsight Compute CLI on a compute node for capture (`cuda/13.1` module → 2025.4.1).
- An ncu-capable node (`ACD1-10/20/21/31/40/62`); see `references/09-common-issues.md` for `ERR_NVGPUCTRPERM`.
