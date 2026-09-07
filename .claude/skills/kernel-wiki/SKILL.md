---
name: kernel-wiki
description: Use when a kernel question on this repository's NVIDIA H100 (sm90) needs an answer with evidence — why a wgmma, TMA, mbarrier, cluster/DSMEM, PDL or persistent-kernel body is slow, which move answers a Nsight Compute stall reason (gmma, long_scoreboard, no eligible warp, mio_throttle), what a fusion, a split-K reduction, a PDL trigger position or a weight stream costs, how FlashAttention-3, DeepGEMM, FlashMLA or a megakernel is built, or what changes on Blackwell (tcgen05, TMEM, CLC). Do NOT use for a machine constant (hardware-unit-test), a latency measurement (benchmark-kernel), capturing or reading a profile (gpu-profiler-analysis, ncu-report), running the kernel workflow itself (kernel-design), generic CUDA Q&A, host-side framework integration, or distributed systems (DeepEP/EPLB/DualPipe).
argument-hint: "[natural-language-question] | [--symptom <stall-reason> | --tag <t> --type <kernel|technique|pattern|hardware>] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---

# kernel-wiki — Hopper (sm90) Kernel Optimization Knowledge Base

Query a structured, cross-referenced knowledge base of GPU kernel
optimization for NVIDIA Hopper (SM90), with Blackwell (SM100) kept as the
port-forward appendix. Built on
[mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki) (MIT):
its three layers, schema, validator, query tools and Hopper-relevant pages
are carried here, and this repository's sm90 measurements are added as pages
of the same shape. Run `.venv/bin/python scripts/repo_status.py` for current
corpus counts; the evidence boundary is recorded in `README.md`.

## When To Use This Skill

- **A Nsight Compute finding** — the report names a stall reason or a rule
  (`gmma`, `warpgroup_arrive`, `long_scoreboard`, `mio_throttle`, no eligible
  warp, a flat tile sweep): `--symptom` routes it to a pattern page whose
  candidate techniques are the next moves.
- **sm90 mechanisms** — wgmma, TMA and 3-D boxes, mbarrier phases, clusters
  and DSMEM, PDL (`griddepcontrol`), `setmaxnreg`, cache hints, proxy fences.
- **Pricing a design** — a fusion against the launches it removes, a split-K
  join, a layout producer, a PDL trigger position, a cold weight stream.
- **Measurement pathologies** — an isolated timer that overstates a fusion,
  a persistent kernel's heavy tail, a null ablation, a one-sided gradient.
- **Kernel case studies** — FlashAttention-3, DeepGEMM, FlashMLA, grouped and
  fused MoE, the four megakernel forms, each with its templates and numbers.
- **Blackwell** — tcgen05, TMEM, CLC, 2-SM MMA, NVFP4 and the wgmma to
  tcgen05 migration, for reading upstream code or planning a port.

Do NOT use this skill for a machine number (`hardware-unit-test` owns the
constants this wiki cites by tag), a latency claim (`benchmark-kernel`),
capturing or interpreting a profile (`gpu-profiler-analysis`, `ncu-report`),
the kernel workflow itself (`kernel-design`), generic CUDA questions,
host-side framework integration, or distributed systems.

## How To Query

All commands run from the skill directory. The scripts auto-resolve the wiki
root; no environment variable is required. This cluster's system `python3`
is 3.6, so use the repository venv: `.venv/bin/python`. The scripts use host
PyYAML when present and fall back to the bundled pure-Python loader
(`tests/check_yaml_fallback.sh` proves it).

### Path 1: By symptom, from a report (preferred inside the candidate loop)

```bash
.venv/bin/python scripts/query.py --symptom stall-gmma --compact
.venv/bin/python scripts/query.py --symptom fusion-regression --compact
.venv/bin/python scripts/query.py --symptom null-ablation
```

`symptoms` are a controlled vocabulary (`data/tags.yaml`) spelled after
`ncu-report`'s sm90 stall reasons and rule names.

### Path 2: Unified search (natural language, filters, aliases)

```bash
.venv/bin/python scripts/query.py "release ring frames before the epilogue"
.venv/bin/python scripts/query.py --architecture sm90 --type technique --compact
.venv/bin/python scripts/query.py --tag wgmma --type pattern
.venv/bin/python scripts/query.py --confidence measured --compact
```

Filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--limit`, `--compact`, `--paths-only`. `--tag`
and `--architecture` accept aliases — `--architecture H100` matches `sm90`,
`--tag WGMMA` matches `wgmma`.

### Path 3: Fetch a page by id or path

```bash
.venv/bin/python scripts/get_page.py technique-release-on-retirement
.venv/bin/python scripts/get_page.py pattern-serialized-wgmma --follow-sources
.venv/bin/python scripts/get_page.py kernel-megakernel-forms --body-only
```

### Path 4: Regex over bodies

```bash
.venv/bin/python scripts/grep_wiki.py "fence.proxy.async" --only wiki
.venv/bin/python scripts/grep_wiki.py "griddepcontrol" "trigger" --any
```

### Path 5: Pre-built indices and companion docs

Auto-generated under `queries/`: `by-problem.md` (symptom → pattern →
candidate techniques), `by-technique.md`, `by-hardware-feature.md`,
`by-kernel-type.md`, `by-language.md`, `by-repo.md`, `by-architecture.md`.
Under `references/`: `primer.md` (topic map, read first when the question
is broad), `schema.md` (front matter, confidence, reproducibility, aliases),
`examples.md` (worked query patterns for the situations the candidate loop
meets).

## Output Pattern

When answering from this KB:

1. **Cite specific pages** with paths and ids (`wiki/patterns/serialized-wgmma.md`,
   `pattern-serialized-wgmma`).
2. **Follow `sources:`** to trace a claim to a doc, a PR page, a sibling
   skill's document, or an Agent Note (`note-<stem>` resolves to
   `.agents/notes/**/<stem>.md`).
3. **Respect confidence levels** — `verified` > `measured` >
   `source-reported` > `inferred` > `experimental`. `measured` means this
   repository's H100 with the evidence named in `evidence_basis`; call out
   `inferred` and `experimental`, and re-measure `source-reported` numbers
   before pricing anything against them.
4. **Cite machine constants by tag**, never by number: a bracketed
   `[wgmma.issue.wg.ss]` resolves through `hardware-unit-test`
   (`constants.py --tag <t>`), which owns the value and its validity range.
5. **Include the snippet** — technique, kernel and hardware pages carry a
   compilable fragment, a contiguous excerpt of a pinned upstream file or of
   a template of the sm90 bundle named by file; the validator checks the
   excerpt is verbatim.
6. **Report performance claims with all six fields** — `gpu`, `dtype`,
   `shape`, `metric`, `value`, `source_id` (plus `source_locator`), and say
   which are a template's `STATUS` measurement and which are upstream's.

## Knowledge Base Contents

- Synthesized wiki pages under `wiki/{hardware,techniques,patterns,kernels,languages,migration}`;
  source pages under `sources/{docs,blogs,contests,prs}`; verbatim upstream
  bundles under `artifacts/` pinned by SHA in `PROVENANCE.yaml`.
- The sm90 templates bundle, `artifacts/kernels/sm90-templates/variants`:
  twenty-three compilable, graded templates and their shared headers, a
  derived bundle pinned by sha256 and compile-checked by
  `scripts/check_templates.py`; `queries/by-template.md` maps each file to
  the pages that excerpt it. `get_page.py <id> --include-code` prints a
  page's bundle.
- Sibling-skill documents as sources: `doc-hardware-unit-test`,
  `doc-benchmark-kernel`, `doc-ncu-report`; the bundle's own source page is
  `doc-kernel-design-templates`.
- Controlled vocabulary in `data/tags.yaml` (tags, architectures,
  confidence, reproducibility, symptoms); alias map in `data/aliases.yaml`;
  version-sensitive claims in `data/version-claims.yaml`.
- Validator `scripts/validate.py`; index generator
  `scripts/generate-indices.py`; numeric scanner `scripts/scan_numbers.py`;
  status `scripts/repo_status.py`; tests under `tests/`.

## Quality Guarantees

- Every wiki page names at least one resolvable source; every `related`,
  `prerequisites` and `candidate_techniques` id and every relative link
  resolves.
- Every `verified` page has official-doc plus upstream-code evidence; every
  `measured` page has benchmark or reproduction evidence in
  `evidence_basis`.
- Every technique, kernel and language page carries a compilable snippet
  (`reproducibility >= snippet`).
- Every `symptoms` value is in the controlled list; every bracketed machine
  constant resolves to a `hardware-unit-test` tag.
- Every PR page carries an evidence-backed status and a verbatim upstream
  excerpt with its digest; every artifact bundle's files match their sha256;
  every template excerpt on a page is verbatim in the named file, and every
  bracketed citation in a template header resolves.

## Scope Rules

- **Hopper-first** — sm90 content is primary and is what the candidate loop
  queries (`--architecture sm90`). Blackwell pages are kept whole as the
  port-forward appendix; check `architectures:` before transferring a rule.
- **Kernel-only** — no distributed-system topics.
- **English canonical.**
- **Skills carry experience; evidence lives project-side** — a measurement's
  job ids and revisions are in the Agent Note a page cites, not in the page.
