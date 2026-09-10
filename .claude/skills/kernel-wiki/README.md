# kernel-wiki — Hopper Kernel Optimization Knowledge Base

> **Evidence boundaries:** KernelWiki's audited upstream corpus, as copied
> here, closes at 2026-05-20 with documentation verified through 2026-08-18
> UTC. This repository's sm90 pages were migrated and validated on
> 2026-09-07; the machine constants they cite carry their own dates in
> `hardware-unit-test`.

A structured knowledge base of NVIDIA Hopper (SM90, H100) GPU kernel
optimization, packaged as a Claude Code skill inside this repository, with
Blackwell (SM100) retained as the port-forward appendix. It is
[mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki) (MIT;
`LICENSE`) — the three-layer layout, schema, validator, query tools, query
indices and every Hopper-relevant page — plus this repository's own sm90
experience written as pages of the same shape, with sources, confidence and
reproducibility the validator enforces.

## Use

```bash
cd .claude/skills/kernel-wiki
../../../.venv/bin/python scripts/query.py --symptom stall-gmma --compact
../../../.venv/bin/python scripts/get_page.py pattern-serialized-wgmma --follow-sources
../../../.venv/bin/python scripts/repo_status.py
```

The scripts auto-resolve the wiki root. The cluster's system `python3` is
3.6; the scripts need 3.7+, so use the repository venv. PyYAML is optional:
the bundled pure-Python fallback is exercised by `tests/check_yaml_fallback.sh`.

## What's Here

- Source pages (`sources/`), synthesized wiki pages (`wiki/`), generated
  indices (`queries/`), verbatim upstream bundles (`artifacts/`) pinned by
  SHA in `PROVENANCE.yaml`.
- The sm90 templates bundle (`artifacts/kernels/sm90-templates/variants`):
  compilable, graded templates whose excerpts are the pages' snippets and
  whose `STATUS` blocks are the measured numbers; a derived bundle with
  `PROVENANCE.yaml`, re-pinned with `scripts/pin_artifacts.py` after an edit
  and compiled by `scripts/check_templates.py`.
- Sibling skills as sources: `hardware-unit-test` (machine constants, cited
  by bracketed tag), `benchmark-kernel` (timing method), `ncu-report`
  (metric vocabulary and playbook).
- Agent Notes as sources: `note-<stem>` resolves to `.agents/notes/**/<stem>.md`.
- Controlled vocabulary (`data/tags.yaml`) including `symptoms` spelled after
  the Nsight Compute sm90 stall reasons; aliases (`data/aliases.yaml`);
  version-sensitive claims (`data/version-claims.yaml`).

## Query Tools

| Tool | Purpose |
|---|---|
| `scripts/query.py` | Unified search across source and wiki pages (keywords, `--symptom`, `--tag`, `--type`, `--architecture`, alias-aware) |
| `scripts/get_page.py` | Fetch any page by `id` or path; `--follow-sources` expands cited sources |
| `scripts/grep_wiki.py` | Regex text search across wiki bodies and source pages |

## Companion Docs

- [`SKILL.md`](SKILL.md) — skill entry: when to engage, five navigation paths, output contract.
- [`references/primer.md`](references/primer.md) — topic map: hardware features, techniques, patterns, kernels, symptoms → canonical page ids.
- [`references/schema.md`](references/schema.md) — front matter schema, confidence rules, reproducibility ladder, controlled vocabulary, aliases.
- [`references/examples.md`](references/examples.md) — worked query patterns for the candidate loop.
- [`CLAUDE.md`](CLAUDE.md) — extended schema and navigation reference.
- [`index.md`](index.md) — human-facing curated index.

## Architecture

Three layers (KernelWiki's, after Karpathy's LLM-wiki pattern):

1. **`sources/`** — raw data: immutable summaries of docs, papers, blogs,
   contests, PRs, and this repository's sibling skills.
2. **`wiki/`** — synthesized pages cross-referenced by `id`, all with YAML
   front matter.
3. **`queries/`** — auto-generated indices; regenerate with
   `scripts/generate-indices.py`, never edit.

Supporting files: `data/schemas.yaml`, `data/tags.yaml`, `data/aliases.yaml`,
`data/version-claims.yaml`, `data/tool-versions.yaml`; `audit/` (evidence
ledger, numeric-claims ledger, corrections, open claims).

## Maintenance Tooling

| Script | Purpose |
|---|---|
| `scripts/validate.py` | Front matter against schema, vocabulary, confidence and reproducibility rules, sources (including `note-*`), cross-references, relative links, bracketed machine-constant tags, artifact bundles, version-claims registry |
| `scripts/generate-indices.py` | Regenerate `queries/*.md` |
| `scripts/scan_numbers.py` | Unit-bearing numbers in prose, the input to the numeric ledger |
| `scripts/check_templates.py` | Compile every sm90 template and assert the PTX its header declares (needs nvcc and a compatible host compiler; no GPU) |
| `scripts/pin_artifacts.py` | Re-pin a derived bundle's `PROVENANCE.yaml` after editing its files |
| `scripts/repo_status.py` | Corpus counts |

```bash
.venv/bin/python scripts/validate.py
.venv/bin/python scripts/generate-indices.py
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
bash tests/check_yaml_fallback.sh
```

## Quality Gates and Scope

Stated once, in [`SKILL.md`](SKILL.md) ("Quality Guarantees", "Scope Rules"); `scripts/validate.py` enforces the gates.

## Provenance and License

Tooling, references and data are KernelWiki's under MIT (`LICENSE`), with
this repository's additions under the same terms. Copied wiki syntheses and
source summaries are KernelWiki's derivative works citing upstream docs,
papers, blogs and PRs; upstream bundles under `artifacts/` carry their own
licence in `PROVENANCE.yaml`. Pages added here are written from this
repository's measurements and cite the Agent Notes and templates behind them.
