# Hopper Kernel Optimization Knowledge Base — Schema

A structured knowledge base of GPU kernel optimization for NVIDIA Hopper
(SM90) with Blackwell (SM100) as the port-forward appendix, optimized for LLM
agent retrieval. The schema, layout and tooling are KernelWiki's; this file
records the additions this repository makes.

## Navigation

### Recommended: use the skill scripts
```bash
.venv/bin/python scripts/query.py --symptom <stall-reason>          # from a Nsight Compute finding
.venv/bin/python scripts/query.py "<keywords>" [--tag X --type Y --architecture sm90]
.venv/bin/python scripts/get_page.py <id-or-path> [--follow-sources]
.venv/bin/python scripts/grep_wiki.py "<pattern>" [--only wiki]
```
See `SKILL.md` and `references/examples.md`.

### Direct navigation (when skill not available)

1. **Start**: `index.md`
2. **By problem**: `queries/by-problem.md` → symptom → pattern page → candidate techniques
3. **By technique**: `queries/by-technique.md`
4. **By hardware**: `queries/by-hardware-feature.md`
5. **By kernel type**: `queries/by-kernel-type.md`
6. **By architecture**: `queries/by-architecture.md` (sm90 exact is the loop's lane)
7. **By language / repo**: `queries/by-language.md`, `queries/by-repo.md`
8. **Deep dive**: follow `sources:` ids; a `note-*` id is an Agent Note under `.agents/notes/`

## Three-Layer Architecture

- `sources/prs/{repo}/PR-{N}.md`, `sources/docs/*.md`, `sources/blogs/*.md`,
  `sources/contests/{contest}/*.md` — raw, immutable summaries. This
  repository adds `doc-*` pages for its sibling skills (`hardware-unit-test`,
  `benchmark-kernel`, `ncu-report`, `kernel-design` templates) and for the
  sm90 PTX ISA sections, FlashAttention-3, the megakernel references.
- `wiki/{hardware,techniques,patterns,kernels,languages,migration}/` —
  synthesized pages cross-referenced by `id`.
- `queries/` — generated indices; never edit.

## Page Types and Required Fields

See `data/schemas.yaml` (KernelWiki's, unchanged). Summary of the wiki types:

| Page Type | ID Prefix | Key Required Fields |
|-----------|-----------|---------------------|
| wiki-hardware | hw- | title, type=hardware, architectures, tags, confidence, related, sources, aliases |
| wiki-technique | technique- | title, type=technique, architectures, tags, confidence, reproducibility(>=snippet), prerequisites, related, sources |
| wiki-pattern | pattern- | title, type=pattern, tags, symptoms, candidate_techniques, related, sources |
| wiki-kernel | kernel- | title, type=kernel, architectures, tags, confidence, reproducibility(>=snippet), kernel_types, languages, related, sources, performance_claims |
| wiki-language | lang- | title, type=language, tags, related, sources, reproducibility(>=snippet) |
| wiki-migration | migration- | title, type=migration, from_arch, to_arch, tags, related, sources, blackwell_relevance |

## Controlled Vocabulary

All `tags` values must appear in `data/tags.yaml` under `hardware_features`,
`techniques`, `kernel_types` or `languages`; the validator rejects unknown
values. This repository adds Hopper features (`dsmem`, `cp-async-bulk`,
`mma-sync`, `setmaxnreg`, `cache-hint`, `griddepcontrol`) and the techniques
its pages name (`task-loop`, `split-k`, `megakernel`, `measurement`,
`ablation`, `fusion-pricing`, `bulk-store`, `proxy-fence`, ...).

`symptoms` is a controlled list too: KernelWiki's values plus names spelled
after the Nsight Compute sm90 stall reasons and rule names the `ncu-report`
skill emits (`stall-gmma`, `no-eligible-warp`, `smem-bandwidth-bound`,
`serialized-wgmma`, `null-ablation`, ...).

Aliases (`data/aliases.yaml`): `H100`/`Hopper` → `sm90`, `WGMMA` → `wgmma`,
`TMA`/`cp.async.bulk` → `tma`, `UMMA` → `tcgen05`, and the rest of
KernelWiki's map.

## Confidence, Reproducibility, Citations

Defined once in [`references/schema.md`](references/schema.md): the confidence levels including this repository's `measured`, the reproducibility ladder, and the two citation forms this repository adds (bracketed machine-constant tags resolved through `hardware-unit-test`, `note-<stem>` sources resolved to Agent Notes). The performance-claim record is shown there too.

## Tooling

- `scripts/validate.py` — schema, vocabulary, confidence, reproducibility,
  sources, cross-references, links, machine tags, bundles, version claims.
- `scripts/generate-indices.py` — regenerate `queries/*.md`.
- `scripts/scan_numbers.py` — unit-bearing numbers in prose.
- `scripts/repo_status.py` — corpus counts.
- `scripts/check_templates.py` — compile the sm90 templates bundle
  (`artifacts/kernels/sm90-templates/variants`) and assert its PTX.
- `scripts/pin_artifacts.py` — re-pin a derived bundle after editing.
- Interpreter: `.venv/bin/python` (3.12); system `python3` is 3.6.

## Scope Rules

- **Hopper-first**: sm90 primary; Blackwell pages are the appendix and keep
  KernelWiki's `blackwell_relevance` field where they had it.
- **Kernel-only**; **English canonical**.
- **Experience in the wiki, evidence project-side**: no job ids or revisions
  on a page; the Agent Note it cites holds them.
