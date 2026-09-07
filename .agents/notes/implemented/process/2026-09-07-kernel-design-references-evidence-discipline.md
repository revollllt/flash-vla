# Agent Note: the kernel-design references inherit KernelWiki's evidence discipline

Status: implemented

Related: the [kernel-design workflow note](2026-09-01-kernel-design-workflow.md)
adopted KernelWiki's organizational method and deliberately left its evidence
layer out; the [template grades note](2026-09-06-template-grades-and-wiki-linter.md)
named the runnable/structural split and deferred lifting the structural tier.
This note records what was taken from
[mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki) once
the corpus was large enough for its gaps to cost something.

## Problem

The references under `kernel-design` are consulted by every candidate loop,
and they are thin where it matters and unverifiable where they are not thin.

- Of 25 wiki entries, 21 declare `confidence: measured`; the corpus itself
  cannot support that claim for any of them, because an entry names no
  source, no note, and (for 15 of the 25) no machine constant. The linter
  checks shape and hygiene; nothing checks truth, and the grades note
  recorded that as a design choice.
- Of 23 templates, 18 are `structural`: they compile and their PTX is
  asserted, but they have never run, and their headers state design rules
  ("this halves A traffic", "why BLOCK_K is 128") as the author's synthesis
  with zero or one upstream locator per file. The five that readers find
  useful — tier 5, `40` and `42`–`45` — are exactly the ones that run, check
  themselves, carry a `STATUS` block, and cite 11–16 upstream locators each,
  with a "what was changed and why" ledger against a named upstream design.
  Quality tracks evidence, not tier.
- The upstream sources these files distil are pinned as `third_party/`
  submodules, but no entry or template cites a file at a SHA; `ext-*` entries
  link a repository URL. A reader cannot tell whether a rule was read off
  upstream code, measured here, or inferred.
- The routing table is hand-maintained; the linter can only check that a row
  exists, not that the symptom is spelled in the vocabulary the profiler
  emits. Now that the loop's kernel-level profile is an `ncu-report` pass,
  the wiki's index and the playbook's patterns are two unrelated vocabularies.

KernelWiki solved each of these. Item by item, what it has and what is here:

| KernelWiki | Here today |
|---|---|
| Three layers: immutable `sources/` (PR, doc, blog, contest summaries) -> synthesized `wiki/` -> generated `queries/` | one layer: `wiki/` + `templates/`; upstream lives in `third_party/` uncited |
| Typed page schema (`data/schemas.yaml`), ID prefixes, controlled tag vocabulary with aliases; validator rejects unknown values | five front-matter keys; tags checked against `hardware-unit-test` constants only; no aliases |
| `confidence` with enforceable definitions: `verified` requires an official-doc source and an upstream-code source; `source-reported`; `inferred`; `experimental` | `measured / source-reported / inferred`, self-declared, uncheckable |
| `reproducibility` ladder `concept < pseudocode < snippet < runnable < benchmarked`; technique/kernel/language pages must be at least `snippet` and the validator looks for a code block | grade on templates only (`structural / reference`); nothing on entries |
| `performance_claims` as six-field records (gpu, dtype, shape, metric, value, source id + locator), "shape not stated" preserved as a boundary | "a number appears only when it is the rule", unenforced |
| `sources:` on every wiki page; `--follow-sources` walks to the PR page, its verbatim excerpt, and an `artifacts/` bundle pinned by upstream SHA with per-file sha256 and license in `PROVENANCE.yaml`; `verify_verbatim.py` rechecks | no `sources` field; no locator resolution |
| Snippets on wiki pages are contiguous excerpts of pinned upstream files, never authored; a page states what it is not ("not a standalone kernel", "does not imply a universal warp count") | skeletons authored here; headers assert without boundaries |
| Pattern pages are a diagnosis flow: Symptom as the profiler shows it -> Likely causes -> Candidate techniques table -> Diagnosis checklist -> Caveats; `symptoms:` and `candidate_techniques:` fields | `Context / Move / Why / Caveats`; two of eleven patterns open with an NCU signature |
| Seven generated indices (by problem, technique, hardware feature, kernel type, language, repo, architecture) plus a primer topic map and ten worked query examples | one hand-written symptom table |
| Query tools: unified search with filters and aliases, page fetch with source expansion, regex over bodies; zero dependencies | grep |
| Audit ledgers: evidence ledger (29 authoritative sources with version, date, locator, claims supported or refuted), numeric-claims ledger (3 812 candidates: 1 121 supported, 2 corrected, 1 946 removed, 0 unresolved), 65 correction families, open-disputes page | none; evidence sits in Agent Notes the wiki may not cite |
| Version-claims registry: a claim valid for a semver range, last verified release and date, pages it applies to, bidirectional pointer validation | constants carry machine and toolchain; entries and templates do not |
| Candidate ledgers record include / defer / exclude with a reason per upstream PR | `ext-*` entries record what to read and what does not transfer |
| Scope rules stated and enforced (Blackwell-first with `blackwell_relevance` on Hopper-only pages; kernel-only; dated evidence boundary) | sm90-only stated in the README; no evidence date |
| Tests for the validator and policies; pre-commit hook; repository size budget | linters, negative-tested by hand |

The classes of error KernelWiki's audit found in its own earlier corpus are
the ones to expect here: invented pseudo-APIs, the wrong instruction kind for
a scaling mode, "universal" swizzle or fence requirements, a number without
its shape, and illustrative code presented as if it were the library
implementation.

The second axis is how KernelWiki guides an agent, independent of what its
corpus holds. Its skill is a retrieval procedure with a contract on the
answer; ours is a workflow with a reading list for the knowledge step.

| KernelWiki, as a skill | Here today |
|---|---|
| Trigger precision: the description names concrete instructions, kernels and repositories as positive triggers and carries an explicit "do not use for" clause; `argument-hint` shows the three invocation shapes; `allowed-tools` restricts what the skill may run | `kernel-design` names positive triggers only; no negative clause, no argument hint; the wiki has no trigger of its own |
| Progressive disclosure: a short entry file that points outward — primer when the question is broad, schema when writing, examples when unsure — and the corpus is queried, never read whole | one README with a symptom table; the loop says "pick the move via the README" without a procedure |
| Five navigation paths in order of preference (natural-language search, fetch by id, regex, prebuilt indices, primer), each a copy-paste command; `--follow-sources` walks provenance, `--compact` and `--paths-only` save tokens | grep |
| An output contract on every answer: cite page path and id, follow sources, name the confidence level and call out `inferred` or `experimental`, include the snippet, report a performance number with all six fields | none; a candidate's `thesis` names no entry, a quoted number names no tag |
| Ten worked examples, each a question, a navigation path, the command, and the synthesis including which number to cite from where | none |
| A primer whose technique rows carry a one-line applicability condition with its hedge ("when profiling justifies the synchronization cost") — judgment, not a catalogue | template tables say what each carries, not when to start there |
| Quality guarantees stated in the entry file so the agent knows what it may trust before citing | the grades are stated; nothing is stated about the entries |
| Dated evidence boundary in the README and a status script for corpus counts, so the skill text never carries a stale number | undated |
| Epistemic hedges as house style: "treat a concrete warp count as part of the cited configuration, not a property of the architecture", "a lower wait counter does not by itself mean lower runtime" | present in some entries, not a stated rule |
| Two audiences, two files: an agent-facing schema and navigation file, a human-facing curated index, a maintainer-facing README with refresh and validate steps | one file serves all three |

## Decision

KernelWiki is taken whole rather than imitated. A sibling skill,
`.claude/skills/kernel-wiki/`, is KernelWiki's root layout: its scripts
(query, page fetch, grep, validator, index generator, status; the PR-intake
policy trimmed to the constants the rest needs), its schema and vocabulary,
its references, all 52 of its wiki pages with the closure of the sources and
SHA-pinned bundles they cite, and its licence. This repository's 25 entries
were migrated into that schema as 11 technique, 12 pattern, 2 hardware and 2
kernel pages plus two merges, with `sources`, `confidence` and
`reproducibility` the validator enforces; the old wiki directory and its
linter are retired. The templates then moved into the wiki as a derived bundle
(`artifacts/kernels/sm90-templates/variants`), so `kernel-design` is the
flow only, as kernel-design-agents keeps it: the validator checks every page
excerpt verbatim against the named file, every header citation, and every
file's digest; `check_templates.py` and `pin_artifacts.py` moved with them. Scope is Hopper-first; the Blackwell pages stay as
the port-forward appendix. The eleven items below are what that carries, in
the order they paid off; items marked *open* are not yet done.

1. **Audit with a ledger.** Every migrated claim carries its evidence
   (`audit/evidence-ledger.md`); the 28 unit-bearing numbers on sm90 pages
   are dispositioned (`audit/numeric-claims-ledger.md`, zero unresolved); the
   six correction families are recorded (`audit/factual-errors-fixed.md`).
   *Open*: the 18 structural template headers are design statements until
   a run verifies them (`audit/unverified-claims.md`).
2. **Schema with enforceable fields.** KernelWiki's schema unchanged, plus
   `measured` as a confidence level that needs `benchmark` or
   `reproduction` evidence in `evidence_basis`, and `symptoms` as a
   controlled list; KernelWiki's alias map with the Hopper product names.
3. **A sources layer.** KernelWiki's `sources/` genre, with pages added
   for the sm90 PTX ISA sections, FlashAttention-3, the HazyResearch,
   Mirage and learn-cuda megakernels, and the four sibling skills
   (`hardware-unit-test`, `benchmark-kernel`, `ncu-report`, the
   `kernel-design` templates). DeepGEMM and FlashMLA cite the READMEs at
   the `third_party/` submodule commits, which are the commits KernelWiki
   pins. *Open*: source pages for Marlin, humming, the FlashInfer glue
   kernels and the CUTLASS Hopper collectives, wanted by item 4.
4. **Provenance locators in template headers.** *Open*: templates do not yet
   declare an origin mode or a path-at-SHA locator; the wiki side is ready
   for it (`doc-kernel-design-templates` names files, bundles carry
   `PROVENANCE.yaml`).
5. **Pattern pages are diagnosis flows.** Every migrated pattern carries
   `symptoms` from the controlled list spelled after `ncu-report`'s stall
   reasons and rules, names the playbook pattern where one applies, and has
   KernelWiki's body shape: symptom as the report shows it, likely causes,
   candidate techniques, checklist, caveats.
6. **Generated indices replace the hand table.** KernelWiki's seven
   indices, regenerated from front matter. *Open*: a by-template index and a
   staleness check on generated files.
7. **Structured performance claims and a numeric scanner.** Measured
   kernel numbers are `performance_claims` rows citing STATUS blocks;
   `scripts/scan_numbers.py` lists the rest for the ledger. *Open*: the
   scanner reports, it does not fail.
8. **One reproducibility ladder.** Pages declare it and the validator
   requires a snippet at `snippet` or above; `benchmarked` means a
   reference-grade template. *Open*: the shared host harness and the lift of
   `10`–`14` to reference grade.
9. **Version-sensitive claims carry a pointer.** KernelWiki's registry with
   two claims added (the TileLang 0.1.x PDL attribute behaviour, the CUDA 13
   ptxas C7518 observation), checked both ways by the validator.
10. **Notes are citable by id.** `note-<stem>` resolves to the Agent Note;
    a missing notes directory degrades to a printed note. Machine constants
    are cited by bracketed tag and resolved through `hardware-unit-test`.
11. **Query tools and worked examples.** KernelWiki's `query.py`,
    `get_page.py`, `grep_wiki.py`; `references/examples.md` rewritten for
    the situations the candidate loop meets.

**The boundary between the two skills**, after kernel-design-agents, which
keeps its workflow small and puts every mechanism in KernelWiki and every
profile in ncu-report-skill:

| Content | Owner | Form |
|---|---|---|
| The flow: contract, modes, draft to plan, ledger, stop conditions, promotion, parity, reference tiers | `kernel-design` | `SKILL.md`, `assets/`, `references/{loop,parity,reference-tiers}.md` |
| The two-level profiling rule and what the ledger line must contain | `kernel-design`, stated once in `loop.md` | `SKILL.md` carries only the handoff table |
| How an answer cites pages, tags and numbers; how to query | `kernel-wiki` | `SKILL.md` Output Pattern, `references/examples.md` |
| Mechanisms, moves, diagnoses, kernel studies, their evidence | `kernel-wiki` | typed pages with sources, confidence, reproducibility |
| The templates: code, grade, PTX assertions, STATUS, build line, deviations from upstream, PDL sites | `kernel-wiki`, as the derived bundle `artifacts/kernels/sm90-templates/variants` | the file and its header; `PROVENANCE.yaml` pins each file and names the sources it distils |
| A rule that holds beyond one file | `kernel-wiki` | the header cites the page id in brackets; the page cites the file for its excerpt or STATUS |
| Which pages cite which template | generated | `kernel-wiki/queries/by-template.md`; the templates README keeps no column of its own |
| The templates catalog, grades, checkers, rules for adding one | `kernel-wiki` | the bundle's `README.md` (`approach-notes`) |

Under that boundary the general essays in the endgame template headers, the
PDL essay in `sm90_common.cuh`, and the tier essays of the templates README
moved to `kernel-megakernel-forms`, `technique-pdl-placement`,
`technique-cache-policy` (an sm90 section), and two new pages,
`technique-register-unpack-sub-byte` and `technique-row-traversal-fusion`,
with `doc-cutlass-hopper` as the source for the no-sub-byte-atom fact; the
headers keep this file's reasons and a bracketed citation. The wiki's own
docs state confidence, reproducibility and citation rules once, in
`references/schema.md`.

Skill-level guidance, four items alongside the eleven above, all shipped:

12. **Trigger and invocation.** `kernel-design` gains a "do not use for"
    clause (pipeline-level profiling, a Target bring-up, a benchmark-only
    question) and an argument hint naming its modes; the wiki, once
    queryable, gets its own description so an agent mid-loop can reach it
    without re-entering the workflow skill.
13. **A consultation procedure and an output contract in `loop.md`.** The
    knowledge step becomes ordered paths with commands — query by the
    report's symptom, fetch the entry, follow its sources, open the template
    it names — and the ledger's `thesis` must cite the entry id it acts on,
    a quoted number must cite its tag or claim record, an `inferred` entry
    must be named as such, and a performance result carries gpu, dtype,
    shape, metric, value and source.
14. **A primer and worked examples.** A one-page topic map whose rows carry
    the applicability condition and its hedge for every technique and every
    template ("start here if"), and an `examples.md` of the situations the
    loop meets — the report names Pattern Q, the contract prices a fusion, a
    library kernel is faster at the same shape — each mapped to the commands,
    the pages, and what the ledger line should say.
15. **Stated guarantees and a dated boundary.** The skill entry states what
    the linter guarantees about entries and templates and the date the corpus
    was last audited; counts come from a status script, never from prose.

Not inherited: the PR corpus at scale (no payer for a fixed-workload sm90
target), Blackwell content, the multi-model audit quorum, and verbatim
fragments in place of skeletons.

## Alternatives considered

- **Install KernelWiki as a sibling skill and keep this wiki as is.**
  Rejected: its instruction vocabulary is Blackwell-first, so the sm90 rules
  an agent needs would sit beside a corpus that cannot confirm them, and the
  inaccuracy here is structural, not a shortage of pages.
- **Keep the wiki evidence-free and fix wording where it is wrong.**
  Rejected: nothing would stop the next entry from decaying the same way;
  the linter needs a claim to resolve, not prose to admire.
- **Lift every template to reference grade first.** Rejected as in the
  grades note: each of the five runnable templates rolls its own harness; the
  shared harness comes before the lift, and the audit comes before both.
- **A full sm90 PR corpus with candidate ledgers.** Rejected: the sources
  layer is the doc and upstream-summary genre only, capped at about twenty
  pages, because the loop consults it, not a search engine.

## Consequences

- The candidate loop's knowledge step is a query (`loop.md`): by symptom
  from the report, then the pattern's candidate techniques, then the
  template; the ledger's `thesis` names the page it acts on and numbers cite
  tags, STATUS blocks or claim records.
- `kernel-design` is the flow only: SKILL, contract, loop, parity, reference
  tiers; its SKILL carries a "do not use for" clause and an argument hint;
  `ARCHITECTURE.md` and `ncu-report` point at `kernel-wiki`, and
  `sbatch/kernel_template.sh` builds from the bundle.
- Two spellings of "page id" now exist in templates and notes: the old
  entry ids were renamed to typed ids (`pattern-`, `technique-`, `hw-`,
  `kernel-`) at their four template citation sites and in the notes that
  linked them.
- The skill needs the repository venv (`.venv/bin/python`); the cluster's
  system `python3` is 3.6.

## Verification

On the login node, 2026-09-07:

- `kernel-wiki/scripts/validate.py`: all files valid on the full corpus
  (counts from `repo_status.py`); the by-problem index regenerated with
  the twelve sm90 pattern rows; `query.py --symptom stall-gmma` returns
  `pattern-serialized-wgmma`.
- `kernel-wiki/tests`: 8 contract tests pass (measured-evidence rule,
  evidence source in page sources, unknown symptom, dangling candidate,
  unresolved machine tag, dead relative link); `check_yaml_fallback.sh`
  passes with `-S`.
- `kernel-wiki/scripts/validate.py` resolves every tag and page citation in
  the template headers and every page excerpt against the named file;
  `kernel-wiki/scripts/check_templates.py` compiles 23/23 after the move.
- No GPU time; no kernel source changed beyond four citation comments.

## Acceptance criteria (as proposed)

- `check_wiki.py` resolves at least one source per entry, enforces the
  confidence rules, fails a stale generated index, and reports zero
  unresolved numbers; every `pattern` entry's symptoms resolve to the
  ncu-report vocabulary.
- Every template declares an origin mode; every `ported` or `distilled`
  template carries at least one locator the checker resolves.
- `references/audit/claims.md` exists, closes with zero open items, and
  states how many claims were downgraded or removed.
- Templates `10`–`14` are reference grade with a `STATUS` block taken on this
  cluster; the shared harness is one header.
- `loop.md` carries the consultation procedure and the output contract;
  `examples.md` exists with at least one example per loop situation named
  above; the skill entry states its guarantees and audit date.
- The kernel-design workflow note is updated to say the references carry
  their evidence, superseding the "skills carry distilled experience only"
  consequence for the sources layer.

## Risks

- Scope creep into a corpus project. Mitigated by the sequencing (audit and
  schema first) and the twenty-page cap on sources.
- Portability of the skill weakens with note citations. Mitigated by
  resolving by id only and degrading to a printed note when the notes
  directory is absent.
- GPU time: each reference-grade lift needs a job on an ncu-capable node for
  its `STATUS`; budget it per archetype, not as one campaign.
- Vendored excerpts carry upstream licences; `PROVENANCE.yaml` records the
  licence per bundle as KernelWiki does, and excerpts stay bounded.
