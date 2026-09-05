# Agent Note: template grades, and a checker for the wiki

Status: implemented

## Problem

Two gaps opened once the megakernel templates became runnable, self-checking
machines with measurements in their headers
([note](2026-09-05-megakernel-reference-templates.md)).

The first is that the library became two kinds of file with nothing saying so.
Eighteen templates compile and have their PTX asserted but have never been run;
five run, check themselves and report numbers. A reader had no way to tell
which kind a file was without reading it to the end, and the templates README
told them the wrong thing — "structurally verified only" was written when it
was true of everything and stayed after it stopped being true of five files.
Two spellings of the evidence block had also appeared (`MEASURED` in template
40, `STATUS` in 42-45) for the same thing.

The second is that the wiki has a documented shape — front matter vocabulary,
required sections, a symptom row per entry, citations by tag — and nothing
checked any of it. Templates have had `check_templates.py` since the library
was created; the wiki, which is the layer a reader consults first, had no
equivalent. Every one of its invariants decays silently: a renamed machine
constant leaves a citation resolving to nothing, a deleted entry leaves live
links from three others, a new entry nobody added to the routing table is
reachable only by grep.

## Decision

- **Every template declares a grade**, `// CHECK-GRADE: structural |
  reference`, and `check_templates.py` enforces it in both directions. A
  `reference` template must carry `main`, a `STATUS` block and the nvcc build
  line that reproduces it. A `structural` template must say so in its header
  and must NOT carry a `STATUS` block — a template quoting numbers no harness
  in the file can re-take is how a measurement outlives the machine it was
  taken on. A missing or unknown grade fails. The output prints the grade per
  file and tallies both kinds.
- The grade is deliberately **not a quality ranking**, and the split is not a
  backlog. A ladder template exists to show one mechanism; a harness wrapped
  around it buries what it is there to show. The grade states what evidence
  exists, so a reader knows what they are entitled to believe.
- `STATUS` is the one spelling of a template's evidence block; template 40's
  `MEASURED` heading was renamed to it, content unchanged.
- **`scripts/check_wiki.py` checks the wiki against the rules its own README
  states**: front matter (the five keys, in order, from a closed vocabulary,
  `id` equal to the file stem), the sections each entry type requires, a
  symptom row for every entry, every relative link resolving, and every
  bracketed citation resolving — machine-constant tags through
  `hardware-unit-test`, entry ids to live entries — across the templates as
  well as the wiki. It also fails the project residue the wiki forbids: job
  ids, project paths, `flash_vla`, "our kernel".
- Tag resolution **imports `constants.py` rather than reading its YAML**, so
  `hardware-unit-test` stays the sole authority on the tag namespace. A
  missing sibling skill degrades to a printed note, not a failure: the wiki is
  portable and may travel without it, and a checker that refuses to run is
  worse than one that says what it skipped.
- Citations are harvested from **prose only** — front matter stripped (a
  `tags:` list is a taxonomy, not a citation), fenced blocks stripped (a format
  example is not a claim), markdown link syntax handled as links, and in a
  template only comment lines read, so an array subscript is never mistaken for
  a citation. Classification is by shape: dotted tokens are machine tags,
  kebab tokens are entry ids, everything else is prose.

## Alternatives considered

- **A grade inferred from the file's contents** rather than declared: rejected.
  Inference cannot fail, and the value here is that a file states its claim and
  the checker holds it to it. A template that loses its harness should fail,
  not be silently downgraded.
- **Requiring every template to be reference-grade**, treating the 18 as debt:
  rejected for now as a separate decision. It is worth doing for the archetypes
  whose headers carry unbacked performance claims, but it needs a shared host
  harness first (each of the five runnable templates currently rolls its own
  graph capture, argument parsing, min-of-N timing and rms comparison) and it
  is a much larger change than naming the split.
- **Folding the wiki checks into `check_templates.py`**: rejected. That script
  needs nvcc and takes minutes; the wiki checks need neither CUDA nor a GPU and
  should be runnable anywhere, including where only the skill was copied.
- **Stripping inline code spans when harvesting citations**, so documentation
  can show a bracketed placeholder: rejected. The corpus writes citations bare
  and examples in backticks, but the distinction is a convention a linter
  should not depend on; the two README sentences that tripped their own check
  were rewritten instead.
- **Checking that `confidence: measured` cites a tag or a number**: rejected as
  unsound. Confidence is a claim about how the entry was established, which
  lives in the Agent Note behind it, not in the entry's surface text.

## Consequences

- The templates README no longer claims the library is structurally verified
  only; it carries the two grades and what each may claim, and the tier-5
  section points at the grade rather than restating the harness requirements.
- The wiki README's entry format now describes external entries correctly.
  They open with `What to read it for` and close with `Source`; the previous
  text required `Context` and `Move` of them, which no external entry has ever
  had.
- A new template cannot join the library without stating what it is, and a new
  wiki entry cannot join without a symptom row, which is what routes a reader
  to it.
- What neither checker proves is unchanged and stated in both: that a kernel
  computes the right values, that an entry is still true, or that a `STATUS`
  block is current. The STATUS block names its machine and toolchain so a
  reader can judge the last of those.

## Verification

- `python3 .claude/skills/kernel-design/scripts/check_wiki.py` — 25/25 entries
  pass, 0 corpus-wide failures, 20 tag citations and 6 entry citations
  resolved. Negative-tested on a throwaway copy of the tree, one fault at a
  time: an undeclared front-matter key, an `id` not matching the stem, an
  invalid `confidence`, a renamed `## Move`, an injected job id / project path
  / "our kernel", an entry deleted from the routing table, a dead relative
  link, a citation to a non-existent entry, and a machine tag resolving to no
  constant. Each fails the run; the two legitimate links and every real tag in
  the same runs continued to pass.
- `python3 .claude/skills/kernel-design/scripts/check_templates.py` — 23/23
  pass (5 reference, 18 structural) on the login node, `cuda/13.1`,
  `gcc/13.3`. Grade enforcement negative-tested against the real files: no
  grade, an unknown grade, a structural file carrying a `STATUS` block, and a
  reference file with its `STATUS` block, `main` or build line removed each
  fail; the two unmodified files pass.
- No GPU time. This change touches no kernel source: the only edit inside a
  template is one declaration line per file, plus template 40's heading rename.
