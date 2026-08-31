# Agent Note: Machine-constant tags carry their own dimensions, and the naming contract lives in the skill

> **Superseded 2026-08-30.** The compatibility layer this note describes is
> gone: every citation in the repo was rewritten to the dotted tag, the
> `aliases:`/`retired_aliases` map was deleted from `constants.yaml`, and
> `scripts/constants.py` no longer resolves an old spelling -- `--tag` on one
> now fails with `no constant tagged`. This file is the only place the
> UPPER-KEBAB names still appear, because its subject is their removal.


Status: implemented

## Decision

Every tag in `hardware-unit-test` was renamed to
`<engine>.<quantity>.<scope>[.<condition>]`, so that the measured quantity, its
unit, its denominator and its source are all recoverable from the tag alone.
Retired spellings resolve through an `aliases:` field.

The grammar, the full rename map and the migration mechanics are **owned by the
skill**, at
[`references/naming.md`](../../../.claude/skills/hardware-unit-test/references/naming.md).
The planned units that scheme was designed to accommodate are at
[`references/roadmap.md`](../../../.claude/skills/hardware-unit-test/references/roadmap.md).
Neither is duplicated here: the skill ships as a self-contained plugin, and a
second copy of its contract in this repo's notes would be the copy that goes
stale.

This note records only that the rename happened and why the contract is not
kept here.

## Consequences

- Citations elsewhere in this repo (`specs/`, `src/`, `profiles/`) keep
  resolving through `aliases:` and were deliberately **not** migrated in the
  same change, so the rename diff stays reviewable.
- `constants.py --validate` prints an `ALIAS` block listing every file outside
  the skill still citing a retired tag. That list is the remaining migration's
  to-do, and it is visible on every run rather than something to remember. It
  currently names 16 files.
- The old `TMA-*`/`MMA-*` spellings are frozen vocabulary. New constants use the
  grammar; nothing should reintroduce a kebab tag.

## Alternatives considered

- Keeping the contract in `.agents/notes/`. Rejected: the skill is meant to be
  copied into other repos, and a naming contract that lives in one project's
  notes does not travel with the thing it governs.
- Migrating every citation in the same change. Rejected: it would put `specs/`
  and kernel sources into a diff whose subject is a rename, and aliases already
  make the delay safe.
- Splitting `MMA-SYNC-DEPTH` into a latency tag and a depth tag while renaming.
  Rejected: the split needs a value that is stated in prose but never measured
  as its own constant. It is filed in the skill's roadmap as `mma.lat.warp`
  instead of invented during a rename.

## Verification

- `python3 scripts/constants.py --validate` -> `PASS sm90: 32 constants, 0
  problems`, same count before and after.
- Both spellings resolve: `--tag TMA-CTA-CEIL` prints the deprecation line and
  the body of `tma.bw.cta.dram`; `--tag tma.bw.cta.dram` prints it directly.
- `scripts/frontier.py --copy-floor` renders its provenance brackets in the new
  spelling.
