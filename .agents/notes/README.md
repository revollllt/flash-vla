# Agent Notes

`.agents/notes/` holds durable decision records. Decisions retain their problem,
alternatives, consequences, and verification; source code remains authoritative
for implementation.

## Layout and lifecycle

Each note is `{lifecycle}/{class}/YYYY-MM-DD-topic-title.md`.

- `proposed/`: a reviewable decision not yet shipped.
- `implemented/`: a decision represented by the current repository state.
- `rejected/`: a declined proposal whose rationale remains useful.
- `archived/`: a sealed, historical implemented decision; never edit, move,
  or treat it as current authority.

Valid classes are `feature`, `bug-fix`, `simplification`, `architecture`,
`process`, and `testing`.

Every non-trivial PR adds or updates the active note that owns its decision.
Search before creating a note. Do not duplicate a decision; cross-link partial
supersessions. A fully superseded decision may be archived only after its
successor preserves any rationale that remains useful.

## Format

Each active note starts with:

```markdown
# Agent Note: <title>

Status: <proposed | implemented | rejected — reason>
```

A proposed note contains `## Problem`, `## Proposal`,
`## Alternatives considered`, `## Acceptance criteria`, and `## Risks`.
An implemented note uses `## Decision` and `## Consequences` in place of
proposal-only sections, and states its current `## Verification`.

## Content boundary

Write decision-level, current-state facts only. Link to source, tests,
benchmark commands, or artifacts solely as locators; never restate their code,
control flow, or PR history. A material reversal needs a new cross-linked note.
