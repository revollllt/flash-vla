# Documentation and changes

README owns setup/use, ARCHITECTURE owns module boundaries, and
[the optimization workflow](../../docs/optimization.md) owns experiment steps.
Skills provide task-specific methods. Update one owner and link to it elsewhere;
examples should help someone run the project, not duplicate implementation.

Historical plans and [Agent Notes](../../.agents/notes/README.md) explain past
work. Read a relevant note when its rationale matters; no broad note scan is
required. A routine fix or experiment needs only its diff and result. Record a
durable design decision when its alternatives or constraints would otherwise be
lost, updating an existing owner when one is already known.

Before editing, inspect worktree status and preserve unrelated changes. Before
publishing, review the scoped diff and run affected checks once. Report what was
verified and what remains uncertain. Inspect CI after push; resolve failing
required checks before merging. Do not rewrite shared history without an
explicit reason and a checked remote head.
