# Documentation, Agent Notes, and Pull Requests

## Documentation is normative, not a second copy of the code

Documentation records interfaces, invariants, ownership, supported configurations,
safety/correctness constraints, performance measurement conditions, and
reproducible validation requirements.

Never put implementation content in documentation: no code excerpts, repeated
control-flow narratives, function or file inventories, pseudocode, generated
diffs, or review diaries. A source path or symbol is only a locator; source is
authoritative for implementation. Keep prose current-state only.

When source code changes a documented contract, configuration, invariant, or
validation requirement, update its one owning document in the same PR. If the
document only repeats implementation, delete that material or replace it with a
link to the source or generated reference; never maintain a second copy.

## Agent Notes preserve decisions

[`Agent Notes`](../../.agents/notes/README.md) defines the note lifecycle and
format. Before a non-trivial change, search active notes for an existing owner;
update that note rather than creating a duplicate.

A change is non-trivial when it affects behavior, kernel architecture or
synchronization, a shared contract, a performance policy or target,
correctness/validation strategy, tooling, or a file/configuration format.

Notes record the problem, decision or proposal, genuine alternatives,
consequences, and required verification. They do not narrate code or retain PR
history.

Non-trivial work adds or updates its Agent Note in the same PR. Proposals use
`.agents/notes/proposed/`; shipped decisions use `.agents/notes/implemented/`; rejected
proposals use `.agents/notes/rejected/`. Archived notes are frozen: never edit, move,
or use them as current authority.

## Pull-request workflow

1. Before editing, inspect `git status --short --branch`, identify the PR base,
   and review the complete diff against it. Preserve unrelated changes.
2. Keep each PR focused on one decision. Split independent changes; repair the
   introducing PR before changing dependent work.
3. Run the narrowest validation that can falsify the claim. CUDA changes need
   relevant correctness evidence; performance claims also need the stated,
   reproducible benchmark configuration and evidence.
4. Before push or review, re-inspect the diff and run selected checks once.
   Report exactly which commands ran and any unavailable evidence.
5. After push, verify the remote branch matches `HEAD` and inspect CI. Do not
   merge with required checks pending or failing. Rewrite history only with
   `--force-with-lease` after checking the remote head; raw `--force` is
   prohibited.
