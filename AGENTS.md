# Agent instructions

For optimization, start with [docs/optimization.md](docs/optimization.md). Use its
direct commands and load only the skill needed for the current step.
[ARCHITECTURE.md](ARCHITECTURE.md) owns module boundaries; source owns API details.
Historical plans and Agent Notes do not override this path.

Project skills live in [.agents/skills/](.agents/skills/). Keep this as the single
source; `.claude/skills` is a relative symlink for Claude Code compatibility.
Add new skills under `.agents/skills/` so both entries expose them.

- State one testable hypothesis and use the cheapest informative check first.
- Control workload, environment and noise before attributing a performance change.
  Expose uncertainty rather than assuming missing evidence.
- Make the smallest useful change. Reuse code and measurements; do not add
  hashes, frozen contracts, baselines, gates or speculative abstractions.
- Run affected tests using [tests/README.md](tests/README.md). Preserve unrelated
  work and keep errors visible. No default corpus-wide review or hashing.
- Update the one owning document when behavior changes. A short experiment log
  is enough; use an Agent Note only for a durable design decision that needs rationale.
- Project instructions use relative paths. Machine paths, job launchers and local
  environment repairs belong in user-directory skills or ignored files.

Read the relevant [coding rules](.claude/rules/) when changing implementation.
