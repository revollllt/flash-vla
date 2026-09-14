---
name: correction-review
description: Independently check one proposed entry for docs/corrections/agent.md before it is admitted. Use in a fresh context; never in the session that reached the conclusion or wrote the retrospective about it.
allowed-tools: "Bash Read Grep Glob"
---

# Correction review

Own the admission check for a single self-reported correction. The record it
guards is in [docs/corrections](../../../docs/corrections/README.md); the
failure modes and the review list live there and in
[the workflow](../../../docs/optimization.md).

This exists because the loop's self-assessment is not independent evidence: the
reasoning that reached a wrong conclusion also writes the account of why it was
wrong, and both can be wrong the same way. One entry at a time, on request.
Not a standing gate and not a corpus sweep.

1. Read the proposed entry and the artifacts it cites. Do not read the session
   that produced it — its reasoning is the thing under test, and reading it is
   how a reviewer inherits the error.
2. **Resolve the artifact.** Re-run the measurement, read the commit, run the
   validator. An entry whose only support is a narrative — "on reflection this
   was wrong" — fails here and no further check is needed.
3. **Check contradiction, not difference.** A number taken under other
   conditions is a different measurement. Ask what the earlier conclusion
   claimed and whether this artifact makes that claim false, or merely sits
   beside it.
4. **Check the generalization against the case.** The entry names a failure
   mode; verify the case supports that mode and not a wider one. Over-reach here
   is the common failure, because a single case reads as a law to whoever just
   lived it.
5. **Check it is a mode, not a bug.** A bug is fixed once. If nothing about the
   entry would change a future decision, it belongs in the run's README as an
   ordinary result.
6. Return admit or reject, what was re-run, and what was found. Restating the
   entry's claim is not a confirmation. On admit, the finding is recorded beside
   the entry, with who checked it.

A rejected entry is not deleted work: it stays in the run's README, and the
reason it did not clear the bar is worth a line there.

## Example

> A session proposes: "tile sweeps were run against a cache; re-swept cold,
> three of nine tiles moved." Resolve `lab/sm120/pi0_cold_sweep.py` and re-run
> it. Confirm the earlier sweep reused one buffer — read that code, not the
> account of it. Then ask whether "cold" and "warm" are the same measurement
> under different conditions or whether the warm one was invalid for a route
> that cannot cache its weights, and whether the mode claimed is "an estimate
> assumed the wrong memory" or something narrower about this kernel.
