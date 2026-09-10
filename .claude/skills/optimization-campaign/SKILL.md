---
name: optimization-campaign
description: Inspect or explicitly resume a legacy Flash-VLA Campaign and its historical results. Use only for a requested Campaign operation; ordinary optimization follows docs/optimization.md.
---

# Legacy Campaign continuation

Own compatibility with the existing Campaign records. This skill does not own
normal kernel optimization, independent latency measurement or new Target setup.

1. Find the requested lineage through `results/index.json` and its summary/resume
   files. Recover the recorded incumbent, context and next action.
2. Read only the [continuation procedure](references/continuation.md) relevant to
   that action. Preserve the recorded protocol rather than inventing evidence.
3. Execute the requested resume, transition or publication repair with the
   existing [Campaign CLI](../../../lab/optimize/README.md).
4. Report the recovered state, completed action and any unresolved requirement.
   Do not create a replacement Campaign merely because history is incomplete.

## Example

> Resume the existing Campaign for this Target. First report its incumbent and
> next unfinished action; reuse completed results and continue from that state.

For a publication consistency check explicitly requested as part of that work:

```bash
python -m lab.results validate
```

For daily work, use the [model workflow](../../../docs/optimization.md) instead.
Machine paths and job launchers are resolved from user-local environment guidance.
