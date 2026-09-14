# Task briefs

One brief per entry point, carrying this round's values and nothing else. The
procedure lives in the skill named on the first line and is injected in full by
the launching agent — binding the skill is what puts its text in context; the
`$name` line alone only asks the model to go looking for it. Keep both the
injected text and the filled-in values in the run's record, so a later
comparison can show two agents were given the same instructions.

| Brief | Skill | Use |
|---|---|---|
| [optimize.md](optimize.md) · [en](optimize.en.md) | `model-optimization` | Cut a registered Target's end-to-end latency |
| [onboard.md](onboard.md) · [en](onboard.en.md) | `target-onboarding` | Bring up a model, or a new GPU for one |

Nothing about *how* to do the work belongs here. A brief that starts explaining
method is text that belongs in its skill.
