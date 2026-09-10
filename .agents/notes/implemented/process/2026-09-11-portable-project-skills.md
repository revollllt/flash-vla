# Agent Note: portable project skills and local environment guidance

Status: implemented

## Problem

Project methods had accumulated machine setup instructions and overlapping
workflow requirements. Following a kernel diagnosis could imply a full capture,
fixed noise threshold or legacy Campaign process unrelated to the question.

## Decision

Project skills own portable methods, short examples and optional reference
material. User-directory skills own access, scheduler launchers and environment
repairs; existing launcher directories remain ignored local working files.
Historical measurements keep their provenance and are not current setup policy.

The model workflow owns top-down profiling and serial deployment validation.
Kernel design, local timing, timeline analysis, NCU diagnosis, hardware probes,
knowledge lookup, new Target bring-up and legacy Campaign continuation each
have one skill. Detailed references are read as needed. This supersedes the
host-binding and mandatory full-report portions of the
[NCU template decision](2026-09-06-ncu-report-skill-template.md).

## Alternatives considered

Keeping a local-environment appendix in every project skill would preserve both
portability problems and duplicated ownership. Removing the technical reference
corpus would discard useful architecture-specific knowledge.

The organization draws on [KDA’s workflow/workspace separation](https://github.com/NVlabs/kda/blob/main/docs/agent-flow.md)
and [DeepSeek Harness’s focused simplification guidance](https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/skills/dsh-find-simplifications/SKILL.md).
These are structural examples, not additional project gates.

## Consequences

A new machine provides its own environment setup. Ordinary optimization does
not require legacy onboarding receipts, Campaign activation or full NCU captures.
Noise bounds come from the experiment, not a universal percentage.

## Verification

Skill metadata, changed local links and representative CLI examples are checked.
Tool-discovery and launcher migration receive focused CPU checks; no kernel or
model performance claim is made by this documentation/tooling change.
