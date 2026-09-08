# Acceptance contract

The onboarding handoff is complete only when the derived state reports
`READY_FOR_OPTIMIZATION` and the evidence supports every item below. This is a
sequencing contract for integration, not a new deployment gate; the existing
acceptance registry remains authoritative for deployability.

- [ ] upstream repository and immutable commit are frozen
- [ ] checkpoint and model revision are frozen
- [ ] official oracle outputs exist
- [ ] compatibility report answers all nine required questions
- [ ] model package exists
- [ ] Target and pipeline exist
- [ ] benchmark target is registered
- [ ] acceptance policy is registered
- [ ] official baseline adapter exists
- [ ] declaration smoke passes
- [ ] the complete correctness ladder passes
- [ ] all four baseline tiers are recorded
- [ ] call-site floor and profile evidence exist
- [ ] a persistent Campaign is created

The human is not required to write implementation code. The agent may ask for
a deployment-affecting choice only when neither the request nor official
deployment configuration resolves it; that uncertainty stays visible in the
spec until resolved.
