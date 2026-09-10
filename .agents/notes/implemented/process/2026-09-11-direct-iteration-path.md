# Agent Note: a direct optimization path

Status: implemented

## Problem

Current instructions repeated obsolete identity and workflow rules. Everyday
measurements imported formal qualification settings, historical workflow tests
shared the default test suite, and the Pi0/Pi0.5 expert kernels had identical
implementations in separate Target modules.

## Decision

README owns getting started, ARCHITECTURE owns dependency and execution boundaries,
and docs/optimization.md owns the experiment loop. Skills explain specific tools.
Agent Notes and archived plans explain past decisions; they do not override those
current documents or require a new decision record for every experiment.

Direct benchmark defaults are independent of qualification. Repeated controls
report observed drift; a rejection threshold is applied only when explicitly
requested by a qualification consumer. Existing Campaign/publication commands
remain available for their real consumers, with their tests outside the default
runtime/benchmark suite.

Identical expert kernel builders are shared in the existing component package.
Targets retain their own JIT registries, pass settings, shapes and routing, so
shared source does not introduce import-order-dependent tuning behavior.

## Alternatives considered

A repository-wide runtime rewrite would change working execution boundaries
without evidence of benefit. Renaming every historical CLI would either break
retained consumers or require compatibility wrappers. Keeping the interfaces
while separating their dependencies and tests avoids both costs.

## Consequences

Daily work uses direct correctness, latency and profiler commands. Historical
results stay readable. Kernel sharing changes source ownership, not arithmetic
or the deployed plans. Only affected checks are needed for an iteration.

## Verification

Validate current documentation links, direct CPU tests, affected legacy consumers,
and the shared kernels on their existing Pi0/Pi0.5 execution paths. No performance
improvement is claimed by this refactor.
