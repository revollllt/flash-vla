# Agent Note: Reject unknown route keys before backend construction

Status: implemented

## Problem

A misspelled call site silently selected the default backend, so a purported
candidate could execute an unchanged route. Shallow graphs also prune valid
registered sites that remain present in a full-model plan.

## Decision

Reject unknown registered call-site names and unavailable requested backends
before backend construction. Resolve only active graph sites, while allowing
registered sites pruned by the current workload depth. Preserve existing route
constraints and the runner's effective-route identity.

## Alternatives considered

Warning and continuing retains silent experiments. Validating only against
active graph sites incorrectly rejects shipped plans at shallow depths.

## Consequences

Typos fail on CPU without breaking full plans used for shallow validation.
Equal routes alone do not prove an unchanged implementation or host behavior.
Acceptance tolerances and performance policy are unchanged.

## Verification

Five focused binding tests pass. All 28 existing plans were checked on their
default CPU graphs; both Targets additionally resolve shipped at layers 1/18.
The representative shallow profile succeeds in job 602515. See the execution
checklist and `artifacts/optimization/cpu-tests.txt` for retained evidence.
