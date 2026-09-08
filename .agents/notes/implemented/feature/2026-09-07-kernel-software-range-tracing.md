# Agent Note: CUDA software tracing as diagnostic evidence

Status: implemented

## Problem

External kernel durations cannot identify internal role boundaries. Async issue,
completion observation and engine activity are different quantities; tracing can
also change the scheduling being investigated.

## Decision

Keep GPU-local integer timestamps and explicit launch/replay/CTA/role/stage
identity. Export software ranges and async observation markers to Perfetto;
retain token/generation pairing separately from physical SM placement. Unknown
coverage, migration and dropped records cannot support complete utilization
claims. Gaps are unobserved, not idle.

Use the existing attention kernel's issue calls and waits in a diagnostic source
copy. One score group and one P.V group are committed per iteration; the score
call's existing wait<0> also retires the previous P.V group. Markers bound those
calls, not exact commit instructions or Tensor Core active time. No new device
synchronization is introduced and the deployed kernel source stays unchanged.

## Alternatives considered

Synchronous WMMA alone cannot establish Hopper async behavior. A broad backend
adapter platform is unnecessary for the exercised CUDA and TVM-FFI cases.
Subtracting constant trace overhead is invalid when instrumentation changes
scheduling and memory traffic.

## Consequences

The demo's observed timer granularity is 32 ns, not a guaranteed clock accuracy.
Real attention's coarse/focused modes add about 15%/27% median isolated latency
in job 602528. They are diagnostic and unsuitable for quantitative bottleneck
shares. Independent trace-off results and exact fixture parity are retained;
there is no model speedup claim.

## Verification

Jobs 602410/602503 cover demo parity, memcheck, off removal in SASS, resource
usage and fixed off/coarse/focused samples. Perfetto v58.3 displays the real
one-CTA trace, both role tracks, and the 2.752 us WMMA software scope.

Job 602528 captures a real production attention invocation, 363 coarse ranges,
126 focused ranges and independent replay storage. Jobs 602547/602559 reproduce
its original output and pass memcheck/synccheck; the latter validates 32 TMA and
32 WGMMA observation pairs. Job 602589 checks GPU snapshot mutation and aliases.
CPU exporter/query tests cover partial captures, clock domains, identity,
reused generations and incomplete observations. Detailed conditions and raw
artifact locators are in `docs/optimization-results.md`.
