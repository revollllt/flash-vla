# Cooperative out-projection/RMS/XFS producer with PDL consumer

Date: 2026-08-28

## Decision

Use one 128-CTA cooperative TileLang kernel,
`tl_out_proj_residual_rms_xfs`, for the fixed Pi0.5 decoder shape. It writes
the in-place BF16 residual, exact FP32 row-square partials, resets readiness
counters, crosses one grid sync, then emits contiguous BF16 XFS `[1024,64]`.
Every CTA triggers programmatic completion after the grid sync. The existing
132-CTA persistent FFN consumer remains a PDL launch and waits at kernel entry.

The production wrapper launches this cooperative producer once. The old
`tl_out_proj_residual_partials` and `tl_rms_xfs_from_partials` kernels remain
baseline-only for isolated comparisons.

## Alternatives considered

- Keep the two-launch split producer: correct, but leaves launch/tail overhead
  on the critical path.
- Patch the TileLang launcher to combine cooperative and PDL launch attributes:
  unnecessary because the producer only needs a cooperative launch; PDL is on
  the project-owned CUDA consumer launcher.
- Hand-write the RMS/XFS tail in CUDA or continue local TileLang geometry
  tuning: measured candidates did not supply the required end-to-end margin.

## Consequences

- The fused producer needs cooperative-launch residency and a grid-wide sync.
- `square_partials` remains an explicit FP32 scratch contract.
- Counter reset is part of the producer; the production graph must not contain
  `reset_ffn_counters_kernel`.
- PDL overlap is preserved without changing TileLang's global runtime.

## Required targeted verification

- Residual and XFS bit-exact against the split path for three fixed-shape sets;
  padded XFS columns must stay zero.
- Replayed counters must finish at hidden `4` and down `3`.
- Same-process A/B/A timing must show at least 1.0 us cumulative gain, at least
  0.20 us over the split full chain, and no more than 0.05 us standalone
  producer regression.
- Profiler must show exactly one cooperative producer name, no split A/B names,
  no standalone reset, and one 132-CTA persistent consumer.

## Evidence

- Job 565549 selected the cooperative candidate: minimum full-chain gain
  `+0.363 us`, maximum standalone producer regression `+0.027 us`.
- Job 565569 passed exactness and authoritative full-table A/B/A: minimum
  cumulative gain `+2.141 us`, minimum split-chain gain `+0.613 us`.
- Job 565575 verified the promoted production wrapper: minimum cumulative gain
  `+2.107 us`, minimum split-chain gain `+0.768 us`.
- Job 565587 replayed the one-call production graph 180 times: old midpoint
  `4700.191975 us`, production `4352.704048 us`, gain `347.487926 us` total or
  `1.930488480 us` per call. Its profile contained one
  `tl_out_proj_residual_rms_xfs_kernel`, one 132-CTA `ffn_taskloop_kernel`, no
  split producer, and no standalone reset.
