# Agent Note: LingBot RoPE frequency cache

Status: implemented

## Problem

LingBot rebuilt the input-independent inverse-timescale vector for every RoPE
call. The action stage repeats that work across 36 layers and 10 denoise steps.

## Decision

The shipped LingBot route caches each inverse-timescale vector after warmup,
keyed by width, dtype, device, and wavelength. Position conversion, radians,
sine, cosine, and rotation remain per call and retain upstream arithmetic.

The three LingBot call sites remain one atomic route because they share one
loaded upstream model. Promotion uses the canonical Robotwin seed-42 fixture,
the frozen checkpoint and dependencies, and latency-v2 A/B/A evidence.

## Alternatives considered

- Caching positions or sine/cosine was rejected because positions differ
  between prefix and denoise calls.
- Kernel redesign was deferred because the cheaper invariant removal already
  cleared the promotion bar.

## Consequences

LingBot `shipped` selects the cached route. The reference route remains the
uncached upstream implementation. The optimization does not change Target
shape, precision, checkpoint, fixture, or numerical outputs.

## Verification

Campaign `lingbot-vla-4b-h100-bf16` iteration 4, Slurm job 606866: every
in-engine and official correctness check passed; candidate chunk minimum was
113.724 ms versus 120.091 ms for the same-process control, with 0.019 ms
control spread. The promotion gate verdict was `pass`.
