# LingBot fixed-timestep AdaRMS projection reuse

Status: candidate, not yet accepted. Plan: lab/plans/lingbot-time-modulation.json.

Reuse the fixed-timestep precomputation method from models/pi05/weights.py.
LingBot's actual fixture uses BF16 noise/state, so retain its BF16 accumulated
time schedule and upstream float32 sinusoidal embedding followed by BF16 cast.
Pi0.5's float32 schedule and float64 embedding are not substituted. Keep each
gamma/beta projection's original BF16 arithmetic, rather than folding it into
a GEMM. The real checkpoint enables adanorm_time and separate_time_proj=false.

Each loaded engine computes ten gamma/beta pairs for its active expert norms
once, then selects the current step on each action invocation. All values belong
to that engine and are regenerated from its current weights. No table is saved
or restored as a portable artifact. Only the implementation is inherited; no
checkpoint-specific cached values may cross engine lifetimes. This runtime
rebuild is essential even though there is no external artifact recipe.

Cheap real-weight probe 610892 completed in 64 allocated GPU seconds: 72 norm
sites / 1440 projection calls per chunk, synthetic normalized activations,
bit-identical output. A/B/A 14.873536 / 9.691712 / 14.877728 ms; conservative
local savings 5.181824 ms, control spread 0.004192 ms. Not an E2E claim.

Integrated CPU check uses real modulation tensors and tests one/ten steps,
reset to step zero, and new instances with changed weights: 104 exact matches.
Existing source-engine isolation tests: 7 passed in 3.03 s. Artifacts:
artifacts/optimization/lingbot-time-modulation-qualification/{check.json,source-test.log}.

Falsifier: reject on any full-model correctness failure or same-context paired
latency without meaningful gain. Interleaving and cache residency may change
the isolated savings. Full-model comparison must use accepted RoPE source
9f7b55c, not a historical latency from another job.

Full qualification 610911 completed in 667 s: all gates pass, in-engine relative RMS zero, A/B/A 80.493691377/76.115327887/80.493620597 ms. Conservative same-job reduction 5.43930%. Publication remains pending because the observed driver 610.43.02 differs from the existing 570.86.10 segment. Iter-004 is retained as invalid for the requested old-driver segment, with the successful new-driver report linked in diagnostics. transition-000 is pending the old incumbent numerical check (610937); no incumbent change or cross-environment speedup claim.
