# LingBot fixed-timestep AdaRMS projection reuse

Status: accepted, Campaign iter-005, source c516eb2. Plan: lab/plans/lingbot-time-modulation.json.

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

Full qualification 610911 completed in 667 s: all gates pass, in-engine relative RMS zero, A/B/A 80.493691377/76.115327887/80.493620597 ms. Conservative same-job reduction 5.43930%. Publication remains pending because the observed driver 610.43.02 differs from the existing 570.86.10 segment. Iter-004 is retained as invalid for the requested old-driver segment, with the successful new-driver report linked in diagnostics. transition-000 was subsequently aborted without activation; no incumbent change or cross-environment speedup claim.

Original-driver job 610937 (ACD1-22, 570.86.10) completed all seven correctness checks successfully; in-engine relative RMS zero. Timing was invalid: A/B/A 90.051871 / 76.182591 / 79.529470 ms, control spread 10.522401 ms and candidate p99-min 9.085724 ms. No performance attribution or promotion from that run. Allocated cost 734 s. Timing-only confirmation 610984 reuses this exact-source correctness; no repeated full qualification. Artifacts: artifacts/optimization/lingbot-time-modulation-old-driver/{failed-timing-summary.json,confirmation-summary.json}. This confirmation subsequently passed; details below.

Timing-only confirmation 610984 completed in 303 allocated GPU seconds. A/B/A 90.379346162 / 85.210870951 / 90.378400870 ms: conservative 5.167529918 ms (5.717660269%) reduction, control spread 0.000945292 ms, candidate p99-min 0.112070702 ms. Correctness was reused from unchanged-source 610937. Accepted as iter-005. Aborted unactivated driver transition000 required restoring the original context through existing transition001: retained incumbent correctness plus actual confirmation control leg0, no extra GPU. Segment1 anchor is 90.379346162 ms; do not compare candidate 85.210871 ms directly with historical 79.653956 ms. Driver610 qualification remains separate evidence; no new-driver transition was executed. Original-driver qualification plus confirmation cost 1037 s; prior 610911 and setup costs remain in iter004. Published results validation passes for both Campaigns. The default shipped alias is unchanged; use lab/plans/lingbot-time-modulation.json for the accepted implementation.

Accepted extension: fuse the original FP32 AdaRMS formula with the existing Pi0/Pi0.5 torch.compile pattern, preserving engine-local modulation caches. Earlier BF16 factor storage is not reused. The selected plan is lab/plans/lingbot-fused-norm.json, source26196b3, Campaign iter006. The default shipped alias remains unchanged.

Isolated real-weight probe611019 (75 allocated GPU seconds) covers720 norm calls: A/B/A21.646336/1.239232/21.644352 ms, conservative20.405120 ms saving, spread0.001984 ms. Max relative RMS0.000157023 and max absolute0.0625 over synthetic hidden states; not bitwise equivalent. First compile16.58s. Existing source isolation tests9 passed2.58s.

Full qualification611022 on ACD1-25 / driver610.43.02 completed in703 allocated GPU seconds. All seven checks pass; shallow error zero, deep/multistep maximum relative RMS about0.00294. A/B/A86.392274126/64.803561196/86.418815888 ms: conservative21.588712931 ms (24.989170790%) reduction, control spread0.026541762 ms, candidate p99-min0.060283579 ms. No E2E rerun. Compiler rounding is observable but passes the existing acceptance; there is no bitwise equivalence claim.

Driver610 is a separate measurement segment2. transition002 reuses retained c516eb2 correctness from610911 with matching weights/fixture/environment, plus actual611022 parent leg0 as its86.392274126 ms anchor. This activation required no additional GPU job. Do not compare64.803561196 ms directly with historical570-driver measurements. Canonical evidence: artifacts/optimization/lingbot-fused-norm-qualification/{summary.json,qualify/,context-transition/,publication-validation.json}. Candidate total GPU cost778s includes the isolated probe. No portable derived tables are introduced.
