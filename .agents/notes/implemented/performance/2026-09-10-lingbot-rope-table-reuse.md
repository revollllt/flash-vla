# LingBot per-forward RoPE tables

Status: accepted portable incumbent iter-003; select lab/plans/lingbot-rope-table.json.

The real action trace (610499) contains 730 sin and 730 cos launches totaling
2.353 ms; 720 of each correspond to Q/K across 36 layers and 10 steps.
Within a Qwen forward, the same position_ids feed every Q/K RoPE call.
The optional rope-table route computes trig tables once per forward and clears
them before the next forward. It retains patch-linear, frequency caching,
float32 arithmetic, checkpoint values and shape. Weight dependency is invariant.

Shape-only GPU probe 610740 used 71 allocated seconds, synthetic float32
activations and the existing frequency-cached reference. Two position offsets
are bit-identical. A/B/A conservative savings are 0.460544 ms per 51-token
forward and 0.580992 ms per 264-token forward; spreads are 0.000288/0.000576 ms.
Interleaved full-model work may reduce that gain, so these are screening numbers.

Reject on registered correctness failure, stale results after a new forward,
or no meaningful same-context uninstrumented A/B/A gain. Full-model validation
uses the currently accepted patch-linear source d915154 as incumbent.
Probe: artifacts/optimization/lingbot-rope-table-probe/result.json.

Real qualification 610771 stopped during reference construction after 238
allocated GPU seconds: a prebound route factory bypassed per-engine isolation,
leaking the candidate 51-token table into the 264-token reference. Iter-002 is
retained as invalid and the incumbent remains iter-001. The controller now
wraps every distinct registered backend factory, including prebound aliases.
The reference route restores utils.apply_rope directly. The affected source
engine regression file passes all 7 tests (3.53 s), including this alias path.
A corrected source revision will receive a new iteration; no failed numerical
or performance evidence is relabeled as a pass.


Corrected source 9f7b55c passes full qualification 610816 in 671 allocated GPU
seconds. All seven checks pass; shallow/deep/multistep outputs are bit-identical.
A/B/A minima are 84.164570086 / 79.653955996 / 84.147484042 ms. Conservative
gain against the faster control is 4.493528046 ms (5.34006%); the existing
first-control protocol records 4.510614090 ms. Control spread is 0.017086044 ms,
candidate p99-minus-min is 0.049666502 ms, and unchanged acceptance passes.

The initial bookkeeping review compared baseline_python=None from CPU
preparation with the configured official interpreter used on GPU. All other
acceptance fields are identical. acceptance-resolution.json preserves that
difference and the original review; applicability was recomputed against the
actual measured policy under the same machine environment. No thresholds,
raw measurements or correctness verdicts were changed, and no GPU retry was
needed for this metadata issue. Future CPU preparation must source the existing
reference-runtimes-lab-h100.env before capturing acceptance.

Canonical publication now retains baseline, accepted patch-linear, the invalid
first RoPE trial and accepted corrected RoPE. See
artifacts/optimization/lingbot-rope-table-qualification-retry/summary.json.
