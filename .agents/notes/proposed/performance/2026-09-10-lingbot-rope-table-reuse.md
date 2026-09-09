# LingBot per-forward RoPE tables

Status: candidate; no E2E gain claimed.

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

Real qualification 610771 stopped during reference construction after 238\nallocated GPU seconds: a prebound route factory bypassed per-engine isolation,\nleaking the candidate 51-token table into the 264-token reference. Iter-002 is\nretained as invalid and the incumbent remains iter-001. The controller now\nwraps every distinct registered backend factory, including prebound aliases.\nThe reference route restores utils.apply_rope directly. The affected source\nengine regression file passes all 7 tests (3.53 s), including this alias path.\nA corrected source revision will receive a new iteration; no failed numerical\nor performance evidence is relabeled as a pass.\n