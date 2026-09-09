# LingBot linear projection for expanded vision patches

Status: accepted portable Campaign incumbent (iter-001); select lab/plans/lingbot-patch-linear.json. The Target's legacy shipped alias remains available.

The real LingBot trace (610499, source 5570f19) attributes 22.500 ms to
768 vol2col launches. Upstream Qwen patch embedding reshapes each already
expanded patch into exactly one Conv3d receptive field, with stride equal to
the kernel dimensions and no bias.

The candidate uses the same per-checkpoint weights flattened to a matrix and
F.linear on those patches. It changes only the selected policy instance's
patch-embedding method, retains the shipped RoPE-frequency cache, and preserves
the existing three-call-site route constraint.

The real-weight/canonical-fixture probe 610525 is bit-identical. Same-job graph
minima are Conv/linear/Conv 20.9962/0.011552/20.8672 ms. This supports a full-model
candidate, not an E2E performance claim. Alternative explanations include
convolution algorithm selection and accumulation-order differences on other
values. Full registered correctness, independent source binding and same-context
uninstrumented A/B/A remain required before promotion. An observed numerical
failure vetoes the optimization; the existing acceptance thresholds apply.

No Conv3d kernel rewrite, weight-value specialization, shape change or upstream
repository edit is needed. A retained Conv3d route remains the numerical and
performance control.


Full-model evidence: source d915154, official real checkpoint and canonical seed
42. Job 610585 passes all registered and official correctness checks; the
in-engine shallow, deep and multistep comparisons are bit-identical. Its first
timing attempt is retained as invalid under the 0.1 ms control-spread rule.

Timing-only confirmation 610633 reuses that unchanged-source correctness,
limits CPU affinity to four CPUs, and passes the unchanged acceptance policy:
A/B/A minima 118.018548936 / 96.200978383 / 118.044406176 ms, control spread
0.02585724 ms, conservative gain 21.81757055 ms (18.48656%). Candidate
p99-minus-min is 0.136841 ms. The gain is computed from this paired run, not
from the older 105.4086 ms anchor on ACD1-12. Jobs used 708 + 303 allocated GPU
seconds including setup; no full correctness rerun was needed.

Canonical Campaign 016bff50eb2c9ef0cbd92b17b9da996614e06f1f679306c611f3e822599755a7
now publishes iter-001 as its invariant portable incumbent. See
artifacts/optimization/lingbot-patch-linear-qualification/decision.json and
the tracked Target results for the raw-report references and source receipt.
