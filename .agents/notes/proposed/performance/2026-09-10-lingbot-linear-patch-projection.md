# LingBot linear projection for expanded vision patches

Status: proposed; route exists for qualification, shipped remains unchanged.

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
