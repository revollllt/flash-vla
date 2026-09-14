# Pi0.5 RTX 5090 backbone attention: existing evidence

Source-only analysis; no new GPU experiment. The current RTX 5090 torch wrapper
calls the compiled `llm_backbone_attention` in
`hardware/nvidia/h100/pi05/backends/tilelang/kernels/attention.py`. Its deployed
shape is Q=(7744,256), K/V=(968,256), bf16, with scale=1/16 and a bf16
per-key mask. There are 17 attention calls: the final layer's QKV output is
consumed only through its prefix K/V.

The saved trace
`results/pi05-rtx5090/gpt6-run-01/profiles/000-backbone/0_shipped_llm_backbone.json`
contains four kernels per attention call, 68 launches total. The softmax kernel
and its two adjacent GEMMs plus output copy reproduce the existing call-site
attribution exactly:

| Stage | 17-call sum, us | Median launch, us | Share |
| --- | ---: | ---: | ---: |
| QK GEMM | 525.451 | 30.881 | 46.8% |
| Fused scale, mask and softmax | 133.153 | 7.809 | 11.9% |
| PV GEMM | 431.264 | 25.376 | 38.4% |
| Output copy | 32.992 | 1.952 | 2.9% |
| Total | 1122.860 | | 100% |

The corresponding Inductor generated module was read from the machine's
existing compiler cache, without importing or running it. Its shape assertions
are (7744,256) and (968,256), and its named softmax kernel matches the trace:
`triton_per_fused__softmax_add_mul_prepare_softmax_online_unsqueeze_0`.
The generated code establishes these actual rounding boundaries:

```text
scores_bf16 = torch.mm(Q_bf16, K_bf16.T)       # bf16 output buffer
scores_fp32 = load(scores_bf16).float()
logits_fp32 = scores_fp32 * 0.0625 + mask.float()
probabilities_bf16 = softmax_fp32(logits_fp32) # store into bf16 buffer
result_bf16 = torch.mm(probabilities_bf16, V_bf16)
out.copy_(result_bf16)
```

The compiled scale/add operations have no intervening bf16 cast. The source's
plain expression alone does not establish those intermediate roundings.
This backbone route also differs from the expert's fp32 QK path.

GEMMs account for 85.2% of this call site. Eliminating all softmax and copy time
would save only 166.145 us across the model; a practical pointwise-only change
would recover less. A standalone softmax rewrite is therefore low priority.
The next candidate should target measurable GEMM time while preserving the
bf16 score/probability boundaries. Native SDPA screening is deferred because
it changes the intermediate QK rounding; existing tolerances remain unchanged.
Profiler sums are attribution evidence, not deployed latency measurements.
