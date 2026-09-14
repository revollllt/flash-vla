# Pi0.5 RTX 5090 expert FFN pointwise fusion

Hypothesis: the measured 180-call FFN chain spends recoverable time in repeated
AdaRMS and activation traversals. Preserve both torch.mm GEMMs, merge norm plus
scale into one CUDA kernel and bias plus GELU plus product into another. This
reduces the chain from 21 launches per invocation to four, with an expected
3–5 ms full-forward opportunity before integration.

References searched: RTX 5090 Pi0 pointwise CUDA, the shared H100 Gemma expert
TileLang implementation, vendored CUTLASS activation epilogues, and
[row-traversal fusion](../../.agents/skills/kernel-wiki/wiki/techniques/row-traversal-fusion.md).
The wiki measurement is sm90-specific; only the fusion pattern transfers here.
No other Target's kernels are imported. Packing weights remains a separate
experiment; this candidate keeps both GEMMs unchanged.

The new backend retains bf16 rounding of the RMS factor, both successive input
multiplications, and each GEMM result. Bias and tanh-approximate GELU run in
fp32 before the final bf16 product store. Device buffers belong to the supplied
scratch allocator and each wrapper factory. Warmup builds the CUDA library and
allocates buffers before capture.

Validation on the exclusive RTX 5090, seed 42, checkpoint
`kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha`:

- CPU Python compilation, CUDA compilation and CUDA graph capture succeeded.
- Actual activations were cloned before each reference call; weights retained
  their original addresses. Calls 0, 17, 90 and 179 all produced exact output
  and exact bf16 RMS factors: max_abs=0, rel_rms=0.
- Thirty graph samples per route, each graph cycling all 180 recorded calls
  with 18 distinct layer weight pairs, M=50, K=1024, N=4096, bf16:
  torch median 54.9115 us/call; fused median 25.8879 us/call (-52.86%).
  The corresponding 180-call difference is 5.2242 ms. These are local complete
  call-site measurements; deployed latency is still unmeasured.
- Raw samples and resolved workload identity are retained in
  `artifacts/rtx5090-pi05/gpt6-ffn-local.json` in the main checkout.

Reproduce from the candidate checkout in the target's CUDA/Python environment:

```bash
PYTHONPATH="$PWD/src:$PWD" python -m lab.pi05.rtx5090_fused_ffn --seed 42 \
  --option converted_checkpoint=<converted-checkpoint-directory> \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output artifacts/rtx5090-pi05/gpt6-ffn-local.json
```

The backend is intentionally unregistered here; serial model integration owns
route selection and deployed correctness/latency checks.

## Follow-up: pack gate and up into one GEMM

The single changed hypothesis is that M=50 pays a fixed GEMM cost twice.
The Pi0 RTX 5090 wrapper documents 10.31 us for N=8192 versus 10.34 us for one
N=4096 half at M=51; that is motivating evidence only, with a different target
and GEMM route. The new `packed_ffn` backend concatenates each immutable layer's
gate/up weights during warmup and runs `torch.mm` once into a bf16 Mx8192
scratch buffer. Both CUDA arithmetic stages are the same as `fused_ffn`, with
only activation input indexing specialized for the packed row stride.
`fused_ffn.make_wrappers` stays available as the unchanged two-GEMM route.

The weight cache belongs to the wrapper closure and retains each pair's source
tensors. Scratch owns every packed allocation, giving 288 MiB for 18 distinct
pairs plus 900 KiB for the normalized and projected activation buffers.

Using the same belt-cup seed 42 workload and 30 samples over all 180 calls:

- CUDA build and capture succeeded; calls 0, 17, 90 and 179 were again exact
  against the torch reference for both output and bf16 RMS factor.
- Two-GEMM fused median: 25.8104 us/call. Packed median: 17.1574 us/call.
  This is a local reduction of 33.53%, or 1.5575 ms over 180 calls; deployed
  latency and official parity remain the integration loop's measurements.
- Raw samples: `artifacts/rtx5090-pi05/gpt6-ffn-packed-local.json` in the main
  checkout. Add `--candidate packed` to the command above to reproduce; this
  compares the two-GEMM fused route with the packed route under one timer.
