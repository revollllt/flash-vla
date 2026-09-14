# Vision residual fusion: source/trace screen rejected

Question: can vision_encoder_out_proj_residual and
vision_encoder_ffn_down_residual lose pointwise launches by replacing their
current three-launch expression with native bias/residual fusion?

Source currently computes out.add_(torch.addmm(bias, x, weight)), with res/out
aliased. Its two rounding boundaries are BF16(GEMM accumulator + bias), then
BF16(projected + residual). BF16 mm followed by a fused bias/residual kernel
would insert an extra rounding before bias and is not equivalent.

Evidence comes from the existing deployment-005 trace, without a new GPU run:

    results/pi05-rtx5090/gpt6-run-01/profiles/005-vision/0_shipped_vision_encoder.json

The replay has one cudaGraphLaunch correlation 10478, so that correlation
alone cannot distinguish call sites. Ordering, graph node ids and launch
shapes identify the 27 repeated sequences at each site. The GEMM is
cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8.
Out-projection grid is (96,3,2); FFN-down grid is (96,3,6).

Each sequence contains an 864-byte GPU memset, the GEMM, and one
CUDAFunctor_add<c10::BFloat16>. No standalone bias copy/add launch exists.
Given the source expression and this sequence, bias is handled inside the
GEMM path. The small memset appears to initialize GEMM scratch; that purpose
is an inference, while its byte count and duration are directly in the trace.

| Site, 27 calls | Memset (ms) | GEMM (ms) | Residual add (ms) | Total (ms) |
| --- | ---: | ---: | ---: | ---: |
| out_proj_residual | 0.017793 | 0.469442 | 0.040028 | 0.527263 |
| ffn_down_residual | 0.016544 | 1.215623 | 0.039905 | 1.272072 |

Median residual add duration is 1.472 us at both sites. First out-projection
graph node ids are 8589934605/606/607; first FFN-down ids are
8589934618/619/620. All 54 memsets are 864 bytes.

An independent native residual-add kernel removes zero launches and zero
tensor traversals. The proposed standalone pointwise fusion is therefore
rejected at this CPU/source screening step; no implementation or ABBA test
was added.

A different GEMM epilogue could in principle remove 54 residual-add launches
and the temporary BF16 projection's write/read, 191,102,976 bytes over both
sites at M=768,N=1152. It must explicitly round accumulator+bias to BF16
before adding the residual. The current add kernels total only 0.079933 ms,
so that broader change needs its own measured GEMM hypothesis; these bytes
and durations are headroom context, not an end-to-end speedup claim.
