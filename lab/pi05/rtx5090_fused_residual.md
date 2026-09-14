# Pi0.5 RTX 5090 expert gated residual fusion

Hypothesis: keep the projection GEMM and its bf16 output unchanged, then merge
casts, gate multiplication, residual addition and final bf16 store. The original
profile attributed 2.919 ms to the attention output projection and 3.363 ms to
FFN down projection over 180 calls each, with nine launches per call. Removing
the repeated pointwise launches suggested 1–2 ms recoverable model time.

The existing Pi0.5 torch call sites own the numerical ordering. The new
`fused_residual` partial backend implements both with the same `torch.mm` into
scratch followed by one hand-written CUDA kernel. Gate has width 1024; GEMM K
is 2048 or 4096. Multiplication and addition remain separate fp32 operations
under `--fmad=false`. The backend uses the current Target's CUDA build and
error-reporting path without importing another Target's kernels. Row-traversal
fusion is the same technique tested by the preceding FFN experiment.

Validation used the exclusive RTX 5090, seed 42, checkpoint
`kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha`:

- CUDA compilation and graph capture passed. Both sites' actual calls 0, 17,
  90 and 179 were bit-exact against torch: max_abs=0, rel_rms=0.
- Each invocation retains an actual activation and residual snapshot. Both
  timing routes reset the mutable output from the same residual before every
  call, including every replay. This reset copy is included on both sides.
- Thirty samples per site, cycling all 180 calls with 18 distinct layer weights:
  attention output projection median 19.1919 -> 7.7451 us; FFN down projection
  median 22.8162 -> 12.7092 us. The local sum difference is 3.8797 ms across
  180 calls of each site. This exceeds the initial estimate but is not a
  deployed latency measurement; the reset-inclusive timing boundary differs
  from the initial profiler's attributed kernel sums.
- Both wrappers share one 100 KiB scratch projection buffer within their
  factory. No device tensor is held globally.
- Raw samples and workload identity:
  `artifacts/rtx5090-pi05/gpt6-residual-local.json` in the main checkout.

Reproduce from the candidate checkout in the target's CUDA/Python environment:

```bash
PYTHONPATH="$PWD/src:$PWD" python -m lab.pi05.rtx5090_fused_residual --seed 42 \
  --option converted_checkpoint=<converted-checkpoint-directory> \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output artifacts/rtx5090-pi05/gpt6-residual-local.json
```

Registry, routing, official parity and deployed measurements belong to the
serial model integration loop and are not changed by this candidate commit.
