# Expert QKV pointwise fusion

Hypothesis: preserve the existing cuBLAS BF16 matrix multiplication while fusing
RMS factor plus scale into one kernel and factor/bias/RoPE/scatter into another.
The parent profile reported 7.768 ms over 180 calls and 4860 kernels; avoiding
most pointwise launches should recover roughly 3-5 ms at the model boundary.
The cheapest screen is the same complete call chain with synthetic inputs at
the deployed dimensions, before the parent measures actual model behavior.

Searched this Target's torch implementation, the RTX 5090 Pi0 pointwise and QKV
implementations, the shared Gemma expert kernels, vendored CUTLASS norm examples,
DeepGEMM/FlashMLA, and the row-traversal-fusion wiki page. The Pi0 arithmetic
differs; the new Target-local CUDA kernels implement Pi0.5's exact rounding
sequence without importing another Target's kernels. The GEMM is unchanged.

Environment: ssh yx5090, NVIDIA GeForce RTX 5090, CUDA_HOME configured by the
existing user-local environment file, project Python environment, torch 2.13.0+cu130.
No simultaneous GPU work during this probe. From this branch's worktree:

    source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
    PYTHONPATH=$PWD/src:$PWD /home/ubuntu/flash-vla/.venv/bin/python \
      -m lab.sm120.pi05_qkv_fusion_probe \
      --out results/rtx5090-pi05/gpt6-qkv/local.json

The probe uses x(50,1024), weight(1024,2560), Q(400,256), K/V(50,256)
as slices after a 968-token prefix, and BF16 scale/bias/rope/factor.
For synthetic seeds 42/7/19 at input magnitudes 1/0.001/1000, Q/K/V/factor
all matched torch bitwise, with zero absolute error and unchanged KV prefixes.

The existing bench_gpu_time helper times fresh graphs on their capture streams,
rotating explicit tensor inputs beyond its 5x-L2 threshold (91 calls per graph).
Each leg warms up before 30 samples. Chronological median call-chain times:

| Leg | Torch (us) | Fused (us) |
| --- | ---: | ---: |
| 1 | 53.7449 | 13.3714 |
| 2 | 53.8509 | 13.3749 |

Raw samples and numerical metrics are in local.json. This is approximately
4.02x at the synthetic local chain boundary. Its absolute savings must not be
multiplied into the model profile: synthetic weights and the helper's cache
rotation are a different measurement context. Actual checkpoint correctness
and deployed end-to-end latency remain the serial model loop's work.
