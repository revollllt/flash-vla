# Expert attention score and softmax fusion

Hypothesis: BF16-input QK with FP32 output avoids two input conversions and the
SIMT FP32 matrix multiplication, while one CUDA kernel combines score scaling,
runtime additive mask, stable softmax and BF16 probability storage. P@V remains
the existing torch BF16 matrix multiplication. The parent profile attributed
5.623 ms to 180 calls with 2160 launches; expected model recovery is 2-3 ms.

Inspected the torch call site, runtime mask construction, RTX 5090 Pi0 CUDA
masked-softmax reference, shared Gemma expert primitives, and the
row-traversal-fusion wiki. Pi0's mask and BF16 scores differ, so this Target
has its own FP32-logit CUDA softmax. No imported Target kernels, alternative
GEMM mainloop, torch.compile path, changed tolerance, or registry edit.

CPU inspection confirmed mm.dtype_out in the installed torch schema; the GPU
probe then verified BF16 inputs -> FP32 outputs on this GPU. The first compile
reported a missing math_constants.h include; adding that header fixed the
build and the following probe passed.

Environment: ssh yx5090, RTX 5090, torch 2.13.0+cu130; CUDA_HOME set by the
existing user environment. GPU and compilation were serialized with other
model work. From this branch's worktree:

    source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
    PYTHONPATH=$PWD/src:$PWD /home/ubuntu/flash-vla/.venv/bin/python \
      -m lab.sm120.pi05_attention_fusion_probe \
      --out results/rtx5090-pi05/gpt6-attention/local.json

The synthetic seed-42 screen uses Q(400,256), K/V(1018,256), additive BF16
mask(1018), and out=Q. Valid prompt counts 32 and 200 exercise runtime mask
changes; input magnitudes 1 and 4 exercise diffuse and peaked softmax.
Changing masked V rows to 10000 leaves the output bitwise unchanged.

BF16 tensor-core QK with FP32 output differs from the FP32-input reference
only in accumulation order: score rel_rms 2.38e-7 in these cases. The worst
output rel_rms is 1.565e-4 and cosine is at least 0.9999999877; the magnitude-4
case has max_abs 0.0078125. All pass existing shallow tolerances. This is
rounding drift, not a bitwise-match claim.

Timing uses the existing graph helper, explicit rotating tensors beyond its
5x-L2 threshold, 348 calls per graph and 30 samples per leg. Every timed call
first copies source_Q to Q, on both sides; this common reset is included in
the numbers and preserves the input distribution under repeated graph replay.
The reference and candidate both overwrite Q as the deployed call does.

| Leg | Torch plus Q reset (us) | Fused plus Q reset (us) |
| --- | ---: | ---: |
| 1 | 37.8573 | 15.1383 |
| 2 | 37.8271 | 15.1767 |

Raw samples and metrics are in local.json. Approximately 2.50x is credible
for this synthetic chain. Actual checkpoint accuracy and deployed end-to-end
latency belong to the parent model loop and are not established here.
