# Expert action-output pointwise fusion

Hypothesis: replace the final action projection's RMS-factor and output
pointwise chain with two dedicated CUDA kernels around unchanged torch.mm.
Deployment-010 attributes 0.217349 ms / 170 kernels / 10 calls to this site;
the initial expected headroom was 0.1-0.15 ms with roughly 17 -> 3 launches
per call. No new GEMM implementation is introduced.

The existing Target-local fused_qkv native library gains a private RMS-factor
kernel, a factor/bias/Euler-residual update kernel, and the corresponding
action_expert_action_out_proj wrapper. This reuses its lazy loader and runner
scratch, keeping the implementation small. The existing QKV kernels and
arithmetic are unchanged.

RMS reduces in FP32 then rounds its private factor to BF16. torch.mm writes
a BF16 projection; the update multiplies by the BF16 factor in FP32, adds bias,
then adds the prior action value with separate FP32 operations before final
BF16 storage (--fmad=false). The public norm_factor tensor is never written.
make_wrappers with CUDA_HOME removed was checked on CPU and did not load
the native library. No registry or Target plan change belongs to this commit.

On ssh yx5090, RTX 5090, torch 2.13.0+cu130, existing CUDA environment, with
exclusive GPU/compilation ownership:

    source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
    PYTHONPATH=$PWD/src:$PWD /home/ubuntu/flash-vla/.venv/bin/python \
      -m lab.sm120.pi05_action_out_fusion_probe \
      --checkpoint /home/ubuntu/models/pi05_belt_cup_pytorch \
      --checkpoint-id kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
      --out results/rtx5090-pi05/gpt6-action-out/local.json

The probe snapshots all ten actual seed-42 action-expert inputs from the
reference trajectory: x(50,1024), weight(1024,32), bias(32), and initial
out(50,32). Candidate output is bitwise equal at every step, with zero
absolute/RMS error. Both paths leave norm_factor bitwise unchanged.

This is warm-cache local timing. The ten weights have distinct data pointers
but share one storage allocation, at element offsets 0,32768,...,294912.
The snapshotted tensors plus initial-output copies total 1,745,000 bytes.
The graph cycles these same ten cases without a cache flush; this does not
represent cold weights after the full expert's preceding weight stream.

Both timed paths copy the same initial action output before every update,
so repeated graph replay does not iterate the Euler update on changed inputs.
The existing graph helper uses the capture stream, ten warmup calls, ten
cases per graph and 50 samples per leg, with fresh captures in ABBA order.

| Leg | Backend | Complete ten-call chain including resets (ms) |
| --- | --- | ---: |
| 1 | torch | 0.234208 |
| 2 | fused | 0.066256 |
| 3 | fused | 0.066336 |
| 4 | torch | 0.234160 |

Raw samples and correctness metrics are in local.json. Local savings are
approximately 0.168 ms under these warm-cache conditions. The parent model
loop must determine whether they survive official accuracy checks and the
deployed end-to-end measurement; no model speedup is established here.
