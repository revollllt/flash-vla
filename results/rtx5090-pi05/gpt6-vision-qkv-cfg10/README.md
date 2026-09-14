# Vision QKV on the existing cfg10 bias GEMM

Hypothesis: reuse the existing cutlass_vision config-10 bias GEMM for
vision_encoder_norm_qkv, while retaining exactly the existing fused_vision
LayerNorm. This tests one already-built tile at M=768,K=1152,N=3456.
No CUDA source, tile, native ABI, or additional GEMM implementation is changed.

Prepared in a clean branch from 5f7ccf0. The Python production change adds one
wrapper/export, reusing _norm and the existing gemm closure. Current shipped
QKV remains the control: fused_vision LayerNorm plus torch.addmm.
Registry and Target plan changes belong to the parent integration step.

Before the main branch's later workspace/plan ABI changes, its existing
native libraries were copied with ordinary copyfile into this worktree's
cache. Their sources matched this branch at that point. No hashes, source
recompilation, or subsequent refresh from main was used. Initial syntax
checking used AST only, without importing torch; GPU execution began only
after the parent granted an exclusive slot.

Command on ssh yx5090, RTX 5090, torch 2.13.0+cu130:

    source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
    PYTHONPATH=$PWD/src:$PWD /home/ubuntu/flash-vla/.venv/bin/python \
      -m lab.sm120.pi05_vision_qkv_cutlass_probe \
      --checkpoint /home/ubuntu/models/pi05_belt_cup_pytorch \
      --checkpoint-id kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
      --out results/rtx5090-pi05/gpt6-vision-qkv-cfg10/local.json

All 27 actual seed-42 layers were captured from the shipped trajectory and
checked. Candidate output is bitwise equal to fused-vision at every layer,
with zero absolute/RMS error. The actual x shape is (3,256,1152), weight is
(1152,3456) with strides (3456,1), and every bias is BF16 (3456).

Each graph cycles all 27 immutable input/weight snapshots and fully overwrites
the QKV output. Both paths use the same no-reset policy, separate runner
scratch, 27 warmup calls, 30 samples and fresh captures in ABBA order.
The 27 weight data pointers are distinct. Inputs, affine vectors, weights
and biases total 263,077,632 bytes; there is no artificial cache flush.

| Leg | Backend | Median (ms/27 calls) | Q1 | Q3 |
| --- | --- | ---: | ---: | ---: |
| 1 | fused-vision | 1.010784 | 1.007640 | 1.013328 |
| 2 | cfg10 | 0.962976 | 0.961352 | 0.965216 |
| 3 | cfg10 | 0.964240 | 0.962328 | 0.967584 |
| 4 | fused-vision | 1.026688 | 1.022976 | 1.031624 |

The closest median gap is 0.046544 ms (about 4.6% local). Control legs drift
by 0.015904 ms, while the cfg10 legs differ by 0.001264 ms; both cfg10 IQRs
remain below both control IQRs. This supports handing the small candidate to the parent
for deployed end-to-end evaluation. It is not an end-to-end gain claim.

Raw samples and all layer metrics are in local.json. Only this cfg10 route
was tested; this result says nothing about untested tiles.
