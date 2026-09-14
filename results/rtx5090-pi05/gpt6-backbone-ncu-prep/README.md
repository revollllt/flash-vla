# Focused NCU preparation: deployed backbone gate

Question: is the current Stream-K gate GEMM (M=968,K=2048,N=16384) limited
by tensor work, DRAM/L2 traffic, or insufficient eligible work/occupancy?
Reuse deployed cutlass_backbone.py/.cu config 0: Sm80 tensor op,
CTA 128x128x64, warp 64x64x64, 3 stages, compiled for sm_120a.
No alternate collective, source rebuild, or full-set collection is prepared.

Status: CPU/source preparation only. The module has passed Python syntax
checking; neither operand preparation nor GPU profiling has run. No sudo
command has been executed. NCU --version reports 2026.2.1.0.
The initial live-device metric query failed ERR_NVGPUCTRPERM; specifying
--chips gb202 performs an offline query and verifies all eight metrics below
against the installed NCU. Raw selected query rows are in supported-metrics.csv.

| Metric | Question |
| --- | --- |
| gpu__time_duration.sum | Kernel-only diagnostic duration |
| sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed | Tensor activity including idle SM/time |
| dram__throughput.avg.pct_of_peak_sustained_elapsed | DRAM utilization |
| lts__throughput.avg.pct_of_peak_sustained_elapsed | L2 utilization |
| sm__warps_active.avg.pct_of_peak_sustained_active | Achieved resident warp share |
| smsp__warps_eligible.avg.per_cycle_active | Ready work available to schedulers |
| smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio | Global/local/texture load dependency stalls |
| launch__waves_per_multiprocessor | Grid/residency context for tail hypotheses |

First prepare the actual operands as an ordinary unprofiled process, from the
current deployed project checkout. This runs the shipped model up to the
backbone, saves layer-0 x_norm and its already-folded gate weight, then exits.
The snapshot carries its checkpoint/seed/engine identity. Earlier saved
oracle files do not contain this normalized input, so it must be captured.

    source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
    PYTHONPATH=$PWD/src:$PWD /home/ubuntu/flash-vla/.venv/bin/python \
      -m lab.sm120.pi05_backbone_gate_ncu prepare \
      --checkpoint /home/ubuntu/models/pi05_belt_cup_pytorch \
      --checkpoint-id kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
      --snapshot artifacts/rtx5090-pi05/backbone-gate-layer0.safetensors

The profiling mode loads only those two tensors, preserving BF16 contiguous
row-major strides (2048,1) and (16384,1). It loads the existing deployed .so
directly with ctypes and reuses the production _Plan; it never calls nvcc.
Fifty unprofiled warmup gate calls precede one eager gate launch inside the
pi05_backbone_gate NVTX push/pop range. CUTLASS GemmUniversalBase::run in the
vendored source launches one Kernel2; workspace setup occurs before warmup.

Suggested capture command (the parent owns execution/counter authorization):

    NCU_GATE_METRICS='gpu__time_duration.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,launch__waves_per_multiprocessor'
    PYTHONPATH=$PWD/src:$PWD /opt/nvidia/nsight-compute/2026.2.1/ncu \
      --config-file 0 --nvtx --nvtx-include 'pi05_backbone_gate/' \
      --launch-count 1 --replay-mode kernel --cache-control all \
      --clock-control none --disable-extra-suffixes \
      --metrics "$NCU_GATE_METRICS" \
      --export artifacts/rtx5090-pi05/backbone-gate-layer0-cold \
      /home/ubuntu/flash-vla/.venv/bin/python \
      -m lab.sm120.pi05_backbone_gate_ncu profile \
      --snapshot artifacts/rtx5090-pi05/backbone-gate-layer0.safetensors \
      --library .cache/cuda_ext/rtx5090_pi05_cutlass_backbone/libcutlass_backbone.so

The trailing slash selects the push/pop range; warmups are outside it.
--cache-control all flushes caches before replay passes. This is the initial
cold-weight diagnostic: deployed gate/up weights stream across many layers,
but the real normalized A was just produced and may be warm. The isolated
cold capture therefore does not exactly recreate producer-to-consumer locality.
Warmups still initialize the plan, code and clocks; they do not override NCU's
cache flush. --clock-control none preserves the unlocked-clock policy, so
counter/duration variation must be considered before attributing small changes.

Only if cache residency remains the unresolved question, a second identical
capture with --cache-control none and a separate export path can bracket
warm/replay conditions. Repeatedly using one gate weight is a warm-cache limit,
not the deployed 17-layer gate/up stream. Do not compare those captures as
optimization versions.

Interpret tensor, DRAM and L2 together. Low tensor with low bandwidth and few
eligible warps points toward latency/scheduling; low achieved occupancy by
itself is not evidence against Stream-K's persistent scheduling. Waves/SM
needs the launch's block/register/shared-memory metadata, and aggregate waves
cannot establish a tail. Long-scoreboard can distinguish load dependencies
from a pure tensor-throughput explanation; unresolved barriers or tail behavior
would justify one targeted follow-up metric, not an immediate full set.
NCU duration is diagnostic only; retain the uninstrumented model timing as
the deployment performance evidence.
