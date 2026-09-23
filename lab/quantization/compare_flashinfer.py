"""Our quant ops against FlashInfer 0.7.0's, head to head.

1. Standalone quantize at every distinct GEMM activation shape of the survey:
   ``ops.quantize`` against ``flashinfer.mxfp8_quantize`` / ``fp4_quantize``.
   The outputs are byte-identical (check_quant_ops.py), so only speed differs.
2. RMSNorm -> NVFP4, the one fused producer FlashInfer also has
   (``flashinfer.cute_dsl.rmsnorm_fp4quant``, CuTe DSL). Its ``global_scale``
   is the encode scale, as for ``fp4_quantize`` (its docstring says the output
   is divided by it). It rounds RMSNorm(x) * w once where HF Qwen3 rounds
   twice, so a few percent of bytes differ; the agreement is reported.

Both sides launch with programmatic dependent launch (PDL) by default, so each
pair is timed in three cells:
- kernel: device µs per call from CUPTI (torch.profiler), PDL off on both
  sides. With PDL on, a kernel that starts early and waits for its
  predecessor counts the wait as its own time, so only PDL-off kernel times
  measure the kernels themselves;
- node, PDL on / PDL off: 64 calls per CUDA graph, all four graphs replayed
  in turn for 21 rounds, median µs per call; what a captured chain pays.

Run from the repository root:

    PYTHONPATH=src CUDA_HOME=$HOME/cuda-13.1 flock /tmp/flash-vla-rtx5090.lock \\
        ~/quant-survey/venv-flashinfer/bin/python lab/quantization/compare_flashinfer.py \\
        --out results/quant-fused-rtx5090
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Callable

import flashinfer
from flashinfer import fp4_quantize, mxfp8_quantize
from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import Call, interleaved_times_us
from flash_vla.hardware.nvidia.quant_ops import ops
from flash_vla.quantization import formats
from vla_shapes import SHAPES

# GR00T DiT, Pi0.5 expert, GR00T LLM, GR00T vision, Pi0.5 LLM.
RMS_NORM_SHAPES = [(41, 1536), (50, 1024), (156, 2048), (512, 1024), (968, 2048)]
GRAPH_CALLS = 64
PROFILED_CALLS = 50

Row = dict[str, str | int | float]
CallWithPdl = Callable[[bool], object]   # call(pdl)


def kernel_time_us(call: Call) -> float:
    """Device µs per call: every kernel the call launches, from CUPTI."""
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        for _ in range(PROFILED_CALLS):
            call()
        torch.cuda.synchronize()
    return sum(event.device_time for event in profiler.events()
               if event.device_type == torch.autograd.DeviceType.CUDA) / PROFILED_CALLS


def compare(ours: CallWithPdl, theirs: CallWithPdl) -> dict[str, float]:
    """The three cells for both sides, and ours / FlashInfer for each."""
    variants = {f"{side}_{mode}": [lambda call=call, pdl=pdl: call(pdl)] * GRAPH_CALLS
                for side, call in (("ours", ours), ("flashinfer", theirs))
                for mode, pdl in (("pdl", True), ("no_pdl", False))}
    times = interleaved_times_us(variants, dict.fromkeys(variants, GRAPH_CALLS))
    node_us = {name: statistics.median(values) for name, values in times.items()}
    kernel_us = {"ours": kernel_time_us(lambda: ours(False)),
                 "flashinfer": kernel_time_us(lambda: theirs(False))}
    return dict(ours_kernel_us=kernel_us["ours"], flashinfer_kernel_us=kernel_us["flashinfer"],
                kernel_ratio=kernel_us["ours"] / kernel_us["flashinfer"],
                ours_node_pdl_us=node_us["ours_pdl"],
                flashinfer_node_pdl_us=node_us["flashinfer_pdl"],
                node_pdl_ratio=node_us["ours_pdl"] / node_us["flashinfer_pdl"],
                ours_node_no_pdl_us=node_us["ours_no_pdl"],
                flashinfer_node_no_pdl_us=node_us["flashinfer_no_pdl"],
                node_no_pdl_ratio=node_us["ours_no_pdl"] / node_us["flashinfer_no_pdl"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    rows: list[Row] = []
    for rows_m, cols_k in sorted({(site.m, site.k) for site in SHAPES}):
        x = torch.randn(rows_m, cols_k, device="cuda", dtype=torch.bfloat16)
        encode_scale = ((448 * 6) / x.float().abs().max()).reshape(1).float()
        for fmt in ("mxfp8", "nvfp4"):
            out = ops.empty(rows_m, cols_k, fmt, global_scale=encode_scale)
            theirs = ((lambda pdl: mxfp8_quantize(x, True, enable_pdl=pdl)) if fmt == "mxfp8"
                      else (lambda pdl: fp4_quantize(x, encode_scale, 16, False, True,
                                                     enable_pdl=pdl)))
            row = dict(op="quantize", fmt=fmt, m=rows_m, k=cols_k,
                       **compare(lambda pdl: ops.quantize(x, out, pdl=pdl), theirs))
            rows.append(row)
            print(json.dumps({key: round(value, 3) if isinstance(value, float) else value
                              for key, value in row.items()}), flush=True)
    for rows_m, cols_k in RMS_NORM_SHAPES:
        x = torch.randn(rows_m, cols_k, device="cuda", dtype=torch.bfloat16) * 2
        weight = (torch.randn(cols_k, device="cuda") * 0.2 + 1).to(torch.bfloat16)
        bf16 = ops.rms_norm(x, ops.empty(rows_m, cols_k, "bf16"), weight=weight)
        encode_scale = ((448 * 6) / bf16.values.float().abs().max()).reshape(1).float()
        out = ops.empty(rows_m, cols_k, "nvfp4", global_scale=encode_scale)
        theirs = lambda pdl: rmsnorm_fp4quant(x, weight, global_scale=encode_scale, eps=1e-6,
                                              is_sf_swizzled_layout=True, enable_pdl=pdl)
        their_values, their_scales = theirs(True)
        ops.rms_norm(x, out, weight=weight)
        blocks = cols_k // formats.NVFP4.block
        real_scales_equal = (formats.unswizzle(out.scale, rows_m, blocks)
                             == formats.unswizzle(their_scales.view(torch.uint8), rows_m, blocks))
        row = dict(op="rms_norm_nvfp4", fmt="nvfp4", m=rows_m, k=cols_k,
                   value_bytes_equal=float((their_values.view(torch.uint8) == out.values)
                                           .float().mean()),
                   scale_bytes_equal=float(real_scales_equal.float().mean()),
                   **compare(lambda pdl: ops.rms_norm(x, out, weight=weight, pdl=pdl), theirs))
        rows.append(row)
        print(json.dumps({key: round(value, 3) if isinstance(value, float) else value
                          for key, value in row.items()}), flush=True)

    lines = [f"# quant_ops vs FlashInfer {flashinfer.__version__} ({torch.cuda.get_device_name()})",
             "", "µs per call; each ratio is ours / FlashInfer (below 1: ours faster). Kernel "
             "times are PDL off on both sides; node times are per CUDA-graph node.", "",
             "| op | fmt | M×K | kernel ours | kernel FI | ratio | node PDL ours | node PDL FI "
             "| ratio | node no-PDL ours | node no-PDL FI | ratio |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['op']} | {row['fmt']} | {row['m']}×{row['k']} "
              f"| {row['ours_kernel_us']:.2f} | {row['flashinfer_kernel_us']:.2f} "
              f"| {row['kernel_ratio']:.2f} | {row['ours_node_pdl_us']:.2f} "
              f"| {row['flashinfer_node_pdl_us']:.2f} | {row['node_pdl_ratio']:.2f} "
              f"| {row['ours_node_no_pdl_us']:.2f} | {row['flashinfer_node_no_pdl_us']:.2f} "
              f"| {row['node_no_pdl_ratio']:.2f} |" for row in rows]
    lines += ["", "| op | fmt | kernel | node PDL | node no-PDL | our PDL gain per node |",
              "|---|---|---:|---:|---:|---:|"]
    for op, fmt in (("quantize", "mxfp8"), ("quantize", "nvfp4"), ("rms_norm_nvfp4", "nvfp4")):
        selected = [row for row in rows if row["op"] == op and row["fmt"] == fmt]
        geometric = {cell: statistics.geometric_mean(row[cell] for row in selected)
                     for cell in ("kernel_ratio", "node_pdl_ratio", "node_no_pdl_ratio")}
        pdl_gain = statistics.geometric_mean(row["ours_node_no_pdl_us"] / row["ours_node_pdl_us"]
                                             for row in selected)
        lines.append(f"| {op} | {fmt} | {geometric['kernel_ratio']:.2f} "
                     f"| {geometric['node_pdl_ratio']:.2f} | {geometric['node_no_pdl_ratio']:.2f} "
                     f"| {pdl_gain:.2f}× |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "compare_flashinfer.json").write_text(json.dumps(rows, indent=1) + "\n")
    (args.out / "compare_flashinfer.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
