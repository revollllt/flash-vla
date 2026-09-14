"""Prepare one actual gate operand pair, then profile the deployed cached GEMM."""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def prepare(args):
    """Save deployed layer-0 normalized input and folded gate weight outside NCU."""
    from flash_vla.inference import build, resolve

    engine = build(
        resolve("rtx5090/pi05"), "shipped", seed=42,
        converted_checkpoint=args.checkpoint, checkpoint_id=args.checkpoint_id,
        checkpoint_digest=args.checkpoint_id)
    inputs = engine.sample_inputs(42)
    saved = False

    def inspect(name, function):
        if name != "llm_backbone_norm_gated_ffn":
            return function

        def record(x, gate_w, up_w, out, x_norm):
            nonlocal saved
            result = function(x, gate_w, up_w, out, x_norm)
            if not saved:
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                save_file(
                    {"normalized": x_norm[:x.shape[0]].detach().cpu().contiguous(),
                     "gate_weight": gate_w.detach().cpu().contiguous()},
                    str(args.snapshot),
                    metadata={"checkpoint": args.checkpoint_id, "seed": "42",
                              "layer": "0", "identity": json.dumps(engine.identity.as_dict())})
                saved = True
            return result

        return record

    engine.stage(**inputs)
    with engine.instrument(inspect):
        for step in engine.program:
            if step.kind == "host":
                engine.host(step.name, **inputs)
            elif step.name == "llm_backbone":
                engine.run_eager(step.name)
                break
            else:
                engine.replay(step.name)
    torch.cuda.synchronize()
    print(json.dumps({"snapshot": str(args.snapshot), "saved": saved}), flush=True)


def profile(args):
    """Load only the saved actual operand values and cached production native library."""
    from flash_vla.hardware.nvidia.rtx5090.pi05.backends import cutlass_backbone
    from flash_vla.runtime.runner import Scratch

    with safe_open(str(args.snapshot), framework="pt", device="cpu") as stored:
        a = stored.get_tensor("normalized").to("cuda")
        b = stored.get_tensor("gate_weight").to("cuda")
        metadata = stored.metadata()
    output = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    # CDLL loads the existing deployment artifact; this driver never calls nvcc.
    library = ctypes.CDLL(str(args.library))
    library.backbone_gemm_workspace.argtypes = [ctypes.c_int32] * 3
    library.backbone_gemm_workspace.restype = ctypes.c_int64
    library.backbone_gemm_plan.argtypes = (
        [ctypes.c_int32] * 3 + [ctypes.c_float] + [ctypes.c_void_p] * 5
        + [ctypes.POINTER(ctypes.c_void_p)])
    library.backbone_gemm_plan.restype = ctypes.c_int32
    library.backbone_gemm_run.argtypes = [ctypes.c_void_p] * 2
    library.backbone_gemm_run.restype = ctypes.c_int32
    library.backbone_gemm_destroy.argtypes = [ctypes.c_void_p]
    library.backbone_gemm_destroy.restype = None
    stream = torch.cuda.current_stream().cuda_stream
    scratch = Scratch(a.device)
    plan = cutlass_backbone._Plan(library, scratch, a, b, output, beta=0.0, stream=stream)
    for _ in range(50):
        cutlass_backbone._check(
            library.backbone_gemm_run(plan.handle, stream), "gate warmup")
    torch.cuda.synchronize()
    print(json.dumps({
        "mode": "single eager launch of deployed Stream-K config 0",
        "library": str(args.library), "snapshot": str(args.snapshot),
        "metadata": metadata, "a_shape": list(a.shape), "a_stride": list(a.stride()),
        "b_shape": list(b.shape), "b_stride": list(b.stride()),
        "dtype": str(a.dtype), "warmup_calls": 50,
        "nvtx_range": "pi05_backbone_gate"}), flush=True)
    with torch.cuda.nvtx.range("pi05_backbone_gate"):
        cutlass_backbone._check(
            library.backbone_gemm_run(plan.handle, stream), "gate measured launch")
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "profile"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.checkpoint is None or args.checkpoint_id is None:
            parser.error("prepare requires --checkpoint and --checkpoint-id")
        prepare(args)
    else:
        if args.library is None:
            parser.error("profile requires the deployed cached --library")
        profile(args)


if __name__ == "__main__":
    main()
