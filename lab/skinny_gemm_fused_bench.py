"""Correctness and cold latency for the expert GEMMs with their pointwise
neighbours fused in.

Two fusions, measured separately because they land separately:

  prologue   adarms(x) @ W.T, for the qkv and gate_up call sites, replacing
             `ada_rms_add` plus a cuBLAS GEMM with one launch.
  epilogue   x @ W.T + residual, for o_proj and down_proj, which is where the
             residual add goes once the normalization has moved into the next
             GEMM.

Correctness is checked two ways. Against torch it is a tolerance check, because
torch's reduction order is its own. Against the kernel actually being replaced
it is bit-exact: the fused output must equal running the frozen
`ada_rms_add_kernel` and then the same GEMM kernel unfused, which isolates the
prologue's arithmetic from the GEMM's.

    python lab/skinny_gemm_fused_bench.py
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import os
import statistics
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
_MODULE = (_REPO / "src" / "flash_vla" / "hardware" / "nvidia" / "h100"
           / "lingbot_vla" / "backends" / "cuda" / "skinny_gemm.py")
_REFERENCE = _REPO / "lab" / "reference_pointwise.cu"
_NVCC = "/data/apps/cuda/12.6/bin/nvcc"

_RAMP_US = 1.85
_MARGINAL_TB_S = 2.77
_ROTATION_MB = 220.0
_EPSILON = 1e-6

ROWS = 51
WIDTH = 768
# name -> (n, bias); every prologue call site has k = 768.
PROLOGUE_SHAPES = {"qkv": (2560, True), "gate_up": (5504, False)}
# name -> (k, n); the residual epilogue's call sites.
EPILOGUE_SHAPES = {"o_proj": (2048, 768), "down_proj": (2752, 768)}
# (tile_n, depth, prologue threads)
TILINGS = ((32, 6, 128), (32, 6, 256), (32, 6, 512), (32, 8, 512),
           (64, 6, 128), (64, 6, 256), (64, 6, 512), (64, 8, 512))


def load_kernel():
    spec = importlib.util.spec_from_file_location("skinny_gemm", _MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_library():
    """The frozen `ada_rms_add_kernel`, compiled from this branch's copy."""
    tag = hashlib.sha256(_REFERENCE.read_bytes()).hexdigest()[:16]
    directory = _REPO / ".cache" / "cuda_ext" / f"reference_pointwise_{tag}"
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "libreference.so"
    if not out.exists():
        command = [_NVCC, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
                   "-arch=sm_90a", "-o", str(out), str(_REFERENCE)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{result.stderr}")
    lib = ctypes.CDLL(str(out))
    lib.reference_ada_rms_add_launch.argtypes = [ctypes.c_void_p] * 7 + [
        ctypes.c_int] * 2 + [ctypes.c_float, ctypes.c_void_p]
    lib.reference_ada_rms_add_launch.restype = ctypes.c_int
    lib.reference_silu_multiply_launch.argtypes = [ctypes.c_void_p] * 2 + [
        ctypes.c_int] * 2 + [ctypes.c_void_p]
    lib.reference_silu_multiply_launch.restype = ctypes.c_int
    return lib


def ada_rms_add(lib, source, residual, weight, gamma, beta, total, target):
    code = lib.reference_ada_rms_add_launch(
        ctypes.c_void_p(source.data_ptr()), ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(gamma.data_ptr()),
        ctypes.c_void_p(beta.data_ptr()), ctypes.c_void_p(total.data_ptr()),
        ctypes.c_void_p(target.data_ptr()), source.shape[0], source.shape[1],
        ctypes.c_float(_EPSILON),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if code != 0:
        raise RuntimeError(f"reference_ada_rms_add_launch failed: {code}")


def silu_multiply(lib, source, target):
    code = lib.reference_silu_multiply_launch(
        ctypes.c_void_p(source.data_ptr()), ctypes.c_void_p(target.data_ptr()),
        source.shape[0], target.shape[1],
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if code != 0:
        raise RuntimeError(f"reference_silu_multiply_launch failed: {code}")


def floor_us(weight_mb: float) -> float:
    return _RAMP_US + weight_mb / _MARGINAL_TB_S


def graph_median_us(body, launches: int, reset, replays: int = 30) -> float:
    reset()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(launches):
            body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            body()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(replays):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / launches)
    return statistics.median(samples)


def accuracy(out, reference) -> dict[str, float]:
    a, b = out.float(), reference.float()
    diff = a - b
    return {"max_abs": diff.abs().max().item(),
            "rel_rms": (diff.norm() / b.norm()).item(),
            "cosine": F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
            "exact": bool(torch.equal(out, reference))}


def run_prologue(name: str, kernel, reference, device, verbose: bool):
    n, has_bias = PROLOGUE_SHAPES[name]
    weight_mb = n * WIDTH * 2 / 1e6
    replicas = max(8, int(_ROTATION_MB / weight_mb) + 1)

    torch.manual_seed(0)
    h = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
    zeros = torch.zeros_like(h)
    norm_weight = torch.randn(WIDTH, device=device, dtype=torch.bfloat16)
    gamma = torch.randn(WIDTH, device=device, dtype=torch.bfloat16) * 0.1
    beta = torch.randn(WIDTH, device=device, dtype=torch.bfloat16) * 0.1
    weights = [torch.randn(n, WIDTH, device=device, dtype=torch.bfloat16) * 0.05
               for _ in range(replicas)]
    bias = (torch.randn(n, device=device, dtype=torch.bfloat16) * 0.1
            if has_bias else None)
    out = torch.empty(ROWS, n, device=device, dtype=torch.bfloat16)
    out_reference = torch.empty_like(out)
    total = torch.empty_like(h)
    target = torch.empty_like(h)

    # The kernel this replaces, then the same GEMM unfused: the bit-exact bar.
    ada_rms_add(reference, h, zeros, norm_weight, gamma, beta, total, target)
    torch.cuda.synchronize()

    # The prologue on its own, against the kernel it replaces, before the GEMM
    # can hide a disagreement.
    normed = torch.empty_like(h)
    for threads in (128, 256, 512):
        kernel.adarms_linear(h, weights[0], norm_weight, gamma, beta, out, bias,
                             _EPSILON, 64, 6, threads, normed)
        torch.cuda.synchronize()
        norm_check = accuracy(normed, target)

        def prologue_only(threads=threads):
            kernel.adarms_linear(h, weights[0], norm_weight, gamma, beta, out,
                                 bias, _EPSILON, 64, 6, threads, normed)

        prologue_us = graph_median_us(prologue_only, 16, lambda: None)
        print(f"  prologue alone, {threads:3d} threads: {prologue_us:6.2f} us, "
              f"bit-exact={norm_check['exact']}")

    # torch's own chain, for a tolerance check independent of both kernels.
    values = h.float()
    values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + _EPSILON)
    values = norm_weight.float() * values
    want_norm = ((1 + gamma.float()) * values + beta.float()).to(torch.bfloat16)
    want = F.linear(want_norm.float(), weights[0].float(),
                    bias.float() if bias is not None else None)

    print(f"\n=== prologue {name}: adarms[{ROWS}, {WIDTH}] then "
          f"x [{n}, {WIDTH}]{' + bias' if has_bias else ''}, "
          f"weight {weight_mb:.2f} MB, {replicas} replicas ===")
    print(f"  GEMM floor {floor_us(weight_mb):.2f} us "
          f"(the fused launch also absorbs ada_rms_add)")

    index = [0]

    def reset():
        index[0] = 0

    def unfused_cublas():
        w = weights[index[0] % replicas]
        index[0] += 1
        ada_rms_add(reference, h, zeros, norm_weight, gamma, beta, total, target)
        if bias is None:
            torch.mm(target, w.t(), out=out_reference)
        else:
            torch.addmm(bias, target, w.t(), out=out_reference)

    unfused_us = graph_median_us(unfused_cublas, replicas, reset)
    print(f"  unfused (ada_rms_add + cuBLAS): {unfused_us:.2f} us")

    best = None
    for tile_n, depth, threads in TILINGS:
        try:
            kernel.adarms_linear(h, weights[0], norm_weight, gamma, beta, out,
                                 bias, _EPSILON, tile_n, depth, threads)
            kernel.linear(target, weights[0], out_reference, bias, tile_n=tile_n,
                          depth=depth, k_split=1, producer=1)
            torch.cuda.synchronize()
        except RuntimeError as error:
            print(f"  tile_n={tile_n} depth={depth} threads={threads}: {error}")
            continue
        versus_kernel = accuracy(out, out_reference)
        if not versus_kernel["exact"]:
            print(f"  tile_n={tile_n:3d} depth={depth} threads={threads}: NOT BIT-EXACT against "
                  f"ada_rms_add + unfused GEMM (max_abs={versus_kernel['max_abs']:.3e})")

        def fused_body():
            w = weights[index[0] % replicas]
            index[0] += 1
            kernel.adarms_linear(h, w, norm_weight, gamma, beta, out, bias,
                                 _EPSILON, tile_n, depth, threads)

        fused_us = graph_median_us(fused_body, replicas, reset)
        if verbose:
            print(f"  tile_n={tile_n:3d} depth={depth} threads={threads:3d}: {fused_us:6.2f} us "
                  f"({unfused_us / fused_us:.2f}x unfused, "
                  f"bit-exact={versus_kernel['exact']})")
        if best is None or fused_us < best[0]:
            best = (fused_us, tile_n, depth, threads, versus_kernel)

    if best is not None:
        fused_us, tile_n, depth, threads, versus_kernel = best
        kernel.adarms_linear(h, weights[0], norm_weight, gamma, beta, out, bias,
                             _EPSILON, tile_n, depth, threads)
        torch.cuda.synchronize()
        versus_torch = accuracy(out, want.to(torch.bfloat16))
        print(f"  BEST {name}: {fused_us:.2f} us vs unfused {unfused_us:.2f} us "
              f"({unfused_us / fused_us:.2f}x), tile_n={tile_n} depth={depth} "
              f"threads={threads}")
        print(f"    vs ada_rms_add + unfused GEMM: bit-exact="
              f"{versus_kernel['exact']} max_abs={versus_kernel['max_abs']:.3e}")
        print(f"    vs torch chain: max_abs={versus_torch['max_abs']:.3e} "
              f"rel_rms={versus_torch['rel_rms']:.3e} "
              f"cosine={versus_torch['cosine']:.6f}")
    del weights
    torch.cuda.empty_cache()
    return name, unfused_us, best


def run_epilogue(name: str, kernel, device, verbose: bool):
    k, n = EPILOGUE_SHAPES[name]
    weight_mb = n * k * 2 / 1e6
    replicas = max(8, int(_ROTATION_MB / weight_mb) + 1)

    torch.manual_seed(0)
    x = torch.randn(ROWS, k, device=device, dtype=torch.bfloat16) * 0.5
    residual = torch.randn(ROWS, n, device=device, dtype=torch.bfloat16)
    weights = [torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05
               for _ in range(replicas)]
    out = torch.empty(ROWS, n, device=device, dtype=torch.bfloat16)
    plain = torch.empty_like(out)

    print(f"\n=== epilogue {name}: [{ROWS}, {k}] x [{n}, {k}] + residual, "
          f"weight {weight_mb:.2f} MB, {replicas} replicas ===")

    index = [0]

    def reset():
        index[0] = 0

    best = None
    for tile_n, depth in ((32, 6), (64, 6), (64, 8)):
        for k_split in (4, 8):
            try:
                kernel.linear(x, weights[0], plain, None, tile_n=tile_n,
                              depth=depth, k_split=k_split, producer=3)
                kernel.linear_residual(x, weights[0], residual, out, None,
                                       tile_n=tile_n, depth=depth, k_split=k_split)
                torch.cuda.synchronize()
            except RuntimeError as error:
                print(f"  tile_n={tile_n} depth={depth} split={k_split}: {error}")
                continue
            # The unfused chain rounds the GEMM to bf16 then adds in f32.
            want = (plain.float() + residual.float()).to(torch.bfloat16)
            versus = accuracy(out, want)
            if not versus["exact"]:
                print(f"  tile_n={tile_n} depth={depth} split={k_split}: "
                      f"NOT BIT-EXACT (max_abs={versus['max_abs']:.3e})")

            def plain_body():
                w = weights[index[0] % replicas]
                index[0] += 1
                kernel.linear(x, w, plain, None, tile_n=tile_n, depth=depth,
                              k_split=k_split, producer=3)

            def fused_body():
                w = weights[index[0] % replicas]
                index[0] += 1
                kernel.linear_residual(x, w, residual, out, None, tile_n=tile_n,
                                       depth=depth, k_split=k_split)

            plain_us = graph_median_us(plain_body, replicas, reset)
            fused_us = graph_median_us(fused_body, replicas, reset)
            if verbose:
                print(f"  tile_n={tile_n:3d} depth={depth} split={k_split}: "
                      f"plain {plain_us:5.2f} us, +residual {fused_us:5.2f} us "
                      f"(+{fused_us - plain_us:.2f}), bit-exact={versus['exact']}")
            if best is None or fused_us < best[0]:
                best = (fused_us, plain_us, tile_n, depth, k_split, versus)

    if best is not None:
        fused_us, plain_us, tile_n, depth, k_split, versus = best
        print(f"  BEST {name}: +residual {fused_us:.2f} us vs plain {plain_us:.2f} us "
              f"(+{fused_us - plain_us:.2f} us), tile_n={tile_n} depth={depth} "
              f"split={k_split}")
        print(f"    vs GEMM-then-add: bit-exact={versus['exact']} "
              f"max_abs={versus['max_abs']:.3e}")
    del weights
    torch.cuda.empty_cache()
    return name, best


# (paired tile_n, depth): a CTA owns tile_n // 2 gate columns and their
# tile_n // 2 up partners.
# (paired tile_n, depth, k_split)
SILU_TILINGS = ((64, 6, 1), (64, 8, 1), (64, 6, 2), (64, 8, 2), (64, 12, 2),
                (64, 6, 4), (64, 8, 4), (128, 6, 2), (128, 6, 4), (32, 8, 2))


def run_silu(kernel, reference, device, verbose: bool):
    n, k = 5504, WIDTH
    half_n = n // 2
    weight_mb = n * k * 2 / 1e6
    replicas = max(8, int(_ROTATION_MB / weight_mb) + 1)

    torch.manual_seed(0)
    x = torch.randn(ROWS, k, device=device, dtype=torch.bfloat16) * 0.5
    weights = [torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05
               for _ in range(replicas)]
    packed = torch.empty(ROWS, n, device=device, dtype=torch.bfloat16)
    out = torch.empty(ROWS, half_n, device=device, dtype=torch.bfloat16)
    out_reference = torch.empty_like(out)

    gate, up = (x.float() @ weights[0].float().t()).split(half_n, dim=-1)
    want = (F.silu(gate) * up).to(torch.bfloat16)

    print(f"\n=== silu epilogue gate_up: [{ROWS}, {k}] x [{n}, {k}] then "
          f"silu(gate)*up -> [{ROWS}, {half_n}], weight {weight_mb:.2f} MB, "
          f"{replicas} replicas ===")
    print(f"  GEMM floor {floor_us(weight_mb):.2f} us "
          "(the fused launch also absorbs silu_multiply)")

    index = [0]

    def reset():
        index[0] = 0

    def unfused_cublas():
        w = weights[index[0] % replicas]
        index[0] += 1
        torch.mm(x, w.t(), out=packed)
        silu_multiply(reference, packed, out_reference)

    unfused_us = graph_median_us(unfused_cublas, replicas, reset)
    print(f"  unfused (cuBLAS + silu_multiply): {unfused_us:.2f} us")

    best = None
    for tile_n, depth, k_split in SILU_TILINGS:
        try:
            kernel.linear(x, weights[0], packed, None, tile_n=64, depth=6,
                          k_split=1, producer=1)
            silu_multiply(reference, packed, out_reference)
            kernel.silu_linear(x, weights[0], out, None, tile_n, depth, k_split)
            torch.cuda.synchronize()
        except RuntimeError as error:
            print(f"  tile_n={tile_n} depth={depth} split={k_split}: {error}")
            continue
        versus_kernel = accuracy(out, out_reference)
        ctas = (half_n // (tile_n // 2)) * k_split
        if not versus_kernel["exact"]:
            print(f"  tile_n={tile_n:3d} depth={depth} split={k_split}: NOT "
                  f"BIT-EXACT (max_abs={versus_kernel['max_abs']:.3e})")

        def fused_body():
            w = weights[index[0] % replicas]
            index[0] += 1
            kernel.silu_linear(x, w, out, None, tile_n, depth, k_split)

        fused_us = graph_median_us(fused_body, replicas, reset)
        if verbose:
            print(f"  tile_n={tile_n:3d} depth={depth:2d} split={k_split} "
                  f"ctas={ctas:3d}: {fused_us:6.2f} us "
                  f"({unfused_us / fused_us:.2f}x unfused, "
                  f"bit-exact={versus_kernel['exact']})")
        if best is None or fused_us < best[0]:
            best = (fused_us, tile_n, depth, k_split, ctas, versus_kernel)

    if best is not None:
        fused_us, tile_n, depth, k_split, ctas, versus_kernel = best
        kernel.silu_linear(x, weights[0], out, None, tile_n, depth, k_split)
        torch.cuda.synchronize()
        versus_torch = accuracy(out, want)
        print(f"  BEST silu: {fused_us:.2f} us vs unfused {unfused_us:.2f} us "
              f"({unfused_us / fused_us:.2f}x), tile_n={tile_n} depth={depth} "
              f"split={k_split} ctas={ctas}")
        print(f"    vs GEMM + silu_multiply: bit-exact={versus_kernel['exact']} "
              f"max_abs={versus_kernel['max_abs']:.3e}")
        print(f"    vs torch chain: max_abs={versus_torch['max_abs']:.3e} "
              f"rel_rms={versus_torch['rel_rms']:.3e} "
              f"cosine={versus_torch['cosine']:.6f}")
    del weights
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sections", default="prologue,epilogue")
    args = parser.parse_args()

    device = torch.device("cuda")
    kernel = load_kernel()
    kernel.build(verbose=True)
    reference = reference_library()
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")

    sections = args.sections.split(",")
    if "prologue" in sections:
        for name in PROLOGUE_SHAPES:
            run_prologue(name, kernel, reference, device, args.verbose)
    if "epilogue" in sections:
        for name in EPILOGUE_SHAPES:
            run_epilogue(name, kernel, device, args.verbose)
    if "silu" in sections:
        run_silu(kernel, reference, device, args.verbose)


if __name__ == "__main__":
    main()
