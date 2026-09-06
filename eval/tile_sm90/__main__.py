"""Device parity for the SM90 tile primitives (src/flash_vla/hardware/nvidia/cuda/tile/sm90).

Every case in primitives.cu runs one CTA through g2s -> smem -> [s2r] ->
gemm -> r2s -> s2g using library calls only and is compared against a
float32 torch reference of the same inputs.  A failing case names the
primitive path that is wrong; the tolerances are gates, not reports.

Run on a GPU node::

    sbatch sbatch/pi05_cuda.sh -m eval.tile_sm90

Exit status is non-zero when any case fails.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from eval.metrics import error_metrics

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_SRC = _HERE / "primitives.cu"
_TILE_ROOT = _REPO / "src" / "flash_vla" / "hardware" / "nvidia" / "cuda"
_CUTLASS = Path(os.environ.get("CUTLASS_DIR", _REPO / "third_party" / "cutlass"))


def _build_dir() -> Path:
    hasher = hashlib.sha256(_SRC.read_bytes())
    for header in sorted((_TILE_ROOT / "tile" / "sm90").glob("*.cuh")):
        hasher.update(header.read_bytes())
    tag = hasher.hexdigest()[:16]
    d = _REPO / ".cache" / "cuda_ext" / f"tile_sm90_primitives_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(verbose: bool = False) -> Path:
    """Compile primitives.cu into a shared object keyed by source + header hash."""
    out = _build_dir() / "libtile_sm90_primitives.so"
    if out.exists():
        return out
    cuda_home = os.environ.get("CUDA_HOME", "/data/apps/cuda/13.1")
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [
        nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
        "-arch=sm_90a", "--expt-relaxed-constexpr",
        f"-I{_CUTLASS}/include", f"-I{_TILE_ROOT}",
        "-o", str(out), str(_SRC),
        f"-L{cuda_home}/lib64/stubs", "-lcuda",
    ]
    if verbose:
        print("[tile_sm90 build]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    # ptxas C7518 (serialized wgmma) is a warning; keep the full log visible.
    if r.stdout.strip() or r.stderr.strip():
        print(r.stdout, r.stderr, flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed with {r.returncode}")
    return out


@dataclass(frozen=True)
class Case:
    name: str
    a_dtype: torch.dtype
    b_dtype: torch.dtype
    m: int
    n: int
    k: int
    b_layout: str        # "nk": B stored (N, K), C = A @ B^T; "kn": B stored (K, N), C = A @ B
    out_dtype: torch.dtype
    rel_tol: float       # max |err| / max |ref|


_BF = torch.bfloat16
_E4 = torch.float8_e4m3fn
_E5 = torch.float8_e5m2

# f32 accumulation of exactly-representable inputs differs from torch only by
# summation order: 1e-4 relative is generous for bf16 wgmma / mma.sync, and
# the fp8 mma.sync path is exact (ptxas lowers it to f16 HMMA on sm_90a).
_F32_TOL = 1e-4
# Hopper's fp8 wgmma accumulates with reduced precision inside the tensor
# core (~2^-13 relative per K block; the reason DeepGEMM promotes partial
# sums on CUDA cores).  Measured here at K=64: 2.5e-4 relative, cosine
# 1.0000000.  Gate at 1e-3 so a real wiring bug (which lands at O(1)) still
# fails while the hardware's own rounding passes.
_FP8_WGMMA_TOL = 1e-3

CASES = [
    Case("wgmma_ss_bf16_kk", _BF, _BF, 64, 64, 64, "nk", _BF, 8e-3),
    Case("wgmma_ss_bf16_kmn", _BF, _BF, 64, 32, 128, "kn", torch.float32, _F32_TOL),
    Case("wgmma_rs_bf16", _BF, _BF, 64, 128, 64, "nk", torch.float32, _F32_TOL),
    Case("wgmma_ss_e4m3", _E4, _E4, 64, 64, 64, "nk", torch.float32, _FP8_WGMMA_TOL),
    Case("wgmma_ss_e4m3_e5m2", _E4, _E5, 64, 64, 64, "nk", torch.float32, _FP8_WGMMA_TOL),
    Case("mma_sync_bf16", _BF, _BF, 64, 32, 32, "nk", _BF, 8e-3),
    Case("mma_sync_e4m3", _E4, _E4, 64, 32, 64, "nk", torch.float32, _F32_TOL),
    Case("wgmma_ss_bf16_wg2", _BF, _BF, 128, 64, 64, "nk", torch.float32, _F32_TOL),
]


def _operand(shape: tuple[int, int], dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)
    # fp8 inputs are exactly representable after the cast, so the reference
    # differs from the device only by f32 summation order.
    return x.to(dtype)


def run_case(lib: ctypes.CDLL, case: Case, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = _operand((case.m, case.k), case.a_dtype, gen)
    b_shape = (case.n, case.k) if case.b_layout == "nk" else (case.k, case.n)
    b = _operand(b_shape, case.b_dtype, gen)
    c = torch.full((case.m, case.n), float("nan"), device="cuda", dtype=case.out_dtype)
    torch.cuda.synchronize()
    rc = lib.tile_sm90_run(case.name.encode(), ctypes.c_void_p(a.data_ptr()),
                           ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(c.data_ptr()))
    torch.cuda.synchronize()
    a32, b32 = a.float(), b.float()
    ref = a32 @ (b32.t() if case.b_layout == "nk" else b32)
    got = c.float()
    shared = error_metrics(ref, got)
    scale = ref.abs().max().item()
    max_abs = shared["max_abs"]
    rel = max_abs / scale if scale > 0 else max_abs
    cos = shared["cosine_similarity"]
    ok = rc == 0 and torch.isfinite(got).all().item() and rel <= case.rel_tol
    return {
        "case": case.name, "rc": rc, "max_abs": max_abs, "rel": rel,
        "cosine": cos, "tol": case.rel_tol, "ok": bool(ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    lib = ctypes.CDLL(str(build(verbose=True)))
    lib.tile_sm90_run.restype = ctypes.c_int
    lib.tile_sm90_run.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    wanted = {s for s in args.only.split(",") if s}
    results = [run_case(lib, c, args.seed) for c in CASES if not wanted or c.name in wanted]
    for r in results:
        if args.json:
            print(json.dumps(r))
        else:
            print(f"{'PASS' if r['ok'] else 'FAIL'} {r['case']:<22} rc={r['rc']:<5} "
                  f"max_abs={r['max_abs']:.3e} rel={r['rel']:.3e} (tol {r['tol']:.0e}) "
                  f"cos={r['cosine']:.7f}")
    failed = [r["case"] for r in results if not r["ok"]]
    print(f"[tile_sm90] {len(results) - len(failed)}/{len(results)} cases pass"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
