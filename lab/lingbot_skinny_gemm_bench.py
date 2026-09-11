"""Correctness and cold-read latency for the action expert's skinny GEMMs.

The four expert projections are weight streams, so the only honest timing is a
cold one: each launch must read a weight matrix that L2 has already evicted.
The harness rotates over enough weight replicas to cycle the 50 MB L2 several
times per graph replay, captures one graph per candidate and divides the replay
by the number of launches inside it, which is also how the cuBLAS baseline is
measured. A hot loop over a single resident weight would report a number the
model can never see.

    python lab/skinny_gemm_bench.py --shapes qkv,o_proj,gate_up,down_proj
    LINGBOT_SKINNY_DEFINES=-DSKINNY_PROBE_NO_EPILOGUE \
        python lab/skinny_gemm_bench.py --probe --shapes qkv
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
_MODULE = (_REPO / "src" / "flash_vla" / "hardware" / "nvidia" / "h100"
           / "lingbot_vla" / "backends" / "cuda" / "skinny_gemm.py")

# The measured cold-read model for this machine: a launch costs 1.85 us before
# a byte moves and the marginal rate is 2.77 TB/s [unit-launch ld.bw.dev.dram].
_RAMP_US = 1.85
_MARGINAL_TB_S = 2.77
# Four L2 passes between two launches that share a weight replica, so no
# candidate is ever timed against a resident matrix.
_ROTATION_MB = 220.0

# rows, k, n, bias -- the per-layer per-denoise-step call sites of the expert.
SHAPES = {
    "qkv": (51, 768, 2560, True),
    "o_proj": (51, 2048, 768, False),
    "gate_up": (51, 768, 5504, False),
    "down_proj": (51, 2752, 768, False),
}

# The compiled (tile_n, depth) instantiations of the kernel.
TILINGS = ((32, 6), (32, 8), (32, 12), (64, 4), (64, 6), (64, 8), (64, 12),
           (128, 4), (128, 6), (128, 8), (256, 3), (256, 4))
SPLITS = (1, 2, 3, 4, 5, 6, 8, 11, 16, 22)
PRODUCERS = {0: "cp.async", 1: "tma", 2: "tma+mcast", 3: "tma+dsmem"}


def load_kernel():
    spec = importlib.util.spec_from_file_location("skinny_gemm", _MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def floor_us(weight_mb: float) -> float:
    return _RAMP_US + weight_mb / _MARGINAL_TB_S


def graph_median_us(body, launches: int, reset, replays: int = 30) -> float:
    """Median per-launch microseconds of `launches` calls captured in one graph.

    The warm-up covers every rotation slot before capture, so a producer that
    memoizes a per-tensor descriptor on the host finds a hit inside capture.
    """
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


def accuracy(out: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    a = out.float()
    b = reference.float()
    diff = a - b
    return {
        "max_abs": diff.abs().max().item(),
        "rel_rms": (diff.norm() / b.norm()).item(),
        "cosine": F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
    }


def candidates(n: int, k: int, kernel, producers) -> list[tuple[int, int, int, int]]:
    """(tile_n, depth, k_split, producer) tilings that land in one or two waves."""
    k_tiles = k // 64
    out = []
    for tile_n, depth in TILINGS:
        tiles = kernel.n_tiles(n, tile_n)
        for split in SPLITS:
            if split > k_tiles:
                continue
            ctas = tiles * split
            if ctas < 24 or ctas > 280:
                continue
            for producer in producers:
                # The cluster producer shares one activation tile between a
                # pair of N tiles, so it has no split-K path and needs an even
                # tile count.
                if producer == 2 and (split != 1 or tiles % 2 != 0):
                    continue
                # The cluster reduction makes the splits a cluster, so the
                # split count is the cluster size and only 2/4/8 are compiled.
                if producer == 3 and split not in (2, 4, 8):
                    continue
                out.append((tile_n, depth, split, producer))
    return out


def run_shape(name: str, kernel, device, args) -> tuple:
    rows, k, n, has_bias = SHAPES[name]
    weight_mb = n * k * 2 / 1e6
    replicas = max(8, int(_ROTATION_MB / weight_mb) + 1)

    torch.manual_seed(0)
    x = torch.randn(rows, k, device=device, dtype=torch.bfloat16) * 0.5
    weights = [torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05
               for _ in range(replicas)]
    bias = (torch.randn(n, device=device, dtype=torch.bfloat16) * 0.1
            if has_bias else None)
    out = torch.empty(rows, n, device=device, dtype=torch.bfloat16)

    exact = x.float() @ weights[0].float().t()
    if bias is not None:
        exact = exact + bias.float()
    torch_out = F.linear(x, weights[0], bias)

    print(f"\n=== {name}: [{rows}, {k}] x [{n}, {k}]"
          f"{' + bias' if has_bias else ''}, weight {weight_mb:.2f} MB, "
          f"{replicas} replicas ===")
    print(f"floor (1.85 + MB/2.77) = {floor_us(weight_mb):.2f} us")
    acc = accuracy(torch_out, exact)
    print(f"cuBLAS vs fp32: max_abs={acc['max_abs']:.3e} "
          f"rel_rms={acc['rel_rms']:.3e} cosine={acc['cosine']:.6f}")

    index = [0]

    def reset():
        index[0] = 0

    def cublas_body():
        w = weights[index[0] % replicas]
        index[0] += 1
        if bias is None:
            torch.mm(x, w.t(), out=out)
        else:
            torch.addmm(bias, x, w.t(), out=out)

    cublas_us = graph_median_us(cublas_body, replicas, reset)
    print(f"cuBLAS: {cublas_us:.2f} us  ({cublas_us / floor_us(weight_mb):.2f}x floor)")

    results = []
    for tile_n, depth, split, producer in candidates(n, k, kernel, args.producers):
        workspace = counters = None
        if split > 1:
            workspace, counters = kernel.make_workspace(rows, n, tile_n, split, device)
        tag = (f"  {PRODUCERS[producer]:8s} tile_n={tile_n:3d} depth={depth:2d} "
               f"split={split:2d} ctas={kernel.n_tiles(n, tile_n) * split:3d}")

        out.zero_()
        try:
            kernel.linear(x, weights[0], out, bias, tile_n=tile_n, depth=depth,
                          k_split=split, producer=producer, workspace=workspace,
                          counters=counters)
            torch.cuda.synchronize()
        except RuntimeError as error:
            print(f"{tag}: {error}")
            continue
        acc = accuracy(out, exact)
        if not args.probe:
            if acc["cosine"] < 0.999:
                print(f"{tag}: WRONG cosine={acc['cosine']:.6f} "
                      f"max_abs={acc['max_abs']:.3e}")
                continue
            if split > 1 and int(counters.abs().max().item()) != 0:
                print(f"{tag}: arrival counters not restored to zero")
                continue

        def kernel_body():
            w = weights[index[0] % replicas]
            index[0] += 1
            kernel.linear(x, w, out, bias, tile_n=tile_n, depth=depth,
                          k_split=split, producer=producer, workspace=workspace,
                          counters=counters)

        us = graph_median_us(kernel_body, replicas, reset)
        results.append((us, tile_n, depth, split, producer, acc))
        if args.verbose:
            print(f"{tag}: {us:6.2f} us")

    results.sort()
    print("  top 5:")
    for us, tile_n, depth, split, producer, acc in results[:5]:
        print(f"    {PRODUCERS[producer]:8s} tile_n={tile_n:3d} depth={depth:2d} "
              f"split={split:2d}  {us:6.2f} us  ({cublas_us / us:.2f}x cuBLAS, "
              f"{us / floor_us(weight_mb):.2f}x floor)")

    best = results[0] if results else None
    if best is not None and not args.probe:
        us, tile_n, depth, split, producer, acc = best
        print(f"    vs fp32: max_abs={acc['max_abs']:.3e} "
              f"rel_rms={acc['rel_rms']:.3e} cosine={acc['cosine']:.6f}")
        out.zero_()
        ws = ctr = None
        if split > 1:
            ws, ctr = kernel.make_workspace(rows, n, tile_n, split, device)
        kernel.linear(x, weights[0], out, bias, tile_n=tile_n, depth=depth,
                      k_split=split, producer=producer, workspace=ws, counters=ctr)
        torch.cuda.synchronize()
        versus = accuracy(out, torch_out)
        print(f"    vs cuBLAS: max_abs={versus['max_abs']:.3e} "
              f"rel_rms={versus['rel_rms']:.3e} cosine={versus['cosine']:.6f}")

    del weights
    torch.cuda.empty_cache()
    return name, weight_mb, cublas_us, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--producers", default="0,1")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="skip correctness (for SKINNY_PROBE_* builds)")
    args = parser.parse_args()
    args.producers = [int(p) for p in args.producers.split(",")]

    device = torch.device("cuda")
    kernel = load_kernel()
    kernel.build(verbose=True)
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")

    summary = [run_shape(name.strip(), kernel, device, args)
               for name in args.shapes.split(",")]
    print("\n=== summary (median us per cold launch) ===")
    print(f"{'shape':<11}{'MB':>6}{'floor':>8}{'cuBLAS':>9}{'kernel':>9}"
          f"{'speedup':>9}  config")
    for name, weight_mb, cublas_us, best in summary:
        if best is None:
            print(f"{name:<11}{weight_mb:6.2f}{floor_us(weight_mb):8.2f}"
                  f"{cublas_us:9.2f}{'--':>9}{'--':>9}")
            continue
        us, tile_n, depth, split, producer, _ = best
        print(f"{name:<11}{weight_mb:6.2f}{floor_us(weight_mb):8.2f}"
              f"{cublas_us:9.2f}{us:9.2f}{cublas_us / us:8.2f}x  "
              f"{PRODUCERS[producer]} tile_n={tile_n} depth={depth} split={split}")


if __name__ == "__main__":
    main()
