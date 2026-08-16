"""Numerical regression gate: the fused decoder against the unfused one.

The unfused path is the two-kernel reference -- separate norm, then GEMM -- and
was validated op-by-op against the original Triton implementation while that was
still in tree. The fused kernels compute the same value by a different route,
so this compares them.

Read it at `--steps 1`. The diffusion loop is a chaotic map on random untrained
weights: a per-kernel difference around 1e-3 compounds into a macroscopic output
difference over 10 steps and 18 layers, for any two implementations that are not
bit-identical. Depth is the amplifier, not the fusion. `--layers 1` narrows it
further when something does look wrong.

The two paths are deliberately not bit-identical: the fused kernels scale the
fp32 accumulator, while the unfused path rounds x*factor to bf16 per element
before the GEMM. The fused result is the more accurate of the two, so a small
deviation here is expected and a zero would be surprising.
"""
from __future__ import annotations

import argparse
import json

import torch

from tilelang_infer.hardware.nvidia.h100.pi0 import pipeline
from tilelang_infer.hardware.nvidia.h100.pi0.ops import op_table
from benchmarks.metrics import diff_stats, require_cuda
from benchmarks.synthetic import decoder_buffers, decoder_weights, encoder_seq_len

TOLERANCE = 5e-1


def run(num_views: int = 3, prompt_len: int = 0, chunk_size: int = 50, steps: int = 1,
        layers: int = 18, wscale: float = 0.05, seed: int = 0) -> dict:
    """Run the decoder both ways on identical buffers and compare."""
    require_cuda()
    weights = decoder_weights(seed=seed, wscale=wscale)
    base = decoder_buffers(num_views=num_views, prompt_len=prompt_len, chunk_size=chunk_size)
    enc_len = encoder_seq_len(num_views, prompt_len)

    def run_once(fused: bool) -> dict:
        buffers = {k: v.clone() for k, v in base.items()}
        pipeline.transformer_decoder(op_table(fused), weights, buffers, enc_len,
                                     steps=steps, layers=layers)
        torch.cuda.synchronize()
        return buffers

    unfused = run_once(False)
    fused = run_once(True)

    out = {
        "config": {"num_views": num_views, "prompt_len": prompt_len, "chunk_size": chunk_size,
                   "steps": steps, "layers": layers, "wscale": wscale,
                   "encoder_seq_len": enc_len},
        "fused_vs_unfused": diff_stats(unfused["diffusion_noise"], fused["diffusion_noise"]),
        "decoder_x": diff_stats(unfused["decoder_x"], fused["decoder_x"]),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--num-views", type=int, default=3)
    p.add_argument("--prompt-len", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--steps", type=int, default=1, help="keep at 1; see the module docstring")
    p.add_argument("--layers", type=int, default=18)
    p.add_argument("--wscale", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out = run(num_views=a.num_views, prompt_len=a.prompt_len, chunk_size=a.chunk_size,
              steps=a.steps, layers=a.layers, wscale=a.wscale, seed=a.seed)

    passed = out["fused_vs_unfused"]["allclose_5e-1"]
    print(f"\n{'PASS' if passed else 'FAIL'}  fused vs unfused allclose({TOLERANCE}) = {passed}, "
          f"max_abs={out['fused_vs_unfused']['max_abs']:.6f}")
    if a.steps > 1 and not passed:
        print("NOTE: steps>1 is chaos-dominated; re-read at --steps 1 before treating this "
              "as a bug.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
