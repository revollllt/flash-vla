"""Do two call-site plans produce the same decoder? (`Pi05Inference(plan=...)`)

The reference plan is all-TileLang; the candidate routes call sites to the
`cuda` backend. Both engines see one random checkpoint and one input; the
report compares the action chunk and, per layer, the cache suffix rows the
decoder's qkv projection writes (`buffers["encoder_K"][:, prefix:]`).

**Read it at `--steps 1 --layers 1` first**, then deepen. The flow loop is a
chaotic map on random weights (see `suffix_parity`), so ten steps and eighteen
layers separate any two non-bit-identical implementations; the gate is the
shallow run, the deep run is reported.

    python -m eval.correctness.pi05.plan_parity --steps 1 --layers 1
    python -m eval.correctness.pi05.plan_parity --steps 1 --layers 18
"""

from __future__ import annotations

import argparse
import json

import torch

from benchmarks.e2e_pi05 import DEFAULT_PROMPT, PLANS, parse_plan
from flash_vla.models.pi05.tokenize import Pi05Tokenizer
from flash_vla.models.pi05.weights import fold, random_checkpoint

from .prefix_parity import error_metrics

GATE_COSINE = 0.999


def _forward(engine, images, state, noise):
    engine.forward(images, state, noise)
    torch.cuda.synchronize()
    prefix = engine.encoder_seq_len
    chunk = engine.chunk_size
    return {
        "actions": engine.buffers["diffusion_noise"].clone(),
        "k_suffix": engine.buffers["encoder_K"][:engine.layers, prefix:prefix + chunk].clone(),
        "v_suffix": engine.buffers["encoder_V"][:engine.layers, prefix:prefix + chunk].clone(),
    }


def run(plan: str = "attn-cuda", num_views: int = 3, chunk_size: int = 50, steps: int = 1,
        layers: int = 1, seed: int = 0, prompt: str = DEFAULT_PROMPT,
        tokenizer_path: str | None = None, device: str = "cuda") -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this on an H100 node")
    from flash_vla.hardware.nvidia.h100.pi05 import Pi05Inference

    tokenizer = Pi05Tokenizer(tokenizer_path)
    checkpoint = fold(random_checkpoint(seed=seed, device=device), steps=steps)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    images = torch.randn(num_views, 224, 224, 3, generator=generator, device=device,
                         dtype=torch.bfloat16)
    state = torch.randn(32, generator=generator, device=device, dtype=torch.float32)
    noise = torch.randn(chunk_size, 32, generator=generator, device=device, dtype=torch.bfloat16)

    outputs = {}
    for name in ("tilelang", plan):
        engine = Pi05Inference(checkpoint, tokenizer, num_views=num_views, chunk_size=chunk_size,
                               steps=steps, layers=layers, device=device, plan=parse_plan(name))
        engine.set_task(prompt)
        _forward(engine, images, state, noise)              # settles n_valid
        outputs[name] = _forward(engine, images, state, noise)
        # replay determinism of the candidate: a second replay is bit-identical
        again = _forward(engine, images, state, noise)
        outputs[name]["replay_identical"] = all(
            torch.equal(outputs[name][k], again[k]) for k in ("actions", "k_suffix", "v_suffix"))
        del engine
        torch.cuda.empty_cache()

    ref, got = outputs["tilelang"], outputs[plan]
    report = {"plan": parse_plan(plan), "steps": steps, "layers": layers, "seed": seed,
              "replay_identical": {n: outputs[n]["replay_identical"] for n in outputs},
              "actions": error_metrics(ref["actions"], got["actions"]),
              "k_suffix": error_metrics(ref["k_suffix"], got["k_suffix"]),
              "v_suffix": error_metrics(ref["v_suffix"], got["v_suffix"]),
              "k_suffix_per_layer": [error_metrics(ref["k_suffix"][i], got["k_suffix"][i])["cosine_similarity"]
                                     for i in range(layers)]}
    cos = [report[k]["cosine_similarity"] for k in ("actions", "k_suffix", "v_suffix")]
    finite = all(torch.isfinite(got[k]).all().item() for k in ("actions", "k_suffix", "v_suffix"))
    report["passed"] = bool(finite and got["replay_identical"] and min(cos) > GATE_COSINE)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default="attn-cuda", help=f"one of {sorted(PLANS)} or JSON")
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args(argv)
    report = run(args.plan, args.num_views, args.chunk_size, args.steps, args.layers, args.seed,
                 args.prompt, args.tokenizer)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
