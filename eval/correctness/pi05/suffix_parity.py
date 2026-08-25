"""Compare the H100/Pi0.5 decoder against OpenPI's flow-matching loop.

The prefix gate covers vision and the encoder; `kernel_parity` covers each
AdaRMSNorm kernel in isolation. What neither covers is the *wiring* -- which
per-(step, layer) table slice reaches which kernel, whether the residual aliases
correctly, whether the suffix RoPE offset is right. That is what this checks,
and it is exactly the class of mistake that produces a plausible wrong number.

**The KV cache is transplanted, on purpose.** By default this hands our decoder
OpenPI's own prefix cache rather than the one our encoder built. The two agree
to a layer-0 cosine of 0.99994 but drift to 0.9973 by layer 17 on random
weights, and feeding that drift into the decoder would mix two error sources in
one number. Transplanting removes the prefix entirely from the comparison, so
what remains is decoder wiring and decoder kernels.

`--full` does the opposite: both implementations run end to end from the same
images and noise. That is the deployment number, and it is the right one to read
*after* this passes, never instead of it.

**Read it at `--steps 1` first.** The flow loop is a chaotic map on random
untrained weights: a per-kernel difference around 1e-3 compounds into a
macroscopic output difference over ten steps and eighteen layers, for any two
implementations that are not bit-identical. Depth is the amplifier, not the
wiring. `--layers` narrows it further when something does look wrong, and it truncates
*both* implementations -- an early version of this gate cut only ours and
reported a meaningless 0.995 against an 18-layer reference.

The transplant needs one transform. Our K is stored in the target's
adjacent-pair RoPE layout while OpenPI's is half-split, so channel `2p` here is
OpenPI's `p` and `2p+1` is its `p+128`. V carries no rotation and transfers
directly.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from eval.baselines import openpi05
from eval.correctness.pi05.prefix_parity import (
    DEFAULT_PROMPT,
    _cache_layers,
    _to_pair_layout,
    error_metrics,
)
from flash_vla.models.pi05.spec import VISION_TOKENS
from flash_vla.models.pi05.tokenize import Pi05Tokenizer
from flash_vla.models.pi05.weights import fold

CHUNK = 50

#: One step on random weights is a direct reading of the wiring, so it is held
#: tightly. Ten steps is the chaotic regime and is reported, not gated.
STEP1_COSINE = 0.999


def _transplant(engine, reference_cache, seq_len: int) -> None:
    """Write OpenPI's prefix K/V into the engine's buffers, in the target's layout."""
    keys, values = engine.buffers["encoder_K"], engine.buffers["encoder_V"]
    for index, (ref_k, ref_v) in enumerate(reference_cache):
        keys[index, :seq_len] = _to_pair_layout(ref_k[0, 0, :seq_len]).bfloat16()
        values[index, :seq_len] = ref_v[0, 0, :seq_len].bfloat16()


def run(tokenizer_path: str | None = None, checkpoint: str | None = None,
        prompt: str = DEFAULT_PROMPT, steps: int = 1, layers: int = 18, seed: int = 0,
        full: bool = False, device: str = "cuda") -> dict[str, object]:
    """Run both decoders on identical inputs and report the action-chunk error."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this command on an H100 GPU node")

    from flash_vla.hardware.nvidia.h100.pi05 import Pi05Inference

    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    images = torch.rand((3, 224, 224, 3), generator=generator, device=torch_device,
                        dtype=torch.float32) * 2.0 - 1.0
    state = torch.randn((32,), generator=generator, device=torch_device, dtype=torch.float32)
    noise = torch.randn((CHUNK, 32), generator=generator, device=torch_device,
                        dtype=torch.float32)

    tokenizer = Pi05Tokenizer(tokenizer_path)
    tokenizer.set_task(prompt)
    tokens, mask = tokenizer.encode(state.cpu().numpy())
    n_valid = 3 * VISION_TOKENS + int(mask.sum())

    baseline = openpi05.build_model(checkpoint, torch_device, seed=seed)
    # Both sides must run the same depth; our engine takes `layers`, the
    # reference has to be cut. The prefix is a different module and stays whole.
    reference_layers = openpi05.truncate_expert(baseline, layers)
    past_key_values, _, pad_masks = openpi05.prefix_kv_cache(
        baseline, images, state,
        torch.from_numpy(tokens.astype("int64")).to(torch_device),
        torch.from_numpy(mask).to(torch_device))
    reference = openpi05.denoise(baseline, state, pad_masks, past_key_values,
                                 noise.unsqueeze(0), num_steps=steps)[0].float().clone()
    cache = [(k.detach().clone(), v.detach().clone()) for k, v in _cache_layers(past_key_values)]
    target_weights = fold(openpi05.target_checkpoint(baseline), steps=steps)
    del baseline, past_key_values
    torch.cuda.empty_cache()

    engine = Pi05Inference(target_weights, tokenizer, num_views=3, chunk_size=CHUNK,
                           steps=steps, layers=layers, device=device)
    del target_weights
    torch.cuda.empty_cache()
    engine.set_task(prompt)

    engine.buffers["diffusion_noise"].copy_(noise)
    engine_n_valid = engine.forward_prefix(images, state)     # sets rope, mask, n_valid
    if not full:
        _transplant(engine, cache, engine.encoder_seq_len)
    engine.decoder_graph.replay()
    torch.cuda.synchronize()
    output = engine.buffers["diffusion_noise"].float().clone()

    report: dict[str, object] = {
        "mode": "full pass" if full else "transplanted KV cache",
        "steps": steps,
        "layers": layers,
        "reference_layers": reference_layers,
        "prompt_tokens": int(mask.sum()),
        "n_valid_prefix": n_valid,
        "n_valid_engine": engine_n_valid,
        "checkpoint": checkpoint or "random",
        "metrics": error_metrics(reference, output),
    }
    report["gated"] = steps == 1 and not full
    report["threshold"] = STEP1_COSINE if report["gated"] else None
    report["passed"] = bool(
        engine_n_valid == n_valid
        and torch.isfinite(output).all().item()
        and (not report["gated"]
             or report["metrics"]["cosine_similarity"] > STEP1_COSINE))
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=os.environ.get("PALIGEMMA_TOKENIZER"))
    parser.add_argument("--checkpoint", default=None,
                        help="OpenPI pi05 model.safetensors or its directory (default: random)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=1,
                        help="flow-matching steps; read 1 first, 10 is the chaotic regime")
    parser.add_argument("--layers", type=int, default=18, help="decoder depth, for bisection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full", action="store_true",
                        help="run both end to end instead of transplanting the KV cache")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    passed = run(args.tokenizer, args.checkpoint, args.prompt, args.steps, args.layers,
                 args.seed, args.full, args.device)["passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
