"""Compare the H100/Pi0.5 prefix pass against the official OpenPI PyTorch model.

The prefix pass is everything up to the KV cache the decoder attends over:
vision, the prompt embedding gather, and 18 encoder layers. Its output is
`encoder_K` / `encoder_V`, which is exactly what OpenPI's `sample_actions`
builds before it starts denoising (`pi0_pytorch.py:185-201`), so the two are
directly comparable.

Running this on randomly initialized weights is deliberate. Both sides consume
the same tensors, so any difference is an implementation difference; a
`pi05_base` checkpoint would additionally test the conversion of trained values
but says nothing extra about the code. Use `--checkpoint` for that.

Two things make the comparison less trivial than it looks.

**RoPE layout.** The target rotates adjacent channel pairs while OpenPI rotates
half-splits, and `openpi.py:_interleave_rope` permutes the q/k weights so the
two produce the same values in a different channel order: target channel `2p` is
OpenPI's `p` and `2p+1` is OpenPI's `p+128`. K is compared after applying that
permutation to the reference. V carries no rotation and is compared directly.

**How to read the numbers.** Layer 0 is the gate. A structural error -- wrong
weight layout, wrong RoPE, wrong mask -- shows up there at full size, because
nothing has accumulated yet. Deeper layers drift as bfloat16 rounding compounds
through 27 vision and 18 encoder layers, and on random weights that drift is
large and means nothing: the generic in-engine runner makes the same point
about depth being the amplifier. So the pass criterion is tight at layer 0, loose at depth,
and additionally requires the degradation to be smooth -- a step change between
consecutive layers would be a real bug at that layer, which a single aggregate
number would hide.

**Padded rows.** Only `768 + n_valid` prefix rows carry data. The rest are
masked out of every later attention, and this target lets their query rows
attend normally where OpenPI zeroes them, so the two disagree there by
construction. The comparison is over valid rows only; the padded rows are
checked for finiteness instead, because a NaN there would survive the mask
(`0 * NaN = NaN`) and reach the decoder.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from benchmarks.plans import PLANS, parse_plan
from eval.acceptance import tolerances
from eval.baselines import openpi05
from eval.correctness.metrics import error_metrics
from flash_vla.models.pi05.spec import HEAD_DIM, VISION_TOKENS
from flash_vla.models.pi05.tokenize import Pi05Tokenizer
from flash_vla.models.pi05.weights import fold

DEFAULT_PROMPT = "pick up the plate and put it in the sink"

#: Thresholds come from the acceptance registry, keyed by precision policy.
#: Layer 0 carries no accumulated error, so it is held tightly; by layer 17 the
#: input has been through 45 bfloat16 layers and drift is expected on random
#: weights; a single layer losing more than the step is a bug in that layer.
_TOL = tolerances("bf16")
LAYER0_COSINE = _TOL["layer0_cosine"]
DEEPEST_COSINE = _TOL["deepest_cosine"]
MAX_COSINE_STEP = _TOL["max_cosine_step"]


def _to_pair_layout(x: torch.Tensor) -> torch.Tensor:
    """OpenPI's half-split channel order -> the target's adjacent-pair order."""
    return x.view(*x.shape[:-1], 2, HEAD_DIM // 2).transpose(-1, -2).reshape(x.shape)


def _cache_layers(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Normalize the several shapes a transformers cache can take."""
    if hasattr(past_key_values, "key_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache, strict=True))
    if hasattr(past_key_values, "layers"):
        return [(layer.keys, layer.values) for layer in past_key_values.layers]
    return [(k, v) for k, v in past_key_values]


def run(tokenizer_path: str | None = None, checkpoint: str | None = None,
        prompt: str = DEFAULT_PROMPT, layers: int = 18, seed: int = 0,
        device: str = "cuda", exact_rope: bool = True,
        plan: str | None = None) -> dict[str, object]:
    """Run both implementations on identical inputs and report per-layer error.

    `plan` names the call-site plan the target engine is built with; the default
    is the all-TileLang reference route. Pass the shipped plan to gate what
    production actually runs -- the encoder attention is plan-selected, so the
    reference route does not exercise its CUDA kernel.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this command on an H100 GPU node")

    from flash_vla.hardware.nvidia.h100.pi05 import Pi05Inference

    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    images = torch.rand((3, 224, 224, 3), generator=generator, device=torch_device,
                        dtype=torch.float32) * 2.0 - 1.0
    state = torch.randn((32,), generator=generator, device=torch_device, dtype=torch.float32)

    tokenizer = Pi05Tokenizer(tokenizer_path)
    tokenizer.set_task(prompt)
    tokens, mask = tokenizer.encode(state.cpu().numpy())
    n_tokens = int(mask.sum())
    n_valid = 3 * VISION_TOKENS + n_tokens

    baseline = openpi05.build_model(checkpoint, torch_device, seed=seed,
                                    exact_rope=exact_rope)
    rope_freqs = baseline.paligemma_with_expert.paligemma.model.language_model.rotary_emb.inv_freq
    past_key_values, _, _ = openpi05.prefix_kv_cache(
        baseline, images, state,
        torch.from_numpy(tokens.astype("int64")).to(torch_device),
        torch.from_numpy(mask).to(torch_device))
    reference = [(k.detach().clone(), v.detach().clone()) for k, v in _cache_layers(past_key_values)]
    target_weights = fold(openpi05.target_checkpoint(baseline))
    del baseline, past_key_values
    torch.cuda.empty_cache()

    engine = Pi05Inference(target_weights, tokenizer, num_views=3, chunk_size=50,
                           layers=layers, device=device,
                           plan=parse_plan(plan) if plan else None)
    del target_weights
    torch.cuda.empty_cache()

    engine.set_task(prompt)
    engine_n_valid = engine.forward_prefix(images, state)
    torch.cuda.synchronize()
    keys, values = engine.kv_cache

    report: dict[str, object] = {
        "identity": engine.identity.as_dict(),
        "prompt_tokens": n_tokens,
        "n_valid_prefix": n_valid,
        "n_valid_engine": engine_n_valid,
        "encoder_seq_len": engine.encoder_seq_len,
        "layers_compared": min(layers, len(reference)),
        "checkpoint": checkpoint or "random",
        "plan": engine.plan,
        "exact_rope": exact_rope,
        "reference_inv_freq": [round(float(v), 7) for v in rope_freqs[:4]],
    }
    if engine_n_valid != n_valid:
        report["passed"] = False
        report["error"] = "engine and reference disagree on the valid prefix length"
        print(json.dumps(report, indent=2))
        return report

    per_layer = []
    for index in range(min(layers, len(reference))):
        ref_k, ref_v = reference[index]
        # [batch, kv_heads, seq, head_dim] -> [seq, head_dim]; one kv head.
        ref_k = _to_pair_layout(ref_k[0, 0, :n_valid])
        ref_v = ref_v[0, 0, :n_valid]
        per_layer.append({
            "layer": index,
            "k": error_metrics(ref_k, keys[index, :n_valid]),
            "v": error_metrics(ref_v, values[index, :n_valid]),
        })

    padded = torch.cat([keys[:, n_valid:engine.encoder_seq_len],
                        values[:, n_valid:engine.encoder_seq_len]])
    report["padded_rows_finite"] = bool(torch.isfinite(padded.float()).all().item())
    report["per_layer"] = per_layer

    cosines = [min(l["k"]["cosine_similarity"], l["v"]["cosine_similarity"]) for l in per_layer]
    steps = [cosines[i] - cosines[i + 1] for i in range(len(cosines) - 1)]
    report["cosine"] = {
        "layer0": cosines[0],
        "deepest": cosines[-1],
        "worst_step": max(steps) if steps else 0.0,
    }
    report["worst"] = {
        "max_abs": max(max(l["k"]["max_abs"], l["v"]["max_abs"]) for l in per_layer),
        "min_cosine": min(cosines),
    }
    report["thresholds"] = {"layer0": LAYER0_COSINE, "deepest": DEEPEST_COSINE,
                            "step": MAX_COSINE_STEP}
    report["passed"] = bool(
        report["padded_rows_finite"]
        and cosines[0] > LAYER0_COSINE
        and cosines[-1] > DEEPEST_COSINE
        and (not steps or max(steps) < MAX_COSINE_STEP))
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=os.environ.get("PALIGEMMA_TOKENIZER"),
                        help="paligemma_tokenizer.model (default: $PALIGEMMA_TOKENIZER)")
    parser.add_argument("--checkpoint", default=None,
                        help="OpenPI pi05 model.safetensors or its directory (default: random)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--layers", type=int, default=18, help="encoder depth, for bisection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan", default=None,
                        help=f"call-site plan for the target engine: one of "
                             f"{sorted(PLANS)} or JSON (default: all-TileLang)")
    parser.add_argument("--openpi-rope-bf16", action="store_true",
                        help="keep OpenPI's bfloat16 rotary frequencies (see "
                             "openpi05.restore_rope_precision)")
    args = parser.parse_args(argv)
    passed = run(args.tokenizer, args.checkpoint, args.prompt, args.layers,
                 args.seed, args.device, exact_rope=not args.openpi_rope_bf16,
                 plan=args.plan)["passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
