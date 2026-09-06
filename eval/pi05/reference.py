"""Official-baseline checks of the H100/Pi0.5 Target against OpenPI (two stages).

    python -m eval.pi05.reference --stage llm_backbone      # the prefix KV cache
    python -m eval.pi05.reference --stage action_expert     # the denoised chunk
    python -m eval.pi05.reference                           # both, in that order

Both need the OpenPI environment (`OPENPI_PYTHON`, see `eval/acceptance.py`).

## llm_backbone

The prefix pass is everything up to the KV cache the action expert attends
over: the vision encoder, the prompt embedding gather, and 18 backbone layers.
Its output is `prefix_k` / `prefix_v`, which is exactly what OpenPI's
`sample_actions` builds before it starts denoising (`pi0_pytorch.py:185-201`),
so the two are directly comparable.

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

## action_expert

The backbone stage covers vision and the prefix; `lab/pi05/kernels.py` covers each
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

from benchmarks.targets import PLAN_NAMES
from eval.acceptance import tolerances
from eval.baselines import openpi05
from eval.metrics import error_metrics
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


def run_backbone(tokenizer_path: str | None = None, checkpoint: str | None = None,
        prompt: str = DEFAULT_PROMPT, layers: int = 18, seed: int = 0,
        device: str = "cuda", exact_rope: bool = True,
        plan: str | None = None) -> dict[str, object]:
    """Run both implementations on identical inputs and report per-layer error.

    `plan` names the call-site plan the runner is built with; the default is
    the reference route. Pass `shipped` to gate what production actually runs
    -- the backbone attention is plan-selected, so the reference route does
    not exercise its CUDA kernel.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this command on an H100 GPU node")

    from flash_vla.hardware.nvidia.h100.pi05 import TARGET, forward_prefix
    from flash_vla.runtime import ModelRunner

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

    engine = ModelRunner(TARGET, target_weights, plan=plan or "reference", device=device,
                         num_views=3, chunk_size=50, layers=layers, tokenizer=tokenizer,
                         prompt=prompt)
    del target_weights
    torch.cuda.empty_cache()

    engine_n_valid = forward_prefix(engine, images, state)
    torch.cuda.synchronize()
    keys, values = engine.buffers["prefix_k"], engine.buffers["prefix_v"]
    prefix_len = engine.derived["prefix_len"]

    report: dict[str, object] = {
        "identity": engine.identity.as_dict(),
        "prompt_tokens": n_tokens,
        "n_valid_prefix": n_valid,
        "n_valid_engine": engine_n_valid,
        "prefix_len": prefix_len,
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

    padded = torch.cat([keys[:, n_valid:prefix_len], values[:, n_valid:prefix_len]])
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


CHUNK = 50

#: One step on random weights is a direct reading of the wiring, so it is held
#: tightly. Ten steps is the chaotic regime and is reported, not gated.
STEP1_COSINE = tolerances("bf16")["shallow_cosine"]


def _transplant(engine, reference_cache, seq_len: int) -> None:
    """Write OpenPI's prefix K/V into the runner's cache, in the target's layout."""
    keys, values = engine.buffers["kv_k"], engine.buffers["kv_v"]
    for index, (ref_k, ref_v) in enumerate(reference_cache):
        keys[index, :seq_len] = _to_pair_layout(ref_k[0, 0, :seq_len]).bfloat16()
        values[index, :seq_len] = ref_v[0, 0, :seq_len].bfloat16()


def run_expert(tokenizer_path: str | None = None, checkpoint: str | None = None,
        prompt: str = DEFAULT_PROMPT, steps: int = 1, layers: int = 18, seed: int = 0,
        full: bool = False, device: str = "cuda") -> dict[str, object]:
    """Run both decoders on identical inputs and report the action-chunk error."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this command on an H100 GPU node")

    from flash_vla.hardware.nvidia.h100.pi05 import TARGET, forward_prefix
    from flash_vla.runtime import ModelRunner

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

    engine = ModelRunner(TARGET, target_weights, plan="reference", device=device, num_views=3,
                         chunk_size=CHUNK, steps=steps, layers=layers, tokenizer=tokenizer,
                         prompt=prompt)
    del target_weights
    torch.cuda.empty_cache()

    engine.buffers["actions"].copy_(noise)
    engine_n_valid = forward_prefix(engine, images, state)     # sets rope, mask, n_valid
    if not full:
        _transplant(engine, cache, engine.derived["prefix_len"])
    engine.replay("action_expert")
    torch.cuda.synchronize()
    output = engine.buffers["actions"].float().clone()

    report: dict[str, object] = {
        "identity": engine.identity.as_dict(),
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=["llm_backbone", "action_expert", "all"], default="all")
    parser.add_argument("--tokenizer", default=os.environ.get("PALIGEMMA_TOKENIZER"),
                        help="paligemma_tokenizer.model (default: $PALIGEMMA_TOKENIZER)")
    parser.add_argument("--checkpoint", default=None,
                        help="OpenPI pi05 model.safetensors or its directory (default: random)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--layers", type=int, default=18, help="depth of the checked stage, for bisection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan", default=None,
                        help=f"llm_backbone only: one of {PLAN_NAMES}, JSON or a lab/plans/*.json "
                             "path (default: reference)")
    parser.add_argument("--openpi-rope-bf16", action="store_true",
                        help="llm_backbone only: keep OpenPI's bfloat16 rotary frequencies")
    parser.add_argument("--steps", type=int, default=1,
                        help="action_expert only: flow steps; read 1 first, 10 is the chaotic regime")
    parser.add_argument("--full", action="store_true",
                        help="action_expert only: run end to end instead of transplanting the cache")
    args = parser.parse_args(argv)
    passed = True
    if args.stage in ("llm_backbone", "all"):
        passed &= run_backbone(args.tokenizer, args.checkpoint, args.prompt, args.layers, args.seed,
                               args.device, exact_rope=not args.openpi_rope_bf16,
                               plan=args.plan)["passed"]
    if args.stage in ("action_expert", "all"):
        passed &= run_expert(args.tokenizer, args.checkpoint, args.prompt, args.steps, args.layers,
                             args.seed, args.full, args.device)["passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
