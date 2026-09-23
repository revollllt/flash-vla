"""Fused producer-quantize against producer + quantize at the VLA call sites.

Each site is a GEMM A operand and the kernel that produces it. For sites whose
producer is a norm or an activation, two measurements, each a set of CUDA
graphs captured once and replayed interleaved for 21 rounds, so clock drift
hits every variant alike; savings are medians of per-round differences:

- isolated: the BF16 producer, the standalone quantize (ours and FlashInfer's)
  and the fused producer, each repeated on L2-warm activations as in the model;
- chain: producer -> quantize -> GEMM(s) against fused producer -> GEMM(s),
  the GEMMs being FlashInfer b12x on cold weight copies, as in the survey.

Saving per observation = sum over sites of producers per observation x
(unfused - fused). Sites whose A operand is an attention output (or the
refiner's residual stream) keep a standalone quantize; they are timed with
ours and FlashInfer's quantize. Our kernels launch with PDL unless --no-pdl;
FlashInfer's quantize and GEMMs keep their default, PDL where supported.
Run from the repository root:

    PYTHONPATH=src CUDA_HOME=$HOME/cuda-13.1 flock /tmp/flash-vla-rtx5090.lock \\
        ~/quant-survey/venv-flashinfer/bin/python lab/quantization/bench_fused_quant.py \\
        --out results/quant-fused-rtx5090
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Callable, Literal, NamedTuple

import flashinfer
from flashinfer import fp4_quantize, mm_fp4, mm_mxfp8, mxfp8_quantize
import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import Call, interleaved_times_us, weight_copies
from flash_vla.hardware.nvidia.quant_ops import ops
from vla_shapes import weight_bytes

Producer = Literal["quantize", "gelu", "ln", "ln_affine", "adaln", "rms", "rms_mul", "adarms",
                   "swiglu", "geglu"]
Row = dict[str, str | int | float | list[int]]


class ProducerSite(NamedTuple):
    model: str
    name: str
    m: int
    k: int
    producer: Producer
    per_observation: int
    gemm_n: list[int]      # N of every GEMM that reads this A operand


# GR00T N1.7 (models/groot_n17/reference.py, packed as lab/quantization/vla_shapes.py):
# Qwen3-VL vision (LayerNorm, GELU-tanh MLP), Qwen3 LLM (RMSNorm, SwiGLU),
# refiner (LayerNorm, GELU-tanh), 32-layer DiT alternating cross/self attention
# (AdaLN before attention, affine-free LayerNorm before the FFN), 4 steps.
# Pi0.5 (models/pi05/spec.py): SigLIP (LayerNorm, GELU-tanh), Gemma with folded
# RMSNorm weights and GeGLU, action expert with AdaRMS and GeGLU, 10 steps.
SITES = [
    ProducerSite("groot", "vit ln1 -> qkv",        512, 1024, "ln_affine", 24, [3072]),
    ProducerSite("groot", "vit attn -> o",         512, 1024, "quantize", 24, [1024]),
    ProducerSite("groot", "vit ln2 -> up",         512, 1024, "ln_affine", 24, [4096]),
    ProducerSite("groot", "vit gelu -> down",      512, 4096, "gelu", 24, [1024]),
    ProducerSite("groot", "llm rms1 -> qkv",       156, 2048, "rms_mul", 16, [4096]),
    ProducerSite("groot", "llm attn -> o",         156, 2048, "quantize", 16, [2048]),
    ProducerSite("groot", "llm rms2 -> gateup",    156, 2048, "rms_mul", 16, [12288]),
    ProducerSite("groot", "llm swiglu -> down",    156, 6144, "swiglu", 16, [2048]),
    ProducerSite("groot", "ref ln1 -> q,k,v",      156, 2048, "ln_affine", 4, [2048] * 3),
    ProducerSite("groot", "ref attn -> o",         156, 2048, "quantize", 4, [2048]),
    ProducerSite("groot", "ref ln3 -> up",         156, 2048, "ln_affine", 4, [8192]),
    ProducerSite("groot", "ref gelu -> down",      156, 8192, "gelu", 4, [2048]),
    ProducerSite("groot", "context -> cross k,v",  156, 2048, "quantize", 1, [3072] * 16),
    ProducerSite("groot", "dit adaln -> qkv",      41, 1536, "adaln", 64, [4608]),
    ProducerSite("groot", "dit adaln -> cross q",  41, 1536, "adaln", 64, [1536]),
    ProducerSite("groot", "dit attn -> o",         41, 1536, "quantize", 128, [1536]),
    ProducerSite("groot", "dit ln3 -> up",         41, 1536, "ln", 128, [6144]),
    ProducerSite("groot", "dit gelu -> down",      41, 6144, "gelu", 128, [1536]),
    ProducerSite("pi05", "vit ln1 -> qkv",         768, 1152, "ln_affine", 27, [3456]),
    ProducerSite("pi05", "vit attn -> o",          768, 1152, "quantize", 27, [1152]),
    ProducerSite("pi05", "vit ln2 -> up",          768, 1152, "ln_affine", 27, [4352]),
    ProducerSite("pi05", "vit gelu -> down",       768, 4352, "gelu", 27, [1152]),
    ProducerSite("pi05", "llm rms1 -> qkv",        968, 2048, "rms", 18, [2560]),
    ProducerSite("pi05", "llm attn -> o",          968, 2048, "quantize", 18, [2048]),
    ProducerSite("pi05", "llm rms2 -> gateup",     968, 2048, "rms", 18, [32768]),
    ProducerSite("pi05", "llm geglu -> down",      968, 16384, "geglu", 18, [2048]),
    ProducerSite("pi05", "exp adarms1 -> qkv",     50, 1024, "adarms", 180, [2560]),
    ProducerSite("pi05", "exp attn -> o",          50, 2048, "quantize", 180, [1024]),
    ProducerSite("pi05", "exp adarms2 -> gateup",  50, 1024, "adarms", 180, [8192]),
    ProducerSite("pi05", "exp geglu -> down",      50, 4096, "geglu", 180, [1024]),
]
REPEAT = 64   # calls per isolated-measurement graph


def make_producer(producer: Producer, m: int, k: int, pdl: bool
                  ) -> Callable[[ops.Activation], ops.Activation]:
    """run(out): the site's producer on random bf16 inputs, written into out."""
    def _random(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, device="cuda") * scale).to(torch.bfloat16)

    x, weight, bias = _random(m, k, scale=2.0), _random(k, scale=0.2) + 1, _random(k, scale=0.1)
    gate_up = _random(m, 2 * k)                       # one [M, 2N] gate|up GEMM output
    gate, up = gate_up[:, :k], gate_up[:, k:]
    modulation = (_random(1, k, scale=0.2), _random(1, k, scale=0.2))
    factor = torch.empty(m, device="cuda", dtype=torch.bfloat16)
    return {
        "quantize": lambda out: ops.quantize(x, out, pdl=pdl),
        "gelu": lambda out: ops.gelu_tanh(x, out, pdl=pdl),
        "ln": lambda out: ops.layer_norm(x, out, pdl=pdl),
        "ln_affine": lambda out: ops.layer_norm(x, out, weight=weight, bias=bias, pdl=pdl),
        "adaln": lambda out: ops.layer_norm(x, out, modulation=modulation, pdl=pdl),
        "rms": lambda out: ops.rms_norm(x, out, pdl=pdl),
        "rms_mul": lambda out: ops.rms_norm(x, out, weight=weight, pdl=pdl),
        "adarms": lambda out: ops.rms_norm(x, out, weight=weight, round_factor=True,
                                           factor_out=factor, pdl=pdl),
        "swiglu": lambda out: ops.gated_act(gate, up, out, act="silu", round_act=True, pdl=pdl),
        "geglu": lambda out: ops.gated_act(gate, up, out, act="gelu_tanh", pdl=pdl),
    }[producer]


def gemm_calls_per_copy(activation: ops.Activation, m: int, k: int, gemm_n: list[int],
                        flashinfer_scale_shape: torch.Size) -> list[list[Call]]:
    """For each cold weight copy, the b12x GEMMs that read `activation`."""
    copies = weight_copies(sum(weight_bytes(activation.fmt, k, n) for n in gemm_n))
    calls_per_gemm = []
    for n in gemm_n:
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
        product = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        if activation.fmt == "mxfp8":
            quantized_weights = [mxfp8_quantize(weight, True) for _ in range(copies)]
            calls_per_gemm.append([
                lambda values=values, scales=scales, product=product: mm_mxfp8(
                    activation.values, values.t(), activation.scale, scales, out=product,
                    out_dtype=torch.bfloat16, backend="b12x")
                for values, scales in quantized_weights])
        else:
            weight_encode_scale = ((448 * 6) / weight.float().abs().max()).reshape(1).float()
            alpha = (1.0 / (activation.global_scale * weight_encode_scale)).float()
            quantized_weights = [fp4_quantize(weight, weight_encode_scale, 16, False, True)
                                 for _ in range(copies)]
            activation_scales = activation.scale.view(flashinfer_scale_shape)
            calls_per_gemm.append([
                lambda values=values, scales=scales, product=product: mm_fp4(
                    activation.values, values.t(), activation_scales, scales.t(), alpha,
                    torch.bfloat16, product, backend="b12x")
                for values, scales in quantized_weights])
    return [list(calls) for calls in zip(*calls_per_gemm)]


def measure_site(site: ProducerSite, fmt: Literal["mxfp8", "nvfp4"], pdl: bool) -> Row:
    run_producer = make_producer(site.producer, site.m, site.k, pdl)
    bf16 = run_producer(ops.empty(site.m, site.k, "bf16"))
    encode_scale = ((448 * 6) / bf16.values.float().abs().max()).reshape(1).float()
    fused = ops.empty(site.m, site.k, fmt, global_scale=encode_scale)
    separate = ops.empty(site.m, site.k, fmt, global_scale=encode_scale)
    quantize_separately = lambda: ops.quantize(bf16.values, separate, pdl=pdl)
    flashinfer_quantize = ((lambda: mxfp8_quantize(bf16.values, True)) if fmt == "mxfp8"
                           else (lambda: fp4_quantize(bf16.values, encode_scale, 16, False, True)))
    row: Row = dict(model=site.model, site=site.name, m=site.m, k=site.k, producer=site.producer,
                    per_obs=site.per_observation, gemm_n=site.gemm_n, fmt=fmt)
    quantize_variants: dict[str, list[Call]] = {
        "quant_ours": [quantize_separately] * REPEAT,
        "quant_flashinfer": [flashinfer_quantize] * REPEAT}
    if site.producer == "quantize":
        times = interleaved_times_us(quantize_variants, dict.fromkeys(quantize_variants, REPEAT))
        return row | {f"{name}_us": statistics.median(values) for name, values in times.items()}

    isolated_variants = quantize_variants | {
        "producer_bf16": [lambda: run_producer(bf16)] * REPEAT,
        "producer_fused": [lambda: run_producer(fused)] * REPEAT}
    times = interleaved_times_us(isolated_variants, dict.fromkeys(isolated_variants, REPEAT))
    row |= {f"{name}_us": statistics.median(values) for name, values in times.items()}
    row["isolated_saving_us"] = statistics.median(
        bf16_us + quantize_us - fused_us for bf16_us, quantize_us, fused_us
        in zip(times["producer_bf16"], times["quant_ours"], times["producer_fused"]))

    flashinfer_scale_shape = flashinfer_quantize()[1].shape
    unfused_gemms = gemm_calls_per_copy(separate, site.m, site.k, site.gemm_n,
                                        flashinfer_scale_shape)
    fused_gemms = gemm_calls_per_copy(fused, site.m, site.k, site.gemm_n, flashinfer_scale_shape)
    copies = len(fused_gemms)
    iterations = max(1, 128 // copies) * copies
    unfused_chain: list[Call] = [
        call for iteration in range(iterations)
        for call in [lambda: run_producer(bf16), quantize_separately]
        + unfused_gemms[iteration % copies]]
    fused_chain: list[Call] = [
        call for iteration in range(iterations)
        for call in [lambda: run_producer(fused)] + fused_gemms[iteration % copies]]
    times = interleaved_times_us({"chain_unfused": unfused_chain, "chain_fused": fused_chain},
                                 {"chain_unfused": iterations, "chain_fused": iterations})
    row["chain_unfused_us"] = statistics.median(times["chain_unfused"])
    row["chain_fused_us"] = statistics.median(times["chain_fused"])
    row["chain_saving_us"] = statistics.median(
        unfused_us - fused_us for unfused_us, fused_us
        in zip(times["chain_unfused"], times["chain_fused"]))
    return row


def render_summary(rows: list[Row]) -> str:
    lines = ["# Fused producer-quantize, RTX 5090", "",
             f"FlashInfer {flashinfer.__version__}, torch {torch.__version__}. "
             "Times in µs per call (isolated) or per producer + its GEMMs (chain); "
             "saving per observation in ms.", ""]
    for model in ("groot", "pi05"):
        for fmt in ("mxfp8", "nvfp4"):
            selected = [row for row in rows if row["model"] == model and row["fmt"] == fmt]
            if not selected:
                continue
            lines += [f"## {model} {fmt}", "",
                      "| site | M×K | per obs | bf16 producer | + quantize (ours / FlashInfer) "
                      "| fused | isolated saving | chain unfused | chain fused | chain saving |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            isolated_ms = chain_ms = standalone_ms = 0.0
            for row in selected:
                quantize_cell = f"{row['quant_ours_us']:.2f} / {row['quant_flashinfer_us']:.2f}"
                if row["producer"] == "quantize":
                    standalone_ms += row["per_obs"] * row["quant_ours_us"] / 1000
                    lines.append(f"| {row['site']} | {row['m']}×{row['k']} | {row['per_obs']} "
                                 f"| — | {quantize_cell} | (standalone) | | | | |")
                    continue
                isolated_ms += row["per_obs"] * row["isolated_saving_us"] / 1000
                chain_ms += row["per_obs"] * row["chain_saving_us"] / 1000
                lines.append(
                    f"| {row['site']} | {row['m']}×{row['k']} | {row['per_obs']} "
                    f"| {row['producer_bf16_us']:.2f} | {quantize_cell} "
                    f"| {row['producer_fused_us']:.2f} | {row['isolated_saving_us']:.2f} "
                    f"| {row['chain_unfused_us']:.2f} | {row['chain_fused_us']:.2f} "
                    f"| {row['chain_saving_us']:.2f} |")
            lines += ["", f"Saving per observation: isolated {isolated_ms:.3f} ms, chain "
                      f"{chain_ms:.3f} ms. Standalone quantize left (attention outputs, "
                      f"context): {standalone_ms:.3f} ms.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="",
                        help="comma-separated substrings of model/site to run")
    parser.add_argument("--no-pdl", action="store_true", help="launch our kernels without PDL")
    args = parser.parse_args()
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    patterns = args.only.split(",")
    rows = []
    for site in [site for site in SITES
                 if any(pattern in f"{site.model}/{site.name}" for pattern in patterns)]:
        for fmt in ("mxfp8", "nvfp4"):
            row = measure_site(site, fmt, pdl=not args.no_pdl)
            rows.append(row)
            print(json.dumps({key: round(value, 3) if isinstance(value, float) else value
                              for key, value in row.items()}), flush=True)
            torch.cuda.empty_cache()
    summary = render_summary(rows)
    (args.out / "raw.json").write_text(json.dumps(rows, indent=1) + "\n")
    (args.out / "summary.md").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
