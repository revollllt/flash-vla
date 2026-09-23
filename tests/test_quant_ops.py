"""Fused producer-quantize ops (hardware/nvidia/quant_ops) against their contracts.

- Fusion: every producer's MXFP8/NVFP4 output equals its own BF16 output passed
  through `quantize`, byte for byte, scales and padding included.
- Producers: the BF16 output matches `quantization.reference`, which rounds where
  the models round, up to reduction order (a rare 1-ulp difference).
- Quantize: MXFP8 equals the reference; NVFP4 scales equal it and values are
  within one E2M1 step (the kernel's reciprocals are approximate). Bit
  exactness against FlashInfer is lab/quantization/check_quant_ops.py.
"""
from typing import Callable

import pytest
import torch

from flash_vla.hardware.nvidia.quant_ops import ops
from flash_vla.quantization import formats, reference
from flash_vla.quantization.fake_quant import fake_quant_matmul, fake_quantize

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11, 12),
    reason="Blackwell GPU (sm_100 or later) required")

# (M, K): GR00T DiT, Pi0.5 action expert, GR00T LLM, and a tile-padding corner.
SHAPES = [(41, 1536), (50, 1024), (156, 2048), (7, 96)]


def random_bf16(*shape: int, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)


def assert_fused(run: Callable[[ops.Activation], ops.Activation], rows: int, cols: int,
                 expected_bf16: torch.Tensor) -> torch.Tensor:
    """run(out) writes the producer into out. Its MXFP8 and NVFP4 outputs must
    equal quantize(its BF16 output); returns that BF16 output."""
    encode_scale = ((448 * 6) / expected_bf16.float().abs().max()).reshape(1).float()
    produced_bf16 = run(ops.empty(rows, cols, "bf16")).values
    for fmt in ("mxfp8", "nvfp4"):
        fused = run(ops.empty(rows, cols, fmt, global_scale=encode_scale))
        separate = ops.quantize(produced_bf16, ops.empty(rows, cols, fmt, global_scale=encode_scale))
        assert torch.equal(fused.values.view(torch.uint8), separate.values.view(torch.uint8)), fmt
        assert torch.equal(fused.scale, separate.scale), fmt
    return produced_bf16


def assert_close_bf16(produced: torch.Tensor, expected: torch.Tensor) -> None:
    """Equal up to reduction order: at most 1 bf16 ulp, on under 1% of values."""
    produced_fp32, expected_fp32 = produced.float(), expected.float()
    ulp = torch.maximum(produced_fp32.abs(), expected_fp32.abs()).clamp(min=1e-30) * 2.0**-7
    difference = (produced_fp32 - expected_fp32).abs()
    assert torch.all(difference <= ulp * 1.01), float((difference / ulp).max())
    assert float((difference > 0).float().mean()) < 1e-2


@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("weight_mode,round_factor,with_residual", [
    ("mul", False, False),        # Qwen3
    ("none", False, False),       # Pi0.5 encoder, weights folded into the GEMM
    ("mul", True, False),         # Pi0.5 decoder AdaRMS, bf16 factor
    ("one_plus", False, False),   # Gemma
    ("mul", False, True),         # pre-norm residual add
])
def test_rms_norm(rows: int, cols: int, weight_mode: str, round_factor: bool,
                  with_residual: bool) -> None:
    x = random_bf16(rows, cols, scale=3.0, seed=1)
    weight = random_bf16(cols, scale=0.5, seed=2) + 1 if weight_mode != "none" else None
    residual = random_bf16(rows, cols, seed=3) if with_residual else None
    factor = torch.zeros(rows, device="cuda", dtype=torch.bfloat16)
    residual_sum = torch.zeros(rows, cols, device="cuda", dtype=torch.bfloat16)
    expected, expected_factor = reference.rms_norm(
        x, weight, weight_mode=weight_mode, eps=1e-6, round_factor=round_factor, residual=residual)
    produced = assert_fused(
        lambda out: ops.rms_norm(x, out, weight=weight, weight_mode=weight_mode, eps=1e-6,
                                 round_factor=round_factor, factor_out=factor,
                                 residual=residual, residual_out=residual_sum),
        rows, cols, expected)
    assert_close_bf16(produced, expected)
    assert_close_bf16(factor, expected_factor)
    assert torch.equal(residual_sum, x + residual if with_residual else torch.zeros_like(x))


@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("affine,mod_group_rows,with_residual", [
    (False, 0, False),     # GR00T DiT AdaLN, one modulation row
    (False, 4, False),     # AdaLN with one modulation row per 4 rows (batched)
    (True, None, False),   # vision encoder LayerNorm
    (True, None, True),    # with the residual add
], ids=["adaln", "adaln_grouped", "affine", "residual"])
def test_layer_norm(rows: int, cols: int, affine: bool, mod_group_rows: int | None,
                    with_residual: bool) -> None:
    x = random_bf16(rows, cols, scale=2.0, seed=4) + 0.5
    weight = random_bf16(cols, scale=0.3, seed=5) + 1 if affine else None
    bias = random_bf16(cols, scale=0.1, seed=6) if affine else None
    modulation_rows = -(-rows // mod_group_rows) if mod_group_rows else 1
    # AdaLN's linear(silu(t)).chunk(2, dim=1): two row-strided halves.
    modulation = (tuple(random_bf16(modulation_rows, 2 * cols, scale=0.2, seed=7).chunk(2, dim=1))
                  if mod_group_rows is not None else None)
    residual = random_bf16(rows, cols, seed=9) if with_residual else None
    expected = reference.layer_norm(x, weight, bias, eps=1e-5, modulation=modulation,
                                    mod_group_rows=mod_group_rows or 0, residual=residual)
    produced = assert_fused(
        lambda out: ops.layer_norm(x, out, weight=weight, bias=bias, eps=1e-5,
                                   modulation=modulation, mod_group_rows=mod_group_rows or 0,
                                   residual=residual),
        rows, cols, expected)
    assert_close_bf16(produced, expected)


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_gelu_tanh(rows: int, cols: int) -> None:
    x = random_bf16(rows, cols, scale=3.0, seed=10)
    expected = reference.gelu_tanh(x)
    assert_close_bf16(assert_fused(lambda out: ops.gelu_tanh(x, out), rows, cols, expected),
                      expected)


@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("act,round_act", [("gelu_tanh", False), ("silu", True)],
                         ids=["pi05_geglu", "hf_swiglu"])
def test_gated_act_on_packed_halves(rows: int, cols: int, act: str, round_act: bool) -> None:
    gate_up = random_bf16(rows, 2 * cols, scale=2.0, seed=11)   # one [M, 2N] gate|up GEMM output
    gate, up = gate_up[:, :cols], gate_up[:, cols:]
    expected = reference.gated_act(gate, up, act=act, round_act=round_act)
    produced = assert_fused(lambda out: ops.gated_act(gate, up, out, act=act, round_act=round_act),
                            rows, cols, expected)
    assert_close_bf16(produced, expected)


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_quantize_against_reference(rows: int, cols: int) -> None:
    x = random_bf16(rows, cols, scale=5.0, seed=12)
    x[:, :32] = 0                                   # an all-zero block in every row
    mxfp8 = ops.quantize(x, ops.empty(rows, cols, "mxfp8"))
    expected_values, expected_scales = reference.quantize_mxfp8(x)
    assert torch.equal(mxfp8.values.view(torch.uint8), expected_values.view(torch.uint8))
    assert torch.equal(formats.unswizzle(mxfp8.scale, rows, cols // 32), expected_scales)

    encode_scale = ((448 * 6) / x.float().abs().max()).reshape(1).float()
    nvfp4 = ops.quantize(x, ops.empty(rows, cols, "nvfp4", global_scale=encode_scale))
    expected_packed, expected_scales = reference.quantize_nvfp4(x, encode_scale)
    assert torch.equal(formats.unswizzle(nvfp4.scale, rows, cols // 16), expected_scales)
    produced_dequant = reference.dequantize_nvfp4(nvfp4.values, expected_scales, encode_scale)
    expected_dequant = reference.dequantize_nvfp4(expected_packed, expected_scales, encode_scale)
    e2m1_step = (expected_scales.view(torch.float8_e4m3fn).float() / encode_scale
                 ).repeat_interleave(16, 1) * 2     # the widest E2M1 gap, 4 -> 6, in real units
    assert torch.all((produced_dequant - expected_dequant).abs() <= e2m1_step)
    assert float((produced_dequant != expected_dequant).float().mean()) < 1e-2


@pytest.mark.parametrize("fmt", ["mxfp8", "nvfp4"])
def test_fake_quant_matmul_is_the_block_scaled_product(fmt: str) -> None:
    """Blocks run along K for both operands, and NVFP4's encode scales divide out."""
    x = random_bf16(41, 1536, scale=2.0, seed=40)
    weight_nk = random_bf16(512, 1536, scale=0.05, seed=41)
    product = fake_quant_matmul(fake_quantize(x, fmt), fake_quantize(weight_nk, fmt))
    dequantized = []
    for operand in (x, weight_nk):
        encode_scale = (448 * 6) / operand.float().abs().max()
        dequantized.append(
            reference.dequantize_mxfp8(*reference.quantize_mxfp8(operand)) if fmt == "mxfp8"
            else reference.dequantize_nvfp4(*reference.quantize_nvfp4(operand, encode_scale),
                                            encode_scale))
    expected = dequantized[0].double() @ dequantized[1].double().t()
    assert torch.allclose(product.double(), expected, rtol=1e-4,
                          atol=1e-4 * expected.abs().max().item())


def test_scale_padding_stays_zero_and_graph_replays() -> None:
    """96 columns are 3 MXFP8 blocks, padded to 4; 7 rows pad to 128. The padding
    is never written, and a captured producer reads new inputs on replay."""
    rows, cols = 7, 96
    x = random_bf16(rows, cols, seed=13)
    captured = ops.empty(rows, cols, "mxfp8")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ops.rms_norm(x, captured)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ops.rms_norm(x, captured)
    x.copy_(random_bf16(rows, cols, seed=14))
    graph.replay()
    torch.cuda.synchronize()
    expected = ops.quantize(ops.rms_norm(x, ops.empty(rows, cols, "bf16")).values,
                            ops.empty(rows, cols, "mxfp8"))
    assert torch.equal(captured.values.view(torch.uint8), expected.values.view(torch.uint8))
    assert torch.equal(captured.scale, expected.scale)
    written = torch.zeros_like(captured.scale, dtype=torch.bool)
    written[formats.swizzle_offsets(rows, cols // 32, "cuda").flatten()] = True
    assert torch.all(captured.scale[~written] == 0)
    assert int(written.sum()) == rows * 3


def test_pdl_chain_reads_each_input_after_its_producer() -> None:
    """A captured PDL chain in which every kernel consumes the one before it: a
    torch kernel writes the AdaLN modulation, then AdaLN -> GELU -> RMSNorm with
    residual -> quantize. Our kernels trigger their dependents early, so a read
    placed above the dependency wait would see the previous replay's data.
    Every replay must equal the same chain run eagerly without PDL."""
    rows, cols = 156, 2048
    x = random_bf16(rows, cols, seed=15)
    modulation_source = random_bf16(1, 2 * cols, scale=0.2, seed=16)
    modulation_rows = torch.empty_like(modulation_source)
    scale, shift = modulation_rows[:, :cols], modulation_rows[:, cols:]
    weight = random_bf16(cols, scale=0.5, seed=17) + 1
    def _chain_outputs() -> dict[str, torch.Tensor]:
        return {"adaln": ops.empty(rows, cols, "bf16").values,
                "gelu": ops.empty(rows, cols, "bf16").values,
                "rms": ops.empty(rows, cols, "bf16").values,
                "residual_sum": torch.zeros(rows, cols, device="cuda", dtype=torch.bfloat16)}

    def _run(outputs: dict[str, torch.Tensor], quantized: ops.Activation, pdl: bool) -> None:
        modulation_rows.copy_(modulation_source * 1.5)
        adaln = ops.layer_norm(x, ops.Activation("bf16", outputs["adaln"]),
                               modulation=(scale, shift), pdl=pdl)
        gelu = ops.gelu_tanh(adaln.values, ops.Activation("bf16", outputs["gelu"]), pdl=pdl)
        rms = ops.rms_norm(gelu.values, ops.Activation("bf16", outputs["rms"]), weight=weight,
                           residual=adaln.values, residual_out=outputs["residual_sum"], pdl=pdl)
        ops.quantize(rms.values, quantized, pdl=pdl)

    captured, captured_quantized = _chain_outputs(), ops.empty(rows, cols, "mxfp8")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(captured, captured_quantized, pdl=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _run(captured, captured_quantized, pdl=True)
    torch.cuda.current_stream().wait_stream(stream)
    for replay in range(3):
        x.copy_(random_bf16(rows, cols, seed=20 + replay))
        modulation_source.copy_(random_bf16(1, 2 * cols, scale=0.2, seed=30 + replay))
        graph.replay()
        eager, eager_quantized = _chain_outputs(), ops.empty(rows, cols, "mxfp8")
        _run(eager, eager_quantized, pdl=False)
        torch.cuda.synchronize()
        for name, captured_stage in captured.items():
            assert torch.equal(captured_stage, eager[name]), (replay, name)
        assert torch.equal(captured_quantized.values.view(torch.uint8),
                           eager_quantized.values.view(torch.uint8)), replay
        assert torch.equal(captured_quantized.scale, eager_quantized.scale), replay


def test_mxfp8_rejects_unpadded_siglip_width() -> None:
    """Pi0.5's SigLIP FFN width 4304 is 134.5 MXFP8 blocks; it must be padded to 4352."""
    with pytest.raises(ValueError, match="divisible by 32"):
        ops.empty(768, 4304, "mxfp8")
