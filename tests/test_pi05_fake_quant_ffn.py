"""The Pi0.5 fake-quant FFN backend follows its per-layer recipe and keeps the
BF16 route's rounding points: a BF16 layer equals the torch route exactly, and a
quantized layer equals the fake-quant composition written out here, when
replayed from a CUDA graph as the runner replays it."""
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from flash_vla.hardware.nvidia.rtx5090.pi05.backends import torch_ops
from flash_vla.hardware.nvidia.rtx5090.pi05.backends.fake_quant_ffn import (FakeQuantFFN,
                                                                         kernel_fake_quantize)
from flash_vla.models.pi05.spec import ENCODER_LAYERS
from flash_vla.quantization.fake_quant import fake_quant_matmul
from flash_vla.runtime.runner import Scratch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11, 12),
    reason="Blackwell GPU (sm_100 or later) required")


def random_bf16(*shape: int, scale: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)


def test_fake_quant_ffn_follows_its_recipe_per_layer(tmp_path: Path) -> None:
    rows, dim, ffn = 64, 256, 512
    recipe = {"gate_up": ["mxfp8", "bf16"] + ["mxfp8"] * (ENCODER_LAYERS - 2),
              "down": ["bf16", "nvfp4"] + ["mxfp8"] * (ENCODER_LAYERS - 2)}
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    wrappers = FakeQuantFFN("bf16").make_wrappers(
        Scratch(torch.device("cuda"), assets={"quantization_recipe": tmp_path / "recipe.json"}))
    gated, down = (wrappers["llm_backbone_norm_gated_ffn_masked"],
                   wrappers["llm_backbone_ffn_down_residual_masked"])
    x = random_bf16(rows, dim, scale=2.0, seed=1)
    stacked_gate, stacked_up = (random_bf16(2, dim, ffn, scale=0.05, seed=2),
                                random_bf16(2, dim, ffn, scale=0.05, seed=3))
    stacked_down = random_bf16(2, ffn, dim, scale=0.05, seed=4)
    hidden = random_bf16(rows, ffn, scale=1.0, seed=5)
    residual = random_bf16(rows, dim, scale=1.0, seed=6)
    gated_out = torch.empty(2, rows, ffn, device="cuda", dtype=torch.bfloat16)
    summed = residual.expand(2, rows, dim).clone()
    x_norm = torch.empty_like(x)

    def _forward() -> None:
        for layer in range(2):   # the forward's order numbers the layers
            gated(x, stacked_gate[layer], stacked_up[layer], gated_out[layer], x_norm, None)
            down(hidden, stacked_down[layer], summed[layer], None)

    # Eager warmup quantizes the weights; the captured replay must match it.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _forward()
    summed.copy_(residual.expand(2, rows, dim))
    graph.replay()
    torch.cuda.synchronize()
    outputs = [(gated_out[layer], summed[layer]) for layer in range(2)]

    x_fp32 = x.float()
    normalized = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + torch_ops.RMS_EPS)
                  ).to(torch.bfloat16)
    activation = kernel_fake_quantize(normalized, "mxfp8")
    gate = fake_quant_matmul(activation, kernel_fake_quantize(stacked_gate[0].t().contiguous(), "mxfp8"))
    up = fake_quant_matmul(activation, kernel_fake_quantize(stacked_up[0].t().contiguous(), "mxfp8"))
    expected_gated = (F.gelu(gate.bfloat16().float(), approximate="tanh")
                      * up.bfloat16().float()).bfloat16()
    assert torch.equal(outputs[0][0], expected_gated)
    torch_gated = torch.empty_like(outputs[1][0])
    torch_ops.llm_backbone_norm_gated_ffn(x, stacked_gate[1], stacked_up[1], torch_gated,
                                          torch.empty_like(x))
    assert torch.equal(outputs[1][0], torch_gated)
    assert torch.equal(outputs[0][1], residual.clone().addmm_(hidden, stacked_down[0]))
    product = fake_quant_matmul(kernel_fake_quantize(hidden, "nvfp4"),
                                kernel_fake_quantize(stacked_down[1].t().contiguous(), "nvfp4"))
    assert torch.equal(outputs[1][1], (residual.float() + product).bfloat16())
