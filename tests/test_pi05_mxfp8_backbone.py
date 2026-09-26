"""The MXFP8 backbone FFN computes its fake-quant reference at every row count a
Pi0.5 replay bucket runs, per layer, replayed from one CUDA graph as the runner
replays it."""
import pytest
import torch

from flash_vla.hardware.nvidia.rtx5090.pi05.backends import mxfp8_backbone
from flash_vla.hardware.nvidia.rtx5090.pi05.backends.fake_quant_ffn import FakeQuantFFN
from flash_vla.runtime.runner import Scratch
from eval.metrics import error_metrics

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="RTX 5090 class GPU (sm_120) required")

WIDTH, HIDDEN, LAYERS = 2048, 16384, 2   # model width, FFN width


def random_bf16(*shape: int, scale: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)


# Every row count a replay bucket runs at, each on its own screened down configuration.
@pytest.mark.parametrize("rows", sorted(mxfp8_backbone.DOWN_CONFIGS))
def test_mxfp8_backbone_matches_fake_quant_reference(rows: int) -> None:
    x = random_bf16(rows, WIDTH, scale=3.0, seed=1)
    residual = random_bf16(rows, WIDTH, scale=1.0, seed=2)
    gate_w, up_w = (random_bf16(LAYERS, WIDTH, HIDDEN, scale=0.03, seed=3),
                    random_bf16(LAYERS, WIDTH, HIDDEN, scale=0.03, seed=4))
    down_w = random_bf16(LAYERS, HIDDEN, WIDTH, scale=0.01, seed=5)
    # Each implementation's wrappers and buffers stay alive as long as its graph.
    graphs, wrappers, hidden, summed, x_norm = {}, {}, {}, {}, {}
    for name, factory in (("kernels", mxfp8_backbone.make_wrappers),
                          ("reference", FakeQuantFFN("mxfp8").make_wrappers)):
        wrappers[name] = factory(Scratch(torch.device("cuda")))
        hidden[name] = torch.zeros(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
        x_norm[name] = torch.zeros_like(x)
        summed[name] = residual.expand(LAYERS, rows, WIDTH).clone()

        def forward(name: str = name) -> None:
            gated, down = (wrappers[name]["llm_backbone_norm_gated_ffn"],
                           wrappers[name]["llm_backbone_ffn_down_residual"])
            for layer in range(LAYERS):
                gated(x, gate_w[layer], up_w[layer], hidden[name], x_norm[name])
                down(hidden[name], down_w[layer], summed[name][layer])

        # Eager warmup plans and quantizes the weights; the replay must use them.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
            graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name], stream=stream):
                forward()
    for name, graph in graphs.items():
        summed[name].copy_(residual.expand(LAYERS, rows, WIDTH))
        graph.replay()
    torch.cuda.synchronize()
    for layer in range(LAYERS):
        # The FFN's contribution to the residual.
        metrics = error_metrics(summed["reference"][layer].float() - residual.float(),
                                summed["kernels"][layer].float() - residual.float())
        # Measured 1.7e-4: FP32 summation order, and a 1-ulp RMSNorm difference that
        # can move an E4M3 code (quant_ops README). A layout or scale error is O(1).
        assert metrics["rel_rms"] < 1e-3 and metrics["cosine_similarity"] > 0.999999, (
            rows, layer, metrics)
