"""The MXFP8 backbone FFN computes its fake-quant reference at Pi0.5's shapes,
per layer, replayed from one CUDA graph as the runner replays it: at the
three-view prefix while the mask switches between the two row buckets, and at
the unbucketed two-view prefix."""
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


# Prefix rows (three views, two views) and the valid rows each replay masks in.
@pytest.mark.parametrize("rows, valid_rows_cases", [(968, (968, 896)), (712, (712,))])
def test_mxfp8_backbone_matches_fake_quant_reference(rows: int,
                                                    valid_rows_cases: tuple[int, ...]) -> None:
    x = random_bf16(rows, WIDTH, scale=3.0, seed=1)
    residual = random_bf16(rows, WIDTH, scale=1.0, seed=2)
    gate_w, up_w = (random_bf16(LAYERS, WIDTH, HIDDEN, scale=0.03, seed=3),
                    random_bf16(LAYERS, WIDTH, HIDDEN, scale=0.03, seed=4))
    down_w = random_bf16(LAYERS, HIDDEN, WIDTH, scale=0.01, seed=5)
    mask = torch.zeros(rows, device="cuda", dtype=torch.bfloat16)   # additive, 0 = valid
    graphs, summed = {}, {}
    for name, wrappers in (("kernels", mxfp8_backbone.make_wrappers(Scratch(torch.device("cuda")))),
                           ("reference", FakeQuantFFN("mxfp8").make_wrappers(
                               Scratch(torch.device("cuda"))))):
        gated, down = (wrappers["llm_backbone_norm_gated_ffn_masked"],
                       wrappers["llm_backbone_ffn_down_residual_masked"])
        hidden = torch.zeros(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
        x_norm = torch.zeros_like(x)
        summed[name] = residual.expand(LAYERS, rows, WIDTH).clone()

        def _forward() -> None:
            for layer in range(LAYERS):
                gated(x, gate_w[layer], up_w[layer], hidden, x_norm, mask)
                down(hidden, down_w[layer], summed[name][layer], mask)

        # Eager warmup plans and quantizes the weights; the replay must use them.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _forward()
            graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name], stream=stream):
                _forward()
    for valid_rows in valid_rows_cases:
        mask.fill_(0.0)
        mask[valid_rows:] = float("-inf")
        for name, graph in graphs.items():
            summed[name].copy_(residual.expand(LAYERS, rows, WIDTH))
            graph.replay()
        torch.cuda.synchronize()
        for layer in range(LAYERS):
            # The FFN's contribution to the valid rows.
            metrics = error_metrics(
                summed["reference"][layer, :valid_rows].float() - residual[:valid_rows].float(),
                summed["kernels"][layer, :valid_rows].float() - residual[:valid_rows].float())
            # Measured 1.7e-4: FP32 summation order, and a 1-ulp RMSNorm difference that
            # can move an E4M3 code (quant_ops README). A layout or scale error is O(1).
            assert metrics["rel_rms"] < 1e-3 and metrics["cosine_similarity"] > 0.999999, (
                valid_rows, layer, metrics)
            assert torch.equal(summed["kernels"][layer, valid_rows:], residual[valid_rows:])
