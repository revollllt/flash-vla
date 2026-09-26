"""The bucketed BF16 backbone computes full-row CUTLASS's product on the valid
rows, replayed from one CUDA graph as the runner replays it, at the three-view
and the two-view prefix, while the mask switches between the two row buckets;
under the short bucket the rows past it are not computed."""
import pytest
import torch

from eval.metrics import error_metrics
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import bucketed_backbone, cutlass_backbone
from flash_vla.hardware.nvidia.rtx5090.pi05.backends.row_buckets import bucket_rows
from flash_vla.runtime.workspace import Scratch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="RTX 5090 class GPU (sm_120) required")

WIDTH, HIDDEN = 2048, 16384   # model width, FFN width
SITES = ("llm_backbone_norm_gated_ffn_masked", "llm_backbone_ffn_down_residual_masked")


def random_bf16(*shape: int, scale: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)


# Image tokens and prefix rows (three views, two views) and the valid rows each
# replay masks in: the full bucket, the short one, exactly the short one's rows,
# the full one again.
@pytest.mark.parametrize("visual_tokens, rows, valid_rows_cases",
                         [(768, 968, (903, 869, 896, 903)), (512, 712, (700, 534, 640, 700))])
def test_buckets_match_the_full_row_plan_on_valid_rows(visual_tokens: int, rows: int,
                                                       valid_rows_cases: tuple[int, ...]) -> None:
    x = random_bf16(rows, WIDTH, scale=3.0, seed=1)
    residual = random_bf16(rows, WIDTH, scale=1.0, seed=2)
    gate_w, up_w = (random_bf16(WIDTH, HIDDEN, scale=0.03, seed=3),
                    random_bf16(WIDTH, HIDDEN, scale=0.03, seed=4))
    down_w = random_bf16(HIDDEN, WIDTH, scale=0.01, seed=5)
    mask = torch.zeros(rows, device="cuda", dtype=torch.bfloat16)   # additive, 0 = valid
    shape = {"visual_tokens": visual_tokens, "prefix_len": rows}
    short_rows = bucket_rows(shape)[0]
    # Each backend's wrappers and buffers stay alive as long as its graph.
    graphs, wrappers, hidden, summed, x_norm = {}, {}, {}, {}, {}
    for name, backend in (("buckets", bucketed_backbone), ("full", cutlass_backbone)):
        wrappers[name] = backend.make_wrappers(Scratch(torch.device("cuda"), shape=shape),
                                               frozenset(SITES))
        hidden[name] = torch.zeros(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
        summed[name] = residual.clone()
        x_norm[name] = torch.zeros_like(x)

        def forward(name: str = name) -> None:
            gated, down = (wrappers[name][site] for site in SITES)
            gated(x, gate_w, up_w, hidden[name], x_norm[name], mask)
            down(hidden[name], down_w, summed[name], mask)

        # Eager warmup plans; the replay must use those plans.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
            graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name], stream=stream):
                forward()
    for valid_rows in valid_rows_cases:
        mask.fill_(0.0)
        mask[valid_rows:] = float("-inf")
        for name, graph in graphs.items():
            summed[name].copy_(residual)
            graph.replay()
        torch.cuda.synchronize()
        # The FFN's hidden and its contribution to the residual, on the valid rows.
        # Another row count splits K differently: FP32 summation order only.
        contribution = {name: summed[name][:valid_rows].float() - residual[:valid_rows].float()
                        for name in summed}
        for produced in ({name: hidden[name][:valid_rows].float() for name in hidden},
                         contribution):
            metrics = error_metrics(produced["full"], produced["buckets"])
            assert metrics["rel_rms"] < 1e-3 and metrics["cosine_similarity"] > 0.999999, (
                valid_rows, metrics)
        if valid_rows <= short_rows:
            assert torch.equal(summed["buckets"][short_rows:], residual[short_rows:])
            assert not hidden["buckets"][short_rows:].any()
        else:
            assert not torch.equal(summed["buckets"][short_rows:], residual[short_rows:])
