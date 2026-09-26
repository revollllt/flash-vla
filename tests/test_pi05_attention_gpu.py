"""RTX 5090 Pi0.5 attentions: the expert one at the exact valid length, and the
vision one at both workloads' views.

The expert attention reads the valid prefix rows and the chunk's, which sit
past the whole prefix in the cache. It is captured once in a CUDA graph, then
given valid lengths across the prefix, its key-block boundaries included,
through its replay hook (`Scratch.on_replay`), and replayed. Every replay must
match a float32 softmax over the valid keys: a dropped or repeated key block
shows up as an error, and a length the graph did not pick up reproduces the
previous length's output. It is checked at RoboDojo's and LIBERO's chunk
geometry, where its query tiles differ. The vision attention reads its
heads as 64 columns and a masked tail; it is checked at two and three views.

The runner path (the hook run as buckets alternate, the expert replaying after
any bucket) is gated end to end by `test_pi05_replay_switch_gpu.py` and
`test_model_reference_gpu.py`, which route through these backends.
"""
from __future__ import annotations

import pytest
import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import (
    split_kv_attention, triton_vision_attention)
from flash_vla.models.pi05.spec import (
    DECODER_HEADS, HEAD_DIM, VISION_HEAD_DIM, VISION_HEADS, VISION_TOKENS)
from flash_vla.runtime.workspace import Scratch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="RTX 5090 class GPU (sm_120) required")

def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Float32 softmax attention: q (rows, heads, 256) over k, v (keys, 256)."""
    logits = torch.einsum("rhd,kd->rhk", q.float(), k.float()) * HEAD_DIM ** -0.5
    return torch.einsum("rhk,kd->rhd", torch.softmax(logits, -1), v.float())


def assert_close(expected: torch.Tensor, observed: torch.Tensor) -> None:
    metrics = error_metrics(expected, observed.float())
    limit = tolerances()["shallow"]
    assert (metrics["rel_rms"] <= limit["rel_rms_max"]
            and metrics["cosine_similarity"] >= limit["cosine_min"]), metrics


@pytest.mark.parametrize("prefix_rows, chunk", [(968, 50), (712, 10)])
@pytest.mark.parametrize("aliased", [False, True], ids=["out", "out-is-q"])
def test_expert_attention_reads_each_valid_prefix_and_the_chunk(prefix_rows: int, chunk: int,
                                                                aliased: bool) -> None:
    """The graph passes the queries' buffer as the output (`models/pi05/graph.py`),
    so each case also runs with `out` aliasing `q`, restored before each replay."""
    generator = torch.Generator(device="cuda").manual_seed(1)
    cache_rows = prefix_rows + chunk
    queries, k, v = (torch.randn(*shape, generator=generator, device="cuda").to(torch.bfloat16)
                     for shape in ((chunk * DECODER_HEADS, HEAD_DIM), (cache_rows, HEAD_DIM),
                                   (cache_rows, HEAD_DIM)))
    q = queries.clone()
    out = q if aliased else torch.zeros_like(q)
    mask = torch.zeros(cache_rows, dtype=torch.bfloat16, device="cuda")
    scratch = Scratch(torch.device("cuda"))
    attention = split_kv_attention.make_wrappers(scratch)["action_expert_attention"]
    (write_valid_rows,) = scratch.replay_hooks
    write_valid_rows(prefix_rows, prefix_rows)
    attention(q, k, v, mask, out, prefix_rows)                 # warmup: compile
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attention(q, k, v, mask, out, prefix_rows)
    for valid in (1, 31, 32, 33) + tuple(prefix_rows - rows for rows in (129, 64, 33, 1, 0)):
        q.copy_(queries)
        write_valid_rows(valid, prefix_rows)
        graph.replay()
        torch.cuda.synchronize()
        keys = torch.cat((torch.arange(valid), torch.arange(prefix_rows, cache_rows))).cuda()
        assert_close(reference(queries.view(chunk, DECODER_HEADS, HEAD_DIM), k[keys], v[keys]),
                     out.view(chunk, DECODER_HEADS, HEAD_DIM))


@pytest.mark.parametrize("views", [2, 3])
def test_vision_attention_matches_float_softmax(views: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(2)
    width = VISION_HEADS * VISION_HEAD_DIM
    qkv = torch.randn(views, VISION_TOKENS, 3 * width, generator=generator,
                      device="cuda").to(torch.bfloat16)
    out = torch.zeros(views, VISION_TOKENS, width, dtype=torch.bfloat16, device="cuda")
    triton_vision_attention.make_wrappers(Scratch(torch.device("cuda")))[
        "vision_encoder_attention"](qkv, out)
    heads = qkv.float().view(views, VISION_TOKENS, 3, VISION_HEADS, VISION_HEAD_DIM).permute(0, 2, 3, 1, 4)
    logits = heads[:, 0] @ heads[:, 1].transpose(-1, -2) * VISION_HEAD_DIM ** -0.5
    expected = (torch.softmax(logits, -1) @ heads[:, 2]).transpose(1, 2).reshape(views, VISION_TOKENS, width)
    assert_close(expected, out)
