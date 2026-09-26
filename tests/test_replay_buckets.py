"""Replay-time buckets (`runtime/replay.py`): which buckets a Target captures,
that every bucket's graph runs over the same buffers at its own rows, and that
plans saved with the masked backbone names still bind."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flash_vla.inference import declare
from flash_vla.runtime.replay import ReplayAxis, replay_buckets

PI05 = ReplayAxis(name="prompt_tokens", limit="prompt_len", offset="visual_tokens",
                  slot="prompt", stages=("llm_backbone",),
                  extent=lambda host_state, inputs: 0)
UP, DOWN, OUT = ("llm_backbone_norm_gated_ffn", "llm_backbone_ffn_down_residual",
                 "llm_backbone_out_proj_residual")


@pytest.mark.parametrize("visual_tokens, replay_range, granularity, expected", [
    (768, (43, 101), 64, (64, 128, 200)),    # robodojo: 811-869 rows -> 832, 896, 968
    (768, (43, 101), 128, (128, 200)),       # -> 896, 968
    (512, (6, 22), 64, (64, 200)),           # libero: 518-534 rows -> 576, 712
    (512, (6, 22), 128, (128, 200)),         # -> 640, 712
    (768, None, 64, (200,)),                 # no range: the full bucket alone
    (768, (43, 101), None, (200,)),          # no granularity: the full bucket alone
    (768, (190, 200), 64, (192, 200)),       # 958-968 rows: 960 still saves a tile of 968
])
def test_buckets_end_on_the_tile_boundaries_a_workload_reaches(
        visual_tokens: int, replay_range: tuple[int, int] | None, granularity: int | None,
        expected: tuple[int, ...]) -> None:
    shape = {"visual_tokens": visual_tokens, "prompt_len": 200}
    assert replay_buckets(PI05, shape, replay_range, granularity) == expected


@pytest.mark.parametrize("workload, buckets", [("robodojo", (64, 128, 200)), ("libero", (64, 200))])
def test_each_bucket_runs_its_rows_over_the_same_buffers(workload: str,
                                                          buckets: tuple[int, ...]) -> None:
    runner = declare("rtx5090/pi05", workload=workload)
    assert runner.replay_buckets == runner.identity.replay_buckets == buckets
    assert runner.replay_stages == {"llm_backbone"}
    assert list(runner.variants) == list(buckets)
    image_tokens = runner.shape["visual_tokens"]
    for bucket, graph in runner.variants.items():
        assert graph.buffers.keys() == runner.graph.buffers.keys()
        rows = {node.args[0].shape[0] for node in graph.nodes if node.call_site == UP}
        assert rows == {image_tokens + bucket}
    assert runner.variants[buckets[-1]] is runner.graph


def test_a_model_without_a_granularity_captures_the_full_bucket_alone() -> None:
    for target in ("h100/pi05", "rtx5090/pi0", "h100/lingbot_vla", "rtx5090/groot_n17"):
        runner = declare(target)
        assert len(runner.variants) == 1 and len(runner.replay_buckets) <= 1
        assert runner.identity.replay_buckets == () and not runner.replay_stages


def test_plans_saved_with_the_masked_backbone_names_still_bind(tmp_path: Path) -> None:
    control = declare("rtx5090/pi05")
    saved = {({UP: UP + "_masked", DOWN: DOWN + "_masked", OUT: OUT + "_masked"}).get(site, site):
             backend for site, backend in control.identity.plan.items()}
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(saved))
    assert declare("rtx5090/pi05", str(path)).identity.plan == control.identity.plan
