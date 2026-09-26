"""RTX 5090 Pi0.5 replay buckets on the GPU, at two layers so every backbone call
site runs (one layer stops after its QKV projection).

- One engine switches buckets between inferences: a short task runs the
  backbone graph of its bucket, a long one another's, and the short one again
  reproduces its first result bit for bit.
- At each bucket's last length, the bucketed engine computes what the same
  construction captured at the full bucket alone computes.
"""
from __future__ import annotations

from dataclasses import replace
import os

import pytest
import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.inference import build, build_runner, get_target
from flash_vla.models.pi05.session import set_task

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12
    or "PALIGEMMA_TOKENIZER" not in os.environ,
    reason="RTX 5090 class GPU (sm_120) and the PaliGemma tokenizer required")

TARGET = "rtx5090/pi05"
DEPTH = {"steps": 1, "layers": 2}
#: RoboDojo's shortest and longest instructions (docs/workloads.md): with the 14
#: state values they fall in the 64- and the 128-token bucket.
SHORT = "Fold the clothes neatly."
LONG = ("Lift the basket more than 8 cm, identify the target object on the conveyor according "
        "to the image on the board, pick it up, and place it into the basket.")


def test_buckets_switch_between_inferences_and_repeat_exactly() -> None:
    engine = build(TARGET, workload="robodojo", **DEPTH)
    inputs = engine.sample_inputs(0)
    results = []
    for task in (SHORT, LONG, SHORT):
        set_task(engine, task)
        actions = engine.forward(**inputs).clone()
        torch.cuda.synchronize()
        results.append((engine.bucket, engine.extent, actions))
    (short_bucket, short_extent, first), (long_bucket, long_extent, _), (again, _, repeated) = results
    assert (short_bucket, long_bucket, again) == (64, 128, 64)
    assert short_extent <= 64 < long_extent <= 128
    assert torch.equal(first, repeated)


@pytest.mark.parametrize("workload, extent", [("robodojo", 64), ("robodojo", 128),
                                              ("robodojo", 200), ("libero", 64)])
def test_each_bucket_computes_what_the_full_bucket_computes(workload: str, extent: int) -> None:
    full_only = replace(get_target(TARGET), replay_granularity=None)
    outputs = {}
    for name, target in (("buckets", get_target(TARGET)), ("full", full_only)):
        engine = build_runner(target, "shipped", workload=workload, device="cuda",
                              prompt_tokens=extent, **DEPTH)
        engine.forward(**engine.sample_inputs(0))
        torch.cuda.synchronize()
        rows = engine.shape["visual_tokens"] + extent
        outputs[name] = (engine.bucket, engine.buffers["actions"].float().clone(),
                         engine.buffers["prefix_k"][:, :rows].float().clone())
        del engine
    assert outputs["buckets"][0] == extent and outputs["full"][0] == 200
    assert all(bool(torch.isfinite(tensor).all()) for name in outputs for tensor in outputs[name][1:]), {
        name: [bool(torch.isfinite(tensor).all()) for tensor in outputs[name][1:]] for name in outputs}
    # Another row count runs another GEMM tile (or cuBLAS for CUTLASS): bf16
    # rounding differences, held to the shared shallow tolerance.
    limit = tolerances()["shallow"]
    for index in (1, 2):
        metrics = error_metrics(outputs["full"][index], outputs["buckets"][index])
        assert (metrics["rel_rms"] <= limit["rel_rms_max"]
                and metrics["cosine_similarity"] >= limit["cosine_min"]), metrics
