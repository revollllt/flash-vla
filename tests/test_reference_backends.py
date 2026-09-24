"""Backends that run the model reference itself, against the reference, on a GPU.

GR00T's plans and LingBot's reference plan bind the runner's weights to the
reference's own modules and only stage the tables capture needs, so the
captured engine and the eager bfloat16 reference agree bit for bit: GR00T on
the real checkpoint and fixture (the machine's `FLASH_VLA_ASSETS`), LingBot on
its synthetic construction, which needs no asset. Faithfulness to upstream is
separate: GR00T's official oracle (`eval/groot_n17/parity.py`), and for
LingBot, which has no local checkpoint or upstream run, the reference's
line-by-line translation.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from flash_vla.models.groot_n17.weights import CHECKPOINT_ID as GROOT_CHECKPOINT


def configured(asset: str) -> bool:
    path = os.environ.get("FLASH_VLA_ASSETS")
    return path is not None and asset in json.loads(Path(path).read_text())


@pytest.mark.skipif(not torch.cuda.is_available() or not configured(GROOT_CHECKPOINT),
                    reason="needs CUDA and the GR00T checkpoint in FLASH_VLA_ASSETS")
def test_groot_engine_runs_the_bfloat16_reference_bit_for_bit() -> None:
    from eval.model_reference import run

    report = run("rtx5090/groot_n17", steps=None, layers=None, reference_precision="bfloat16")
    assert {stage: metrics["rel_rms"] for stage, metrics in report["stages"].items()} == {
        stage: 0.0 for stage in report["stages"]}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lingbot_reference_plan_runs_the_bfloat16_reference_bit_for_bit() -> None:
    from eval.model_reference import run

    report = run("h100/lingbot_vla", "reference", steps=None, layers=None,
                 reference_precision="bfloat16", synthetic=True)
    assert {stage: metrics["rel_rms"] for stage, metrics in report["stages"].items()} == {
        stage: 0.0 for stage in report["stages"]}
