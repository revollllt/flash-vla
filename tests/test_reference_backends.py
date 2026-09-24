"""Backends that run the model reference itself, against the reference, on a GPU.

GR00T's backend binds the runner's weights to the reference's own modules and
only stages the tables capture needs, so its captured engine and the eager
bfloat16 reference agree bit for bit on the real checkpoint and fixture (the
machine's `FLASH_VLA_ASSETS`). Faithfulness to upstream is the separate
official oracle, `eval/groot_n17/parity.py`.
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
