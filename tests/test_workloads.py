"""Every model declares its workloads, and docs/workloads.md records each one."""
from __future__ import annotations

from pathlib import Path
import re

import pytest

from flash_vla.inference import TARGETS, build, declare, get_target

REPO = Path(__file__).resolve().parent.parent


def test_every_declared_workload_has_its_column_in_the_workload_page() -> None:
    page = (REPO / "docs" / "workloads.md").read_text()
    # Each model's section runs from its heading to the next one.
    sections = dict(re.findall(r"^## (.+?)\n(.*?)(?=^## |\Z)", page, flags=re.M | re.S))
    headings = {"pi0": "Pi0", "pi05": "Pi0.5", "groot-n17": "GR00T N1.7", "lingbot-vla": "LingBot-VLA"}
    models = {model.name: model for model in (get_target(name).model for name in TARGETS)}
    assert set(models) == set(headings)
    missing = [(name, workload.name) for name, model in models.items()
               for workload in model.workloads
               if f"`{workload.name}`" not in sections[headings[name]]]
    assert not missing


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_a_workload_fixes_its_options_and_labels_the_runner(target: str) -> None:
    runner = declare(target)
    assert runner.workload == runner.target.workloads[0]
    options = runner.target.model.workload(runner.workload).options
    fixed = next(iter(options), None)
    if fixed is None:
        return
    with pytest.raises(ValueError, match="fixes"):
        declare(target, **{fixed: options[fixed]})


def test_a_target_declares_every_model_workload_but_builds_only_its_own() -> None:
    assert declare("rtx5090/groot_n17", workload="robodojo").shape["views"] == 3
    assert declare("thor/groot_n17", workload="robodojo").shape["visual_tokens"] == 264
    with pytest.raises(ValueError, match="runs the workloads"):
        build("h100/pi05", workload="libero", device="cpu")
