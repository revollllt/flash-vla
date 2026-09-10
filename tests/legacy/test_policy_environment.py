import json
import pytest


@pytest.mark.parametrize("configured", [False, True])
def test_reference_locations_come_from_machine_environment(tmp_path, configured):
    import os
    import subprocess
    import sys

    selected = {
        "OPENPI_PYTHON": str(tmp_path / "openpi runtime/python"),
        "LINGBOT_PYTHON": str(tmp_path / "lingbot runtime/python"),
        "OPENPI_PI0_CHECKPOINT": str(tmp_path / "pi0 weights"),
        "OPENPI_PI0_MODEL_REVISION": "registered-weights/v1",
    }
    env = {key: value for key, value in os.environ.items() if key not in selected}
    if configured:
        env.update(selected)
    probe = """
import json
from lab.optimize import policy as acceptance
print(json.dumps({
    "openpi": acceptance.for_target("hardware/nvidia/h100/pi05")["baseline_python"],
    "lingbot": acceptance.for_target("hardware/nvidia/h100/lingbot_vla")["baseline_python"],
    "checkpoint": acceptance.OPENPI_PI0_CHECKPOINT,
    "checkpoint_id": acceptance.OPENPI_PI0_MODEL_REVISION,
}))
"""
    process = subprocess.run([sys.executable, "-c", probe], env=env,
                             check=True, text=True, capture_output=True)
    actual = json.loads(process.stdout)
    assert actual == (dict(zip(("openpi", "lingbot", "checkpoint", "checkpoint_id"),
                               selected.values())) if configured else
                      dict.fromkeys(("openpi", "lingbot", "checkpoint", "checkpoint_id")))
