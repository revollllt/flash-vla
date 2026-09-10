from unittest.mock import patch
import pytest


def test_qualification_rejects_explicit_source_unbound_to_receipt(tmp_path):
    from lab.optimize import runner
    checkout = str(tmp_path / "recorded")
    record = dict(target="hardware/nvidia/h100/lingbot_vla", repository=checkout,
                  spec=dict(qualification_sources=dict(incumbent={}, candidate={},
                      incumbent_checkout=checkout, candidate_checkout=checkout),
                      options={"source_checkout": str(tmp_path / "unrecorded")}))
    with patch.object(runner.store, "changed_inputs", return_value=[]):
        with pytest.raises(ValueError, match="candidate source option differs"):
            runner._command(record, "qualify")
