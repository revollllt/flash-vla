"""The masked FFN graph must preserve dense saved plans and reference routes."""
import json

import torch

from flash_vla.inference import declare

_UP = "llm_backbone_norm_gated_ffn"
_DOWN = "llm_backbone_ffn_down_residual"
_MASKED_UP = _UP + "_masked"
_MASKED_DOWN = _DOWN + "_masked"


def test_saved_dense_plan_keeps_both_cutlass_routes(tmp_path):
    dense = dict(declare("rtx5090/pi05").identity.plan)
    dense[_MASKED_UP] = dense[_MASKED_DOWN] = "cutlass-backbone"
    control = declare("rtx5090/pi05", dense)
    saved = dict(control.identity.plan)
    saved[_UP] = saved.pop(_MASKED_UP)
    saved[_DOWN] = saved.pop(_MASKED_DOWN)
    path = tmp_path / "control.json"
    path.write_text(json.dumps(saved))
    loaded = declare("rtx5090/pi05", str(path))
    assert loaded.identity.plan == control.identity.plan
    assert loaded.identity.plan[_MASKED_UP] == "cutlass-backbone"
    assert loaded.identity.plan[_MASKED_DOWN] == "cutlass-backbone"
    assert _UP not in loaded.plan and _DOWN not in loaded.plan


def test_candidate_changes_only_two_routes_and_accepts_legacy_keys():
    dense = dict(declare("rtx5090/pi05").identity.plan)
    dense[_MASKED_UP] = dense[_MASKED_DOWN] = "cutlass-backbone"
    control = declare("rtx5090/pi05", dense)
    plan = {name: backend for name, backend in control.identity.plan.items()
            if name not in (_MASKED_UP, _MASKED_DOWN)}
    plan[_UP] = plan[_DOWN] = "bucketed-backbone"
    candidate = declare("rtx5090/pi05", plan)
    changed = {name for name, backend in candidate.identity.plan.items()
               if backend != control.identity.plan[name]}
    assert changed == {_MASKED_UP, _MASKED_DOWN}
    assert all(candidate.identity.plan[name] == "bucketed-backbone" for name in changed)
    assert candidate.identity.shape == control.identity.shape


def test_explicit_masked_override_wins_over_saved_standard_key():
    plan = {_MASKED_UP: "bucketed-backbone", _UP: "cutlass-backbone"}
    runner = declare("rtx5090/pi05", plan)
    assert runner.identity.plan[_MASKED_UP] == "bucketed-backbone"
    assert runner.identity.plan[_MASKED_DOWN] == "torch"


def test_reference_and_graph_keep_explicit_runtime_mask():
    runner = declare("rtx5090/pi05", "reference")
    assert set(runner.identity.plan.values()) == {"torch"}
    graph = runner.graph
    for name in (_MASKED_UP, _MASKED_DOWN):
        nodes = [node for node in graph.nodes if node.call_site == name]
        assert len(nodes) == 17
        for node in nodes:
            mask = node.args[-1]
            assert mask.name == "mask_bias"
            assert mask.shape == (968,)
            assert mask.dtype == torch.bfloat16
            assert "mask_bias" in graph.reads_writes(node)[0]
    assert graph.buf("prefix_k").shape == (18, 968, 256)
    assert graph.buf("suffix_k").shape == (18, 50, 256)
