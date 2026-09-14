"""Compare short896 -> long903 -> short896 using one captured RTX5090 engine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from eval.metrics import error_metrics
from eval.pi05.reference import to_pair_layout
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.target import set_task
from flash_vla.inference import build


def _within(metrics, pair):
    return (metrics["cosine_similarity"] > pair["cosine_min"]
            and metrics["rel_rms"] < pair["rel_rms_max"])


def _compare(expected, keys, values, actions, n_valid, limits):
    """Use parity.compare's full-depth KV/action formulas and tolerances."""
    rows = [
        {"layer": layer,
         "k": error_metrics(to_pair_layout(expected["prefix_k"][layer, :n_valid]),
                            keys[layer, :n_valid]),
         "v": error_metrics(expected["prefix_v"][layer, :n_valid],
                            values[layer, :n_valid])}
        for layer in range(keys.shape[0])
    ]
    cosines = [min(row["k"]["cosine_similarity"], row["v"]["cosine_similarity"])
               for row in rows]
    worst_step = max((a - b for a, b in zip(cosines, cosines[1:])), default=0.0)
    endpoints = [
        min(rows[index]["k"], rows[index]["v"], key=lambda m: m["cosine_similarity"])
        for index in (0, -1)
    ]
    padded_finite = bool(torch.isfinite(keys[:, n_valid:]).all()
                         and torch.isfinite(values[:, n_valid:]).all())
    actions_finite = bool(torch.isfinite(actions).all())
    action_metrics = error_metrics(expected["actions"], actions)
    return {
        "per_layer": rows,
        "cosine": {"layer0": cosines[0], "deepest": cosines[-1], "worst_step": worst_step},
        "padded_rows_finite": padded_finite,
        "actions_finite": actions_finite,
        "actions": action_metrics,
        "passed": (padded_finite and actions_finite
                   and _within(endpoints[0], limits["layer0"])
                   and _within(endpoints[1], limits["deepest"])
                   and worst_step < limits["max_cosine_step"]
                   and _within(action_metrics, limits["deepest"])),
    }


def run(args):
    # Load the two existing fixtures on CPU; never regenerate seeded tensors.
    cases = {}
    for name, directory in (("short896", args.short_oracle), ("long903", args.long_oracle)):
        metadata = json.loads((directory / "official-eager.json").read_text())
        inputs = load_file(str(directory / "fixture.safetensors"), device="cpu")
        cases[name] = {
            "directory": str(directory),
            "metadata": metadata,
            "inputs": inputs,
            "expected": load_file(str(directory / "official-eager.safetensors"), device="cpu"),
        }
    short = cases["short896"]["metadata"]
    engine = build(
        "rtx5090/pi05", args.plan, converted_checkpoint=str(args.checkpoint),
        checkpoint_id=args.checkpoint_id,
        checkpoint_digest=args.checkpoint_digest or args.checkpoint_id,
        chunk_size=short["fixture"]["chunk"], steps=short["fixture"]["steps"],
        layers=short["layers_captured"], device="cuda",
        prompt=short["fixture"]["prompt"], tokenizer_path=args.tokenizer_path,
        seed=short["fixture"]["seed"],
    )
    program, steps = engine.graphs, engine.program
    graphs = dict(program.graphs)
    cuda_graphs = {name: graph._graph for name, graph in graphs.items()}
    for case in cases.values():
        inputs = case["inputs"]
        # Exactly the device/dtype conversion used by parity.compare.
        case["device_inputs"] = {
            "images": inputs["images"].to("cuda").bfloat16(),
            "state": inputs["state"].to("cuda"),
            "noise": inputs["noise"].to("cuda").bfloat16(),
        }
    limits = tolerances(engine.identity.precision)
    report = {
        "stage": "bucket-switch-official-parity",
        "identity": engine.identity.as_dict(),
        "plan": engine.plan,
        "tolerance": limits,
        "captured_program_id": id(program),
        "forward_steps_id": id(steps),
        "sequence": [],
    }
    for name in ("short896", "long903", "short896"):
        case = cases[name]
        metadata = case["metadata"]
        n_valid = metadata["fixture"]["n_valid_prefix"]
        set_task(engine, metadata["fixture"]["prompt"])
        actions = engine.forward(**case["device_inputs"]).float().cpu()
        torch.cuda.synchronize()
        prefix_len = engine.derived["prefix_len"]
        keys = engine.buffers["prefix_k"][:, :prefix_len].detach().float().cpu()
        values = engine.buffers["prefix_v"][:, :prefix_len].detach().float().cpu()
        result = _compare(case["expected"], keys, values, actions, n_valid, limits)
        same_graphs = {
            stage: {
                "same_stream_graph": engine.graphs.graphs[stage] is graph,
                "same_cuda_graph": engine.graphs.graphs[stage]._graph is cuda_graphs[stage],
                "stream_graph_id": id(engine.graphs.graphs[stage]),
                "cuda_graph_id": id(engine.graphs.graphs[stage]._graph),
            }
            for stage, graph in graphs.items()
        }
        staged_tokens = engine.buffers["prompt_ids"].detach().cpu().to(torch.int64)
        result.update({
            "case": name, "oracle": case["directory"], "fixture": metadata["fixture"],
            "expected_n_valid": n_valid, "actual_n_valid": engine.host_state.n_valid,
            "tokens_equal": torch.equal(staged_tokens, case["inputs"]["prompt_tokens"].long()),
            "same_program": engine.graphs is program,
            "same_forward_steps": engine.program is steps,
            "same_segment_names": tuple(engine.graphs.graphs) == tuple(graphs),
            "graphs": same_graphs,
        })
        result["passed"] = bool(
            result["passed"] and result["tokens_equal"]
            and result["actual_n_valid"] == n_valid
            and n_valid == {"short896": 896, "long903": 903}[name]
            and result["same_program"] and result["same_forward_steps"]
            and result["same_segment_names"]
            and all(row["same_stream_graph"] and row["same_cuda_graph"]
                    for row in same_graphs.values()))
        report["sequence"].append(result)
    report["passed"] = all(result["passed"] for result in report["sequence"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-oracle", type=Path, required=True)
    parser.add_argument("--long-oracle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-digest")
    parser.add_argument("--plan", default="shipped")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
