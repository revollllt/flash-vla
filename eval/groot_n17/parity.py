"""Compare the project Target with the official GR00T forward on real observations.

Run 'official' and 'compare' in separate processes so model loading, GPU memory
and framework state do not leak between the oracle and project implementation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances


def processor_for(checkpoint, backbone):
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    processor = Gr00tN1d7Processor.from_pretrained(checkpoint, model_name=str(backbone))
    processor.eval()
    return processor


def decode(processor, actions, states):
    from gr00t.data.embodiment_tags import EmbodimentTag
    values = processor.decode_action(actions.float().cpu().numpy(), EmbodimentTag.LIBERO_PANDA,
                                      {name: value.numpy()[None] for name, value in states.items()})
    return {name: torch.from_numpy(value).float() for name, value in values.items()}


def official(args):
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    import gr00t

    config = Gr00tN1d7Config.from_pretrained(args.checkpoint)
    config.model_name = str(args.backbone)
    config.use_flash_attention = False
    model = Gr00tN1d7.from_pretrained(args.checkpoint, config=config,
        transformers_loading_kwargs={"local_files_only": True, "attn_implementation": "sdpa"})
    model.eval().requires_grad_(False).to(device="cuda", dtype=torch.bfloat16)
    processor = processor_for(args.checkpoint, args.backbone)
    cases = []
    for index, path in enumerate(args.fixture):
        fixture = torch.load(path, map_location="cpu", weights_only=True)
        inputs = {name: value.to("cuda") for name, value in fixture["inputs"].items()}
        seed = args.seed + index
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn((1, 40, 132), dtype=torch.bfloat16, device="cuda", generator=generator)
        torch.manual_seed(seed)
        with torch.inference_mode():
            actions = model.get_action(dict(inputs))["action_pred"].clone()
        torch.cuda.synchronize()
        if not torch.equal(torch.cuda.get_rng_state(), generator.get_state()):
            raise RuntimeError("Official RNG consumption differs from the explicit noise fixture")
        inputs["noise"] = noise
        cases.append({"fixture": str(path), "seed": seed, "states": fixture["states"],
                      "inputs": {name: value.cpu() for name, value in inputs.items()},
                      "actions": actions.cpu(), "decoded": decode(processor, actions, fixture["states"])})
        print(f"Official observation {path.name}, seed {seed}: {tuple(actions.shape)}", flush=True)
    source = Path(gr00t.__file__).resolve().parent.parent
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cases": cases, "official_revision": revision,
                "checkpoint": str(args.checkpoint), "backbone": str(args.backbone),
                "attention": "pytorch-sdpa", "precision": "bf16",
                "versions": {name: importlib.metadata.version(name)
                             for name in ("torch", "transformers", "diffusers")}}, args.output)
    print(f"Saved official outputs: {args.output}", flush=True)


def compare(args):
    from flash_vla.inference import build
    oracle = torch.load(args.oracle, map_location="cpu", weights_only=True)
    processor = processor_for(oracle["checkpoint"], oracle["backbone"])
    runner = build("groot-n17", args.plan, asset_config=str(args.assets))
    limits = tolerances("bf16")["shallow"]

    def within(metrics):
        return metrics["rel_rms"] < limits["rel_rms_max"] and metrics["cosine_similarity"] > limits["cosine_min"]

    results = []
    for case in oracle["cases"]:
        inputs = {inp.name: case["inputs"][inp.name].to("cuda") for inp in runner.target.INPUTS}
        got = runner.forward(**inputs).clone()
        repeat = runner.forward(**inputs).clone()
        torch.cuda.synchronize()
        metrics = error_metrics(case["actions"], got.cpu())
        decoded = decode(processor, got, case["states"])
        keys = list(case["decoded"])
        decoded_metrics = error_metrics(torch.cat([case["decoded"][k] for k in keys], dim=-1),
                                         torch.cat([decoded[k] for k in keys], dim=-1))
        runner.stage(**inputs)
        for step in runner.program:
            runner.run_eager(step.name)
        torch.cuda.synchronize()
        eager_metrics = error_metrics(got, runner.buffers["actions"])
        finite = bool(torch.isfinite(got).all())
        identical = torch.equal(got, repeat)
        result = {"fixture": case["fixture"], "seed": case["seed"], "actions": metrics,
                  "decoded_actions": decoded_metrics, "eager_vs_capture": eager_metrics,
                  "decoded_shapes": {k: list(v.shape) for k, v in decoded.items()},
                  "finite": finite, "repeat_identical": identical}
        result["passed"] = finite and identical and all(within(m) for m in (metrics, decoded_metrics, eager_metrics))
        results.append(result)
    report = {"official_revision": oracle["official_revision"], "versions": oracle["versions"],
              "plan": args.plan, "precision": "bf16", "shape": runner.shape,
              "tolerance": limits, "observations": results, "passed": all(r["passed"] for r in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    upstream = sub.add_parser("official")
    upstream.add_argument("--checkpoint", type=Path, required=True)
    upstream.add_argument("--backbone", type=Path, required=True)
    upstream.add_argument("--fixture", type=Path, action="append", required=True)
    upstream.add_argument("--seed", type=int, default=0)
    upstream.add_argument("--output", type=Path, required=True)
    candidate = sub.add_parser("compare")
    candidate.add_argument("--oracle", type=Path, required=True)
    candidate.add_argument("--assets", type=Path, required=True)
    candidate.add_argument("--plan", default="reference")
    candidate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "official":
        official(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
