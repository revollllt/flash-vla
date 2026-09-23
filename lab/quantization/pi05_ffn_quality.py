"""Quality of fake-quantized Pi0.5 backbone FFN recipes on LIBERO observations.

collect  runs the BF16 shipped policy in LIBERO through eval.libero's episode
         loop and keeps every k-th observation it infers on, with its noise.
compare  replays those observations through runners whose backbone FFN is
         fake-quantized (rtx5090/pi05/backends/fake_quant_ffn.py) and reports
         the error of their normalized action chunks against the BF16 runner's.
         Recipes: all MXFP8, each group alone in MXFP8, all NVFP4, and the
         sensitivity scan (all MXFP8 with one layer's gate/up or down in NVFP4);
         `mxfp8_kernels` is the shipped plan under the mxfp8-llm-ffn recipe,
         the kernels themselves. Fake-quant runners are built under that recipe
         so its reference backend may serve the FFN; the per-layer formats of
         the runner asset `quantization_recipe` override it, which is what the
         scan explores.

For scale: BF16 Flash-VLA against the official OpenPI model on one observation
differs by relative L2 0.0048 on normalized actions
(artifacts/libero/parity-summary.json), with equal LIBERO success.

Run in the LIBERO environment of artifacts/libero/run_libero.sh (its exports),
from the repository root, under the GPU lock.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from eval.libero.__main__ import evaluate
from eval.libero.policy import LiberoTokenizer, Pi05LiberoPolicy
from eval.metrics import error_metrics
from flash_vla.hardware.nvidia.rtx5090.pi05.target import TARGET
from flash_vla.inference import declare
from flash_vla.models.pi05.openpi import converted_checkpoint
from flash_vla.models.pi05.spec import ENCODER_LAYERS
from flash_vla.models.pi05.weights import fold
from flash_vla.runtime import ModelRunner

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
FFN_CALL_SITES = ("llm_backbone_norm_gated_ffn", "llm_backbone_ffn_down_residual")
Recipe = dict[str, list[str]]    # {"gate_up": [fmt per layer], "down": [fmt per layer]}
QUANTIZATION = "mxfp8-llm-ffn"


@dataclass(frozen=True)
class Variant:
    """A runner to compare: its plan and quantization, and the per-layer formats
    the fake-quant backend runs (None: no fake quantization)."""
    plan: str | dict[str, str]
    quantization: str
    recipe: Recipe | None


class RecordingPolicy:
    """Delegates to the BF16 policy and keeps every `every`-th call's inputs."""

    def __init__(self, policy: Pi05LiberoPolicy, every: int) -> None:
        self.policy = policy
        self.metadata = policy.metadata
        self.every = every
        self.calls = 0
        self.kept: list[tuple[dict, np.ndarray]] = []

    def infer(self, obs: dict, noise: np.ndarray) -> dict:
        keep = self.calls % self.every == 0
        self.kept += [({key: np.copy(value) if isinstance(value, np.ndarray) else value
                        for key, value in obs.items()}, np.copy(noise))] if keep else []
        self.calls += 1
        return self.policy.infer(obs, noise)


def variants() -> dict[str, Variant]:
    """name -> variant; "bf16", the BF16 shipped plan, is what the others are compared with."""
    fake_quant_plan = {**TARGET.plan, **{site: "fake-quant-mxfp8" for site in FFN_CALL_SITES}}
    fake_quant = lambda gate_up, down: Variant(fake_quant_plan, QUANTIZATION, {
        "gate_up": [gate_up] * ENCODER_LAYERS, "down": [down] * ENCODER_LAYERS})
    named = {
        "bf16": Variant("shipped", "bf16", None),
        "mxfp8_kernels": Variant("shipped", QUANTIZATION, None),
        "mxfp8": fake_quant("mxfp8", "mxfp8"), "mxfp8_gate_up_only": fake_quant("mxfp8", "bf16"),
        "mxfp8_down_only": fake_quant("bf16", "mxfp8"), "nvfp4": fake_quant("nvfp4", "nvfp4")}
    for group in ("gate_up", "down"):
        for layer in range(ENCODER_LAYERS):
            variant = fake_quant("mxfp8", "mxfp8")
            variant.recipe[group][layer] = "nvfp4"
            named[f"nvfp4_{group}_L{layer:02d}"] = variant
    return named


def collect(args: argparse.Namespace) -> None:
    recorder = RecordingPolicy(Pi05LiberoPolicy(args.checkpoint, args.tokenizer, engine="flashvla"),
                               args.every)
    for suite in args.suites:
        evaluate(argparse.Namespace(suite=suite, task_ids=list(range(args.tasks)), trials=args.trials,
                                    seed=7, replan_steps=5, out=args.reports / f"{suite}.json"),
                 recorder)
    observations = [obs for obs, _ in recorder.kept]
    np.savez_compressed(args.out,
                        image=np.stack([obs["image"] for obs in observations]),
                        wrist_image=np.stack([obs["wrist_image"] for obs in observations]),
                        state=np.stack([obs["state"] for obs in observations]),
                        prompt=np.array([obs["prompt"] for obs in observations]),
                        noise=np.stack([noise for _, noise in recorder.kept]))
    print(f"kept {len(observations)} of {recorder.calls} observations -> {args.out}")


def normalized_actions(runner: ModelRunner, tokenizer: LiberoTokenizer, stats: dict,
                       dataset: np.lib.npyio.NpzFile) -> torch.Tensor:
    """float32 [observations, 10, 32]: the runner's normalized action chunks,
    preprocessed as eval/libero/policy.py does."""
    low, high = np.asarray(stats["state"]["q01"]), np.asarray(stats["state"]["q99"])
    chunks = []
    for index in range(len(dataset["prompt"])):
        tokenizer.set_task(str(dataset["prompt"][index]))
        images = np.stack([dataset["image"][index], dataset["wrist_image"][index]])
        images = torch.from_numpy(images.astype(np.float32) / 127.5 - 1.0).to("cuda", torch.bfloat16)
        state = np.pad(2 * (dataset["state"][index] - low) / (high - low + 1e-6) - 1,
                       (0, 32 - dataset["state"].shape[1]))
        noise = torch.from_numpy(dataset["noise"][index]).to("cuda", torch.bfloat16)
        chunks.append(runner.forward(images=images, state=state, noise=noise).float().cpu().clone())
    return torch.stack(chunks)


def compare(args: argparse.Namespace) -> None:
    directory = Path(args.checkpoint)
    config = json.loads((directory / "config.json").read_text())
    stats = json.loads((directory / "assets/physical-intelligence/libero/norm_stats.json")
                       .read_text())["norm_stats"]
    dataset = np.load(args.dataset)
    weights = fold(converted_checkpoint(directory), steps=10)
    tokenizer = LiberoTokenizer(args.tokenizer, config["max_token_len"])
    args.out.mkdir(parents=True, exist_ok=True)
    selected = {name: variant for name, variant in variants().items()
                if name == "bf16" or any(pattern in name for pattern in args.only.split(","))}
    reference: torch.Tensor | None = None
    rows = []
    for name, variant in selected.items():
        recipe_path = args.out / "recipes" / f"{name}.json"
        recipe_path.parent.mkdir(exist_ok=True)
        recipe_path.write_text(json.dumps(variant.recipe) + "\n")
        runner = ModelRunner(declare("rtx5090/pi05").target, weights, checkpoint_id="openpi/pi05_libero",
                             plan=variant.plan, quantization=variant.quantization,
                             num_views=2, chunk_size=10, steps=10, prompt_len=config["max_token_len"],
                             tokenizer=tokenizer, prompt="pick up the object",
                             assets={} if variant.recipe is None
                             else {"quantization_recipe": recipe_path})
        actions = normalized_actions(runner, tokenizer, stats, dataset)
        del runner
        torch.cuda.empty_cache()
        reference = actions if name == "bf16" else reference
        per_observation = [error_metrics(reference[index, :, :7], actions[index, :, :7])
                           for index in range(len(actions))]
        row = dict(recipe=name, observations=len(actions),
                   rel_rms_mean=statistics.fmean(m["rel_rms"] for m in per_observation),
                   rel_rms_p90=float(np.quantile([m["rel_rms"] for m in per_observation], 0.9)),
                   rel_rms_max=max(m["rel_rms"] for m in per_observation),
                   cosine_min=min(m["cosine_similarity"] for m in per_observation),
                   max_abs=max(m["max_abs"] for m in per_observation),
                   pooled=error_metrics(reference[:, :, :7], actions[:, :, :7]))
        rows.append(row)
        print(json.dumps({key: round(value, 6) if isinstance(value, float) else value
                          for key, value in row.items() if key != "pooled"}), flush=True)
        (args.out / "compare.json").write_text(json.dumps(rows, indent=1) + "\n")
        (args.out / "compare.md").write_text(render_summary(rows))


def render_summary(rows: list[dict]) -> str:
    """Uniform recipes, then the scan ranked by how much one NVFP4 group raises the
    mean relative RMS error over all-MXFP8."""
    by_name = {row["recipe"]: row for row in rows}
    lines = ["| recipe | rel RMS mean | p90 | max | min cosine | max abs |",
             "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['recipe']} | {row['rel_rms_mean']:.4f} | {row['rel_rms_p90']:.4f} "
              f"| {row['rel_rms_max']:.4f} | {row['cosine_min']:.5f} | {row['max_abs']:.4f} |"
              for row in rows if not row["recipe"].startswith("nvfp4_")]
    scan = sorted((row for row in rows if row["recipe"].startswith("nvfp4_")),
                  key=lambda row: row["rel_rms_mean"], reverse=True)
    baseline = by_name["mxfp8"]["rel_rms_mean"] if "mxfp8" in by_name else 0.0
    lines += ["", "Sensitivity: all MXFP8 with one group in NVFP4, most sensitive first.", "",
              "| NVFP4 group | rel RMS mean | increase over all-MXFP8 | max |",
              "|---|---:|---:|---:|"]
    lines += [f"| {row['recipe'][6:]} | {row['rel_rms_mean']:.4f} "
              f"| {row['rel_rms_mean'] - baseline:+.4f} | {row['rel_rms_max']:.4f} |" for row in scan]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--checkpoint", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--suites", nargs="+", default=list(SUITES))
    collect_parser.add_argument("--tasks", type=int, default=10, help="task ids 0..tasks-1 per suite")
    collect_parser.add_argument("--trials", type=int, default=1)
    collect_parser.add_argument("--every", type=int, default=3, help="keep every k-th inference")
    collect_parser.add_argument("--reports", type=Path, required=True)
    collect_parser.add_argument("--out", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--checkpoint", required=True)
    compare_parser.add_argument("--tokenizer", required=True)
    compare_parser.add_argument("--dataset", type=Path, required=True)
    compare_parser.add_argument("--only", default="", help="comma-separated substrings of recipe names")
    compare_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"collect": collect, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
