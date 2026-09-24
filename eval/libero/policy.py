"""Pi0.5 LIBERO preprocessing and inference, following OpenPI's pi05_libero.

The LIBERO checkpoint uses a task-only prompt and chunk 10. Its absent right
wrist view is masked upstream and omitted from the Flash-VLA graph.
"""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch

from flash_vla.models.pi05.session import set_task
from flash_vla.models.pi05.tokenize import TaskTokenizer


def unnormalize_actions(actions: np.ndarray, stats: dict) -> np.ndarray:
    """OpenPI quantile inverse, including its epsilon and seven action outputs."""
    low = np.asarray(stats["q01"])
    high = np.asarray(stats["q99"])
    dim = len(low)
    result = (actions[..., :dim] + 1.0) / 2.0 * (high - low + 1e-6) + low
    return result[..., :7]


class Pi05LiberoPolicy:
    """CPU uint8 RGB views and eight state values -> CPU action chunk [10,7]."""

    def __init__(self, checkpoint: str, tokenizer: str, *, engine: str,
                 target: str = "rtx5090/pi05", plan: str = "shipped", steps: int = 10,
                 exact_rope: bool = False):
        from flash_vla.models.pi05.openpi import converted_checkpoint, restore_rope_precision

        directory = Path(checkpoint)
        config = json.loads((directory / "config.json").read_text())
        if config["discrete_state_input"] or config["action_horizon"] != 10 or not config["pi05"]:
            raise ValueError("This evaluation requires OpenPI pi05_libero: pi05=True, chunk=10, discrete_state_input=False")
        self.stats = json.loads((directory / "assets/physical-intelligence/libero/norm_stats.json").read_text())["norm_stats"]
        self.engine = engine
        self.steps = steps
        self.metadata = dict(engine=engine, checkpoint=str(directory), config=config,
                             steps=steps, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                             exact_rope=exact_rope if engine == "official" else None)
        if engine == "flashvla":
            from flash_vla.inference import get_target
            from flash_vla.models.pi05.weights import fold
            from flash_vla.provenance import git_revision
            from flash_vla.runtime import ModelRunner

            weights = fold(converted_checkpoint(directory), steps=steps)
            self.runner = ModelRunner(get_target(target), weights, engine_revision=git_revision(),
                                      plan=plan, num_views=2, chunk_size=10, steps=steps,
                                      prompt_len=config["max_token_len"], discrete_state=False,
                                      prompt="pick up the object",
                                      assets={"tokenizer": Path(tokenizer)})
            self.metadata.update(target=target, plan=plan, views=2)
        else:
            from eval.pi05 import official
            from safetensors.torch import load_model

            # The official model takes tokens; the Flash-VLA runner tokenizes itself.
            self.tokenizer = TaskTokenizer(tokenizer, config["max_token_len"])
            model_config = official.official_attr("Pi0Config")(
                pi05=True, action_horizon=10, discrete_state_input=False)
            self.model = official.official_attr("PI0Pytorch")(model_config).eval()
            load_model(self.model, str(directory / "model.safetensors"), strict=True, device="cpu")
            self.model.to("cuda")
            if exact_rope:
                restore_rope_precision(self.model)
            self.metadata.update(official=official.module_provenance(official.model_module()),
                                 execution="eager", views=3, image_mask=[True, True, False])

    @torch.inference_mode()
    def infer(self, obs: dict, noise: np.ndarray) -> dict:
        """Inputs are already resized to 224x224 by the official LIBERO image transform."""
        prompt = str(obs["prompt"])
        images = np.stack([obs["image"], obs["wrist_image"]]).astype(np.float32) / 127.5 - 1.0
        images_gpu = torch.from_numpy(images).to("cuda", torch.bfloat16)
        noise_gpu = torch.from_numpy(noise).to("cuda", torch.bfloat16)
        state = np.asarray(obs["state"], dtype=np.float32)
        stats = self.stats["state"]
        low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        state = np.pad(2 * (state - low) / (high - low + 1e-6) - 1, (0, 32 - len(state)))
        if self.engine == "flashvla":
            set_task(self.runner, prompt)
            normalized = self.runner.forward(images=images_gpu, state=state, noise=noise_gpu)
        else:
            self.tokenizer.set_task(prompt)
            keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            views = [images_gpu[0], images_gpu[1], torch.full_like(images_gpu[0], -1)]
            tokens, mask = self.tokenizer.encode(state)
            observation = SimpleNamespace(
                images={k: v.permute(2, 0, 1).unsqueeze(0) for k, v in zip(keys, views)},
                image_masks={k: torch.tensor([i < 2], device="cuda") for i, k in enumerate(keys)},
                state=torch.from_numpy(state).float().to("cuda").unsqueeze(0),
                tokenized_prompt=torch.from_numpy(tokens.astype(np.int64)).to("cuda").unsqueeze(0),
                tokenized_prompt_mask=torch.from_numpy(mask).to("cuda").unsqueeze(0),
                token_ar_mask=None, token_loss_mask=None)
            normalized = type(self.model).sample_actions(
                self.model, "cuda", observation, noise=noise_gpu.float().unsqueeze(0),
                num_steps=self.steps)[0]
        normalized = normalized.float().cpu().numpy()
        if not np.isfinite(normalized).all():
            raise FloatingPointError("Policy returned non-finite actions")
        return {"actions": unnormalize_actions(normalized, self.stats["actions"]),
                "normalized_actions": normalized}
