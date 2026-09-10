"""Official OpenPI Pi0 execution used as a numerical reference."""
from __future__ import annotations
from pathlib import Path
import torch
from flash_vla.models.pi0.openpi import IMAGE_KEYS

def load_model(checkpoint: str | Path, device: str | torch.device = "cuda"):
    """Load the official OpenPI PyTorch Pi0 model without torch.compile."""
    try:
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from safetensors.torch import load_model as load_safetensors_model
    except ImportError as error:
        raise RuntimeError(
            "OpenPI's PyTorch dependencies are required; install OpenPI using "
            "its official PyTorch setup instructions."
        ) from error

    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = PI0Pytorch(Pi0Config(pytorch_compile_mode=None))
    load_safetensors_model(model, str(checkpoint), strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def sample_actions(model, images: torch.Tensor, state: torch.Tensor, noise: torch.Tensor):
    """Run OpenPI on three valid images, an empty prompt, and explicit noise."""
    from openpi.models.model import Observation

    if images.shape != (3, 224, 224, 3):
        raise ValueError(f"expected images [3, 224, 224, 3], got {tuple(images.shape)}")
    if state.shape != (32,):
        raise ValueError(f"expected state [32], got {tuple(state.shape)}")
    if noise.shape != (50, 32):
        raise ValueError(f"expected noise [50, 32], got {tuple(noise.shape)}")

    device = images.device
    observation = Observation(
        images={
            key: images[index].permute(2, 0, 1).unsqueeze(0)
            for index, key in enumerate(IMAGE_KEYS)
        },
        image_masks={key: torch.ones(1, dtype=torch.bool, device=device) for key in IMAGE_KEYS},
        state=state.unsqueeze(0),
        tokenized_prompt=torch.empty((1, 0), dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.empty((1, 0), dtype=torch.bool, device=device),
    )
    return model.sample_actions(
        device, observation, noise=noise.unsqueeze(0), num_steps=10
    )[0]
