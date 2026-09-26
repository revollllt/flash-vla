"""GR00T N1.7: the model's constants, without importing the model.

Values are the official `libero_10` checkpoint's (see README.md): each 256x256
view becomes 256 patches of width 1536 and 64 visual tokens; a 16-layer
Qwen3-VL backbone; a 32-block DiT action head denoising a 40x132 chunk in four
steps. The number of views and the text sequence belong to the workload
(`definition.GrootModel.workloads`).
"""
from __future__ import annotations

MODEL_REVISION = "groot-n17-libero-v1"
INFERENCE_SIGNATURE = "groot-n17-qwen3vl16-dit32-libero-bf16-v1"

IMAGE_SIZE = 256
#: One view's patch grid (frames, rows, columns), a row of the processor's `image_grid_thw`.
VIEW_GRID = (1, 16, 16)
#: Patches per view (a 16x16 grid), and their flattened width.
PATCHES_PER_VIEW = 256
PATCH_WIDTH = 1536
#: Visual tokens per view after the 2x2 patch merger.
VISUAL_TOKENS_PER_VIEW = 64

VISION_BLOCKS = 24
VISION_DIM = 1024
VISION_FFN = 4096
#: Vision blocks whose outputs join the first backbone layers (Qwen3-VL DeepStack),
#: one backbone layer each.
DEEPSTACK_BLOCKS = (5, 11, 17)
DEEPSTACK_LAYERS = len(DEEPSTACK_BLOCKS)

LAYERS = 16
BACKBONE_DIM = 2048
BACKBONE_FFN = 6144
KV_DIM = 1024

DIT_BLOCKS = 32
DIT_DIM = 1536
DIT_FFN = 6144
#: Width of the DiT's output and of the embodiment-conditioned state MLP and action decoder.
HEAD_HIDDEN = 1024
#: Sinusoidal channels of the DiT's timestep embedding.
TIMESTEP_CHANNELS = 256

STATE_DIM = 132
ACTION_DIM = 132
CHUNK = 40
STEPS = 4


__all__ = ["ACTION_DIM", "BACKBONE_DIM", "BACKBONE_FFN", "CHUNK", "DEEPSTACK_BLOCKS", "DEEPSTACK_LAYERS",
           "DIT_BLOCKS", "DIT_DIM", "DIT_FFN", "HEAD_HIDDEN", "IMAGE_SIZE",
           "INFERENCE_SIGNATURE", "KV_DIM", "LAYERS", "MODEL_REVISION", "PATCHES_PER_VIEW",
           "PATCH_WIDTH", "STATE_DIM", "STEPS", "TIMESTEP_CHANNELS", "VIEW_GRID",
           "VISION_BLOCKS", "VISION_DIM", "VISION_FFN", "VISUAL_TOKENS_PER_VIEW"]
