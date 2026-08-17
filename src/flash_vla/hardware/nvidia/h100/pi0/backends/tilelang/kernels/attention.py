"""Vision and encoder attention, in torch.

These two were never Triton kernels and are not ported: both stages run full
bidirectional attention over a long sequence, where cuDNN's fused kernel is
already at the roofline. Only the decoder's attention, which is multi-query over
a KV cache and badly served by a generic kernel, has a TileLang implementation
(`kernels.tl_fd_flat_split` / `_combine`).

`torch.compile` on both, so they fuse and capture into the CUDA graph cleanly.
"""
from __future__ import annotations

import torch

VISION_HEADS = 16
VISION_HEAD_DIM = 72
ENCODER_DIM = 2048


@torch.compile
def vision_attention(QKV: torch.Tensor) -> torch.Tensor:
    """Multi-head self-attention over a packed QKV buffer, (views, 256, 3*1152) -> (views*256, 1152)."""
    QKV = QKV.view(-1, 256, 3, VISION_HEADS, VISION_HEAD_DIM).permute(0, 2, 3, 1, 4)
    Q, K, V = QKV[:, 0], QKV[:, 1], QKV[:, 2]
    attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    return attn.transpose(1, 2).reshape(Q.shape[0], 256, VISION_HEADS * VISION_HEAD_DIM)


@torch.compile
def encoder_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
    """Multi-query attention: all query heads share one K/V head. Q is (seq*heads, head_dim)."""
    logits = torch.matmul(Q, K.T) * scale
    logits = torch.nn.functional.softmax(logits, dim=-1)
    return torch.matmul(logits, V).view(-1, ENCODER_DIM)
