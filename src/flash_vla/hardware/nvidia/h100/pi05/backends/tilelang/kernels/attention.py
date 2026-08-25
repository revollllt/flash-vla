"""Vision and encoder attention, in torch.

Neither was a TileLang kernel in Pi0 and neither is ported here: both stages run
full bidirectional attention over a long sequence. Only the decoder's attention,
which is multi-query over a KV cache and badly served by a generic kernel, gets
a TileLang implementation.

`torch.compile` on both, so they fuse and capture into the CUDA graph cleanly.

The one Pi0.5 change is the mask. Pi0 ran with an empty prompt, so the encoder
prefix had no padding and needed no mask at all. Pi0.5 pads the prompt to 200
and masks the tail, and because the whole prefix is bidirectional the mask is
per-key and identical for every query row -- a single additive vector rather
than a 2-D structure.

That vector diverges from upstream in one harmless place. OpenPI's
`make_attn_mask` also zeroes the *rows* of padded queries, so their outputs are
a uniform average; here they attend normally. Padded rows never reach anything
valid -- their K and V are masked out of every later attention, and the final
encoder layer computes only K and V -- so the difference cannot propagate. It
does mean a parity check must compare valid rows, not the whole buffer.

A note carried over from Pi0 that is worth re-reading here: the docstring used
to claim cuDNN's fused kernel is already at the roofline. That is true of
`vision_attention`, which goes through `scaled_dot_product_attention`. It is not
true of `encoder_attention`, which is a hand-written multi-query chain that
materializes the score matrix -- 7744 x 968 at Pi0.5's sequence length, at least
60 MB of traffic per layer against 0.14 ms of arithmetic over all 18 layers.
Replacing it with a flash-style MQA kernel is an open opportunity (PLAN.md
§2.4), not a settled design.
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
def encoder_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float,
                      mask_bias: torch.Tensor) -> torch.Tensor:
    """Multi-query attention with an additive per-key mask.

    Q is (seq*heads, head_dim); K and V are (seq, head_dim); `mask_bias` is
    (seq,), zero on valid keys and a large finite negative on prompt padding.
    """
    logits = torch.matmul(Q, K.T) * scale + mask_bias[None, :]
    logits = torch.nn.functional.softmax(logits, dim=-1)
    return torch.matmul(logits, V).view(-1, ENCODER_DIM)
