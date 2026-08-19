"""Pi0.5 execution plan specialized for NVIDIA H100.

Currently the prefix pass only: vision, prompt embedding, and the transformer
encoder that builds the KV cache. The decoder is blocked on the tile-dataflow
spec for its AdaRMSNorm call sites (see PLAN.md §2.4).
"""

from .engine import Pi05Prefix
from .ops import op_table

__all__ = ["Pi05Prefix", "op_table"]
