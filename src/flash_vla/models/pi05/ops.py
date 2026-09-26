"""The call sites Pi0.5's graph emits: every standard one of `runtime/ops.py`.

The backbone's call sites run the replay-time bucket's rows (`graph.py`), so a
device's kernels see only the rows a bucket holds; no call site takes the
prefix mask to skip padding itself.
"""
from __future__ import annotations

from flash_vla.runtime.ops import STANDARD

#: The standard call sites: every one the Pi0.5 graph emits.
CALL_SITES = frozenset(spec.name for spec in STANDARD)

__all__ = ["CALL_SITES"]
