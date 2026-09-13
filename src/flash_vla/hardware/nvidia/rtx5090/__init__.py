"""RTX 5090 (consumer Blackwell, sm_120) deployment targets.

Bring-up only: this package carries the hardware axis -- static specification
and the measured table beside it -- and no Target yet. See
``measured/isa-support.md`` for what an sm_90a kernel has to be rewritten into
before one can exist.
"""

from .spec import RTX5090Spec

__all__ = ["RTX5090Spec"]
