"""Raw-CUDA backend for the Gemma backbone, registrable by any H100 Target.

`wrappers` provides the call sites a plan can route here as a stateful
backend, so each op table owns its own library handle and workspace; `BACKEND`
is what a Target registers.

`BACKEND` deliberately declares no route constraint and no graph contract.
Every wrapper reads and writes the graph's own buffers in the layout the
TileLang route uses, and none owns scratch that crosses a call-site boundary,
so each call site may be routed here on its own and `tests.targets`'s route check
needs no per-Target oracle for this backend. A future backbone QKV projection
writing head-major Q into implementation-owned scratch would change that and
would have to declare a constraint, the way the action expert's attention pair
does.
"""

from . import wrappers

BACKEND = wrappers.BACKEND
NAMES = wrappers.NAMES
make_wrappers = wrappers.make_wrappers

__all__ = ["BACKEND", "NAMES", "make_wrappers", "wrappers"]
