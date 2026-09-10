"""Raw-CUDA backend for the Gemma backbone, registrable by any H100 Target.

`wrappers` provides the call sites a plan can route here as a stateful
backend, so each op table owns its own library handle and workspace. The
module satisfies the registry contract of `flash_vla.runtime.registry`.

`ROUTE_CONSTRAINTS` is deliberately empty and there is no `graph_contract`.
Every wrapper reads and writes the graph's own buffers in the layout the
TileLang route uses, and none owns scratch that crosses a call-site boundary,
so each call site may be routed here on its own and `tests.targets`'s route check
needs no per-Target oracle for this backend. A future backbone QKV projection
writing head-major Q into implementation-owned scratch would change that and
would have to declare a constraint, the way the action expert's attention pair
does.
"""

from . import wrappers

NAMES = wrappers.NAMES
OPS = wrappers.OPS
make_wrappers = wrappers.make_wrappers

#: No call site here shares a buffer contract with another.
ROUTE_CONSTRAINTS: tuple = ()

__all__ = ["NAMES", "OPS", "ROUTE_CONSTRAINTS", "make_wrappers", "wrappers"]
