"""The TileLang JIT conventions every NVIDIA backend in this repository shares.

What lives here is the part that is a property of TileLang and of nothing else:
the two `pass_configs` sets, the wrapper that JIT-compiles a builder under a
name, and the raw-builder registry the autotuner re-wraps from. Tile shapes,
call-site configs and kernel bodies stay with the backend that owns them.

Each importing module makes its own `KernelSet`, so each keeps a private
`RAW_KERNELS`. That is deliberate rather than incidental: the two Targets and
the component packages declare kernels under the same names with bodies that
have diverged (`tl_rms_factor` carries a PDL-trigger parameter on Pi0.5 and not
on Pi0), and one shared registry would silently hand the autotuner whichever
module imported last.

`import tilelang` inside this module resolves to the installed TileLang
package, not to this one: Python 3 has no implicit relative imports.
"""
from __future__ import annotations

import tilelang

#: Fast-math lowering, the default for every kernel here.
FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
#: Fast math with TileLang's producer/consumer warp split disabled. Below one
#: wave the producer warp has no work to hide and still costs warps and
#: mbarrier traffic, so sub-wave call sites select this.
NO_WARP_SPEC = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


class KernelSet:
    """One JIT namespace: its decorators and its own raw-builder registry.

    Construct one per module that declares kernels, then export its
    `kernel`, `variant` and `RAW_KERNELS` as module attributes so call sites
    and `autotune.rewrap` read them where they always did.
    """

    def __init__(self) -> None:
        #: name -> (raw builder, out_idx), what the autotuner re-wraps.
        self.RAW_KERNELS: dict[str, tuple] = {}

    def variant(self, builder, name: str, *, warp_spec: bool = True,
                infer_output: bool = False):
        """JIT-compile `builder` under `name` and record it for the autotuner.

        `infer_output=False` passes `out_idx=None`: every tensor in the
        signature is a parameter and the kernel writes its result through one
        of them. `infer_output=True` leaves TileLang's default inference on,
        which is what a builder that ends in `return C` needs.

        The raw builder is kept because `.compile(pass_configs=...)` is
        rejected as unhashable and a compiled kernel does not expose its
        builder, so the autotuner has no other way to re-wrap it with
        different flags. For the same reason -- TileLang's kernel objects
        define equality but not a hash -- the name is stamped onto the object
        as `tl_name`, which is what a wrapper's compile cache keys on.
        """
        out_idx = "default" if infer_output else None
        pass_configs = FAST_MATH if warp_spec else NO_WARP_SPEC
        self.RAW_KERNELS[name] = (builder, out_idx)
        if infer_output:
            jitted = tilelang.jit(builder, pass_configs=pass_configs)
        else:
            jitted = tilelang.jit(builder, out_idx=out_idx, pass_configs=pass_configs)
        jitted.tl_name = name
        return jitted

    def kernel(self, builder=None, *, warp_spec: bool = True,
               infer_output: bool = False):
        """Decorator form of `variant` for kernels that have only one variant."""
        def decorate(fn):
            return self.variant(fn, fn.__name__, warp_spec=warp_spec,
                                infer_output=infer_output)
        return decorate(builder) if builder is not None else decorate


__all__ = ["FAST_MATH", "NO_WARP_SPEC", "KernelSet"]
