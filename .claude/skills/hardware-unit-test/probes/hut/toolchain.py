"""Toolchain and cache discovery, shared by every probe.

The skill ships as a plugin, so nothing here may hardcode a path that exists on
only one cluster: a probe that silently builds against the wrong CUDA or the
wrong CUTLASS produces a number for a machine nobody described, which is the
failure `protocol.md` rule 11 exists to prevent.

Each lookup is the same three steps, in this order:

1. **An environment variable**, so a job script can pin an exact toolchain --
   and because `provenance.toolchain` in `constants.yaml` has to record what was
   actually used, pinning it explicitly is the supported path, not a fallback.
2. **Discovery** from whatever is on PATH.
3. **A failure that names the variable to set.** Never a default that happens to
   work on the author's machine.

Probes reach this by adding `probes/` to `sys.path`; they are run by file path,
not imported as a package, so there is no `__init__.py` to rely on.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def cuda_home() -> Path:
    """CUDA toolkit root, used as ``-L<root>/lib64/stubs -lcuda``.

    The stub library is what lets a probe link the driver API without a driver
    present at build time, so this must be a toolkit root and not just wherever
    ``nvcc`` happens to be symlinked.
    """
    for var in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        val = os.environ.get(var)
        if val:
            return Path(val)
    nvcc = shutil.which("nvcc")
    if nvcc:
        # <root>/bin/nvcc -> <root>. resolve() first so a symlinked nvcc
        # resolves to the toolkit it belongs to rather than to /usr/bin.
        return Path(nvcc).resolve().parent.parent
    for cand in (Path("/usr/local/cuda"), Path("/opt/cuda")):
        if (cand / "lib64" / "stubs").is_dir():
            return cand
    raise RuntimeError(
        "cannot locate the CUDA toolkit. Set CUDA_HOME to a toolkit root "
        "(the directory holding bin/nvcc and lib64/stubs), or put nvcc on PATH.")


def cutlass_include() -> Path:
    """The CUTLASS ``include/`` directory.

    The compute probes need ``cute/`` and the TMA probe needs
    ``cutlass/arch/barrier.h``, so a headers-only checkout is enough; nothing
    here links against a built CUTLASS.
    """
    for var in ("CUTLASS_DIR", "CUTLASS_PATH", "CUTLASS_HOME"):
        val = os.environ.get(var)
        if not val:
            continue
        root = Path(val)
        # Accept either the checkout root or the include/ inside it.
        for cand in (root / "include", root):
            if (cand / "cute" / "tensor.hpp").is_file():
                return cand
        raise RuntimeError(
            "%s=%s does not contain cute/tensor.hpp; point it at a CUTLASS "
            "checkout root or its include/ directory." % (var, val))
    for cand in (Path.home() / "cutlass" / "include",
                 Path("/usr/local/cutlass/include"),
                 Path("/opt/cutlass/include")):
        if (cand / "cute" / "tensor.hpp").is_file():
            return cand
    raise RuntimeError(
        "cannot locate CUTLASS headers. Set CUTLASS_DIR to a CUTLASS checkout "
        "(https://github.com/NVIDIA/cutlass); the probes need cute/ and "
        "cutlass/arch/ headers only, so a shallow clone is enough.")


def _enclosing_repo(start: Path) -> Path | None:
    for d in start.resolve().parents:
        if (d / ".git").exists() or (d / "pyproject.toml").exists():
            return d
    return None


def cache_dir(name: str, key: str) -> Path:
    """Directory for one probe's built ``.so``, keyed by source hash.

    Deliberately never inside the skill. A plugin directory should stay
    read-only, and a stale ``.so`` written into it would be copied along with
    the skill to the next machine, where it is a binary for the wrong GPU.
    """
    root = os.environ.get("HW_UNIT_TEST_CACHE")
    if root:
        base = Path(root)
    else:
        repo = _enclosing_repo(Path(__file__))
        if repo:
            base = repo / ".cache" / "cuda_ext"
        else:
            xdg = os.environ.get("XDG_CACHE_HOME")
            base = Path(xdg) if xdg else Path.home() / ".cache"
            base = base / "hardware-unit-test"
    out = base / ("%s_%s" % (name, key))
    out.mkdir(parents=True, exist_ok=True)
    return out
