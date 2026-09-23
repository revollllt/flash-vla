"""Build and load the hand-written CUDA libraries: one compiler lookup, one cache.

A backend that compiles `.cu` sources declares them once as a `NativeLibrary`
-- sources, the exact target-architecture flags, further nvcc flags, include
directories and the headers the build depends on -- and calls `load()` before
graph capture. The library is compiled on first use into
`<repo>/.cache/cuda_ext/<name>_<key>/`, keyed on everything that determines
the binary: the compiler, the full command and the bytes of every source and
declared header, so an edit never reuses a stale `.so` and two variants never
overwrite each other.

The compiler is `$FLASH_VLA_NVCC`, else `$CUDA_HOME/bin/nvcc`, else `nvcc` on
`PATH`; a library may name one more environment variable that takes precedence
for it alone (LingBot's `LINGBOT_NVCC`). No machine path is assumed.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

#: The repository root: `.cache/` and `third_party/` live here.
REPO = Path(__file__).resolve().parents[4]
#: The CUTLASS tree the CUTLASS-based libraries compile against.
CUTLASS_DIR = Path(os.environ.get("CUTLASS_DIR", REPO / "third_party" / "cutlass"))
#: CUTLASS's version header: part of the key of every CUTLASS-based build.
CUTLASS_VERSION_HEADER = CUTLASS_DIR / "include" / "cutlass" / "version.h"
#: The vendor-level CUDA tile primitives (`hardware/nvidia/cuda/tile`).
TILE_INCLUDE_DIR = Path(__file__).resolve().parent / "cuda"
#: The sm_90 tile primitive headers the H100 kernels include.
SM90_TILE_HEADERS = tuple(sorted((TILE_INCLUDE_DIR / "tile" / "sm90").glob("*.cuh")))


def find_nvcc(override_env: str | None = None) -> Path:
    """The CUDA compiler: `$override_env`, `$FLASH_VLA_NVCC`, `$CUDA_HOME/bin/nvcc`, `PATH`."""
    for variable in (override_env, "FLASH_VLA_NVCC"):
        if variable is not None and os.environ.get(variable):
            return Path(os.environ[variable])
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        return Path(cuda_home) / "bin" / "nvcc"
    found = shutil.which("nvcc")
    if found is None:
        raise RuntimeError("no nvcc: set FLASH_VLA_NVCC or CUDA_HOME, or put nvcc on PATH")
    return Path(found)


@dataclass(frozen=True)
class NativeLibrary:
    """One shared library compiled from CUDA sources.

    `arch` is the exact target flag sequence (`("-arch=sm_90a",)` or
    `("-gencode", "arch=compute_120a,code=sm_120a")`); `flags` follows it
    verbatim. `flags_env` names an environment variable of extra
    space-separated flags (a kernel's ablation switches), which join the key.
    `link_driver` links the CUDA driver API from the toolkit's stubs.
    """
    name: str
    sources: tuple[Path, ...]
    arch: tuple[str, ...]
    flags: tuple[str, ...] = ()
    include_dirs: tuple[Path, ...] = ()
    headers: tuple[Path, ...] = ()
    link_driver: bool = False
    flags_env: str | None = None
    nvcc_env: str | None = None

    def command(self, output: Path) -> list[str]:
        """The nvcc invocation that writes this library to `output`."""
        nvcc = find_nvcc(self.nvcc_env)
        extra = os.environ.get(self.flags_env, "").split() if self.flags_env is not None else []
        link = [f"-L{nvcc.parents[1]}/lib64/stubs", "-lcuda"] if self.link_driver else []
        return [str(nvcc), "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
                *self.arch, *self.flags, *extra,
                *(f"-I{directory}" for directory in self.include_dirs),
                "-o", str(output), *(str(source) for source in self.sources), *link]

    def path(self) -> Path:
        """Where the build for the current compiler, flags and sources lives."""
        key = hashlib.sha256()
        for token in self.command(Path(self.name)):
            key.update(token.encode() + b"\0")
        for dependency in (*self.sources, *self.headers):
            key.update(dependency.read_bytes())
        directory = REPO / ".cache" / "cuda_ext" / f"{self.name}_{key.hexdigest()[:16]}"
        return directory / f"lib{self.name}.so"

    def build(self, verbose: bool = False) -> Path:
        """Compile unless this exact build exists; return the `.so` path.

        `verbose` streams the command and nvcc's own output (ptxas register
        reports among them) to the terminal instead of capturing it.
        """
        output = self.path()
        if output.exists():
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and renamed, so a concurrent process never
        # loads a half-written library.
        staging = output.with_suffix(f".{os.getpid()}.tmp")
        command = self.command(staging)
        result = subprocess.run(command, capture_output=not verbose, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"nvcc failed building {self.name} ({result.returncode}):\n"
                               f"{' '.join(command)}\n{result.stdout}\n{result.stderr}")
        os.replace(staging, output)
        return output

    def load(self, verbose: bool = False) -> ctypes.CDLL:
        """The library, built first if needed. Callers declare its C ABI once
        and keep the handle (`library()` in each backend module)."""
        return ctypes.CDLL(str(self.build(verbose=verbose)))


__all__ = ["CUTLASS_DIR", "CUTLASS_VERSION_HEADER", "NativeLibrary", "SM90_TILE_HEADERS",
           "TILE_INCLUDE_DIR", "find_nvcc"]
