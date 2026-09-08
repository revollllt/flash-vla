"""Isolated loading for the existing Gemma CUDA prefix-attention component only."""
import ctypes
import importlib.util
from pathlib import Path
import uuid
from . import store

MODULE = 'src/flash_vla/hardware/nvidia/h100/gemma_backbone/backends/cuda/enc_attn.py'


def load(source, output, *, cutlass, flags=(), backend='gemma-enc-attn'):
    if backend != 'gemma-enc-attn':
        raise ValueError(f'unsupported_protocol: component loader does not support {backend}')
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    name = 'isolated_enc_attn_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, source / MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._CUTLASS = Path(cutlass).resolve()
    module._TILE_ROOT = source / 'src/flash_vla/hardware/nvidia/cuda'
    module._extra_flags = lambda: list(flags)
    module._build_dir = lambda: output
    library = module.library(verbose=True)
    manifest = dict(id=name, backend=backend, source_root=str(source),
                    loaded_library=str(Path(library._name).resolve()),
                    launch_address=ctypes.cast(library.enc_attn_launch, ctypes.c_void_p).value,
                    flags=list(flags), geometry=module.geometry(), cutlass=str(module._CUTLASS),
                    source_manifest=store.read(source/'source.json'))
    store.write(output / 'loaded.json', manifest)
    return module, manifest


def require_distinct(manifests):
    for key in ('loaded_library', 'launch_address'):
        if len({m[key] for m in manifests}) != len(manifests):
            raise ValueError(f'aliased component artifacts: {key}')
