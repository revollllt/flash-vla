"""Load LingBot Python backends from a clean inference-source checkout."""
import importlib
import importlib.util
from functools import wraps
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = Path("src/flash_vla/hardware/nvidia/h100/lingbot_vla/backends")



def isolate_rope(upstream):
    """Scope the upstream apply_rope global to each sequential engine call."""
    make_wrappers = upstream.make_wrappers

    def make(scratch, *args, **kwargs):
        wrappers = make_wrappers(scratch, *args, **kwargs)
        active_rope = None

        def bind(function):
            @wraps(function)
            def call(*args, **kwargs):
                nonlocal active_rope
                path = str(scratch.assets["upstream"])
                if path not in sys.path:
                    sys.path.insert(0, path)
                lingbot = importlib.import_module("lingbotvla.models.vla.pi0.modeling_lingbot_vla")
                previous = lingbot.apply_rope
                if active_rope is not None:
                    lingbot.apply_rope = active_rope
                try:
                    return function(*args, **kwargs)
                finally:
                    active_rope = lingbot.apply_rope
                    lingbot.apply_rope = previous
            return call
        return {name: bind(function) for name, function in wrappers.items()}

    upstream.make_wrappers = make


def lingbot_target(checkout):
    """Keep shared inference code identical and isolate backend modules and their shared RoPE function."""
    source = Path(checkout).resolve()
    revisions = {}
    for checkout_root in dict.fromkeys((source, ROOT)):
        # Recovery records and agent notes are not executed by the evaluator.
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", ".",
             ":(exclude).goal-task/**", ":(exclude).agents/notes/**"],
            cwd=checkout_root, text=True,
        )
        if dirty:
            raise ValueError("source qualification requires clean committed execution files")
        revisions[checkout_root] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout_root, text=True,
        ).strip()
    revision, controller_revision = revisions[source], revisions[ROOT]
    changed = subprocess.run(
        ["git", "diff", "--name-only", revision, controller_revision, "--", "src/flash_vla"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    unsupported = [name for name in changed if not Path(name).is_relative_to(BACKENDS)]
    if unsupported:
        raise ValueError(f"unsupported source changes outside isolated LingBot backends: {unsupported}")
    name = "_flash_vla_lingbot_" + uuid.uuid4().hex
    package = source / BACKENDS.parent
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    backends = sys.modules[name + ".backends"].BACKENDS
    for backend in dict.fromkeys(backends.values()):
        isolate_rope(backend)
    provenance = dict(revision=revision, checkout=str(source),
                      controller_revision=controller_revision,
                      target_package=str(package), module=name,
                      scope="isolated LingBot backends with per-call upstream RoPE; remaining src/flash_vla matches the source revision")
    return module.TARGET, provenance
