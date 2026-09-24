"""Load LingBot Python backends from a clean inference-source checkout.

`build` is `flash_vla.inference.build` with one more option,
`source_checkout`: the LingBot Target's backends loaded from that clean
checkout of this repository (`lingbot_target`), recorded as the runner's
implementation source, so a latency or parity run can compare two revisions
of those backends in one process.
"""
from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
from functools import wraps
from pathlib import Path
import subprocess
import sys
import uuid

from flash_vla.inference import build as build_registered, build_runner, resolve
from flash_vla.provenance import ImplementationProvenance
from flash_vla.runtime import ModelRunner
from flash_vla.runtime.registry import Backend, Registry, Wrapper
from flash_vla.runtime.vla import ConfigValue, PlanSpec, Target
from flash_vla.runtime.workspace import Scratch

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = Path("src/flash_vla/hardware/nvidia/h100/lingbot_vla/backends")


def isolate_rope(backend: Backend) -> Backend:
    """`backend` with the upstream `apply_rope` global scoped to each of its engines.

    Upstream LingBot keeps `apply_rope` as a module global that a cached route
    replaces; every wrapper call installs its engine's own value and restores
    the previous one afterwards, errors included.
    """
    make_wrappers = backend.make_wrappers

    def make(scratch: Scratch, selected_names: frozenset[str]) -> dict[str, Wrapper]:
        wrappers = make_wrappers(scratch, selected_names)
        active_rope: Wrapper | None = None

        def bind(function: Wrapper) -> Wrapper:
            @wraps(function)
            def call(*args: object, **kwargs: object) -> object:
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

    return replace(backend, make_wrappers=make)


def lingbot_target(checkout: str | Path) -> tuple[Target, ImplementationProvenance]:
    """The LingBot Target of `checkout`: its backend modules, isolated per engine.

    Shared inference code must be identical between `checkout` and this
    controller checkout; only the LingBot backends may differ. Every backend of
    the loaded package is wrapped by `isolate_rope`, and the returned Target
    routes to the isolated copies.
    """
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
    loaded = sys.modules[name + ".backends"]
    registry = Registry({backend_name: isolate_rope(backend)
                         for backend_name, backend in loaded.BACKENDS.items()},
                        default=loaded.REGISTRY.default)
    target = replace(module.TARGET, registry=registry)
    provenance = ImplementationProvenance(
        revision=revision, checkout=str(source), controller_revision=controller_revision,
        target_package=str(package), module=name,
        scope="isolated LingBot backends with per-call upstream RoPE; remaining src/flash_vla matches the source revision")
    return target, provenance


def build(name: str, plan: PlanSpec = "shipped", *, source_checkout: str | None = None,
          **options: ConfigValue) -> ModelRunner:
    """The runner of Target `name` on `plan`; with `source_checkout`, its
    backends come from that checkout, which only the LingBot Target supports."""
    if source_checkout is None:
        return build_registered(name, plan, **options)
    if resolve(name) != resolve("lingbot_vla"):
        raise ValueError(f"only LingBot loads its backends from a source checkout, not {name}")
    target, provenance = lingbot_target(source_checkout)
    return build_runner(target, plan, implementation_source=provenance, **options)
