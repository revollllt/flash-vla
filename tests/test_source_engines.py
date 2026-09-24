from functools import partial
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from flash_vla.runtime.registry import Backend
from measurement import source_checkout

#: A checkout's Target stand-in: a frozen value with a registry, like `runtime.vla.Target`.
FAKE_TARGET = (
    "from dataclasses import dataclass, field\n"
    "@dataclass(frozen=True)\n"
    "class FakeTarget:\n"
    "    marker: str\n"
    "    registry: object = None\n"
    "    state: list = field(default_factory=list)\n")
BACKENDS_PACKAGE = FAKE_TARGET + (
    "from flash_vla.runtime.registry import Backend, Registry\n"
    "from . import upstream\n"
    "BACKENDS = {{'upstream': Backend(names=frozenset({{'vision'}}), "
    "make_wrappers=upstream.make_wrappers)}}\n"
    "REGISTRY = Registry(BACKENDS, default='upstream')\n"
    "TARGET = FakeTarget(marker={marker!r})\n")


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkouts(tmp_path):
    current = tmp_path / "current"
    current.mkdir()
    git(current, "init")
    git(current, "config", "user.name", "Source test")
    git(current, "config", "user.email", "source@example.invalid")
    runtime = current / "src/flash_vla/runtime"
    runtime.mkdir(parents=True)
    (runtime / "identity.py").write_text("# source locator\n")
    package = current / source_checkout.BACKENDS.parent
    (package / "backends").mkdir(parents=True)
    (package / "backends/upstream.py").write_text("def make_wrappers(*args, **kwargs): return {}\n")
    (package / "__init__.py").write_text("from .backends import TARGET\n")
    (package / "backends/__init__.py").write_text(BACKENDS_PACKAGE.format(marker="old"))
    git(current, "add", ".")
    git(current, "commit", "-m", "old")
    old_revision = git(current, "rev-parse", "HEAD")
    old = tmp_path / "old"
    git(current, "worktree", "add", "--detach", str(old), old_revision)
    (package / "backends/__init__.py").write_text(BACKENDS_PACKAGE.format(marker="new"))
    git(current, "add", ".")
    git(current, "commit", "-m", "candidate")
    return current, old


def test_loads_actual_old_and_new_modules_with_independent_globals(checkouts):
    current, old = checkouts
    with patch.object(source_checkout, "ROOT", current):
        a, ap = source_checkout.lingbot_target(old)
        b, bp = source_checkout.lingbot_target(current)
    assert (a.marker, b.marker) == ("old", "new")
    a.state.append("control")
    assert b.state == []
    assert ap.revision == git(old, "rev-parse", "HEAD")
    assert bp.revision == git(current, "rev-parse", "HEAD")
    assert ap.module != bp.module
    assert Path(ap.target_package).is_relative_to(old)


def test_refuses_unisolated_shared_source_change(checkouts):
    current, old = checkouts
    (current / "src/flash_vla/runtime/identity.py").write_text("# different runtime\n")
    git(current, "add", ".")
    git(current, "commit", "-m", "shared change")
    with patch.object(source_checkout, "ROOT", current):
        with pytest.raises(ValueError, match="outside isolated LingBot"):
            source_checkout.lingbot_target(old)


@pytest.mark.parametrize('role', ['source', 'controller'])
def test_refuses_dirty_source(checkouts, role):
    current, old = checkouts
    dirty_root = old if role == "source" else current
    (dirty_root / "src/flash_vla/runtime/identity.py").write_text("# uncommitted\n")
    with patch.object(source_checkout, "ROOT", current):
        with pytest.raises(ValueError, match="clean committed"):
            source_checkout.lingbot_target(old)


def test_recovery_records_and_notes_do_not_invalidate_committed_execution(checkouts):
    current, old = checkouts
    for root in (current, old):
        for name in (".goal-task/task/todo.md", ".agents/notes/progress.md"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("updated recovery state")
    with patch.object(source_checkout, "ROOT", current):
        target, provenance = source_checkout.lingbot_target(old)
    assert target.marker == "old"
    assert provenance.revision == git(old, "rev-parse", "HEAD")
    assert provenance.controller_revision == git(current, "rev-parse", "HEAD")


def test_source_wrappers_restore_rope_between_engines_and_after_error():
    official = lambda: "official"
    external = SimpleNamespace(apply_rope=official)
    names = frozenset({"vision", "prefix"})

    def backend(enabled):
        original = None

        def make(scratch, selected_names, *, enabled):
            initialized = False

            def vision():
                nonlocal original, initialized
                if not initialized:
                    if original is None:
                        original = external.apply_rope
                    external.apply_rope = (lambda: "cached") if enabled else original
                    initialized = True
                return external.apply_rope()

            def prefix(fail=False):
                if fail:
                    raise RuntimeError("execution failed")
                return external.apply_rope()

            return {"vision": vision, "prefix": prefix}
        return source_checkout.isolate_rope(Backend(names=names, make_wrappers=partial(make, enabled=enabled)))

    scratch = SimpleNamespace(assets={"upstream": "/test/upstream"})
    cached = backend(True).make_wrappers(scratch, names)
    reference = backend(False).make_wrappers(scratch, names)
    with patch.object(source_checkout.importlib, "import_module", return_value=external), \
         patch.object(source_checkout.sys, "path", list(source_checkout.sys.path)):
        assert cached["vision"]() == "cached"
        assert external.apply_rope is official
        assert reference["vision"]() == "official"
        for engine, expected in ((cached, "cached"), (reference, "official"), (cached, "cached")):
            assert engine["prefix"]() == expected
            assert external.apply_rope is official
        with pytest.raises(RuntimeError, match="execution failed"):
            cached["prefix"](fail=True)
        assert external.apply_rope is official


def test_official_adapter_forwards_source_without_replacing_oracle(tmp_path):
    from eval.lingbot import parity
    oracle = tmp_path / "official-oracle"
    with patch.object(parity, "run", return_value={"passed": True}) as run:
        assert parity.main(["--oracle", str(oracle), "--seed", "42",
                            "--option", "asset_config=assets.json",
                            "--option", "source_checkout=/actual/source"]) == 0
    run.assert_called_once_with("reference", oracle, 42, 36, 10,
                                asset_config="assets.json", source_checkout="/actual/source")


def test_prebound_route_factory_is_isolated(checkouts):
    """A route whose factory was bound at import still runs isolated: isolation
    wraps the registered backends, not the module attribute a route may bypass."""
    current, _ = checkouts
    package = current / source_checkout.BACKENDS.parent
    (package / "backends/upstream.py").write_text(
        "def make_wrappers(*args, **kwargs):\n"
        "    import probe_rope_external as external\n"
        "    def vision():\n"
        "        external.apply_rope = lambda: 'cached'\n"
        "        return external.apply_rope()\n"
        "    def prefix(): return external.apply_rope()\n"
        "    return dict(vision=vision, prefix=prefix)\n")
    (package / "backends/alias.py").write_text(
        "from .upstream import make_wrappers as bound_factory\n"
        "def make_wrappers(*args, **kwargs): return bound_factory(*args, **kwargs)\n")
    (package / "backends/__init__.py").write_text(
        FAKE_TARGET +
        "from flash_vla.runtime.registry import Backend, Registry\n"
        "from . import upstream, alias\n"
        "NAMES = frozenset({'vision', 'prefix'})\n"
        "BACKENDS = {'reference': Backend(names=NAMES, make_wrappers=upstream.make_wrappers),\n"
        "            'candidate': Backend(names=NAMES, make_wrappers=alias.make_wrappers)}\n"
        "REGISTRY = Registry(BACKENDS, default='reference')\n"
        "TARGET = FakeTarget(marker='prebound')\n")
    git(current, "add", ".")
    git(current, "commit", "-m", "prebound route")
    official = lambda: "official"
    external = SimpleNamespace(apply_rope=official)
    with patch.object(source_checkout, "ROOT", current):
        target, _ = source_checkout.lingbot_target(current)
    backend = target.registry.backends["candidate"]
    scratch = SimpleNamespace(assets={"upstream": "/test/upstream"})
    with patch.dict(source_checkout.sys.modules, probe_rope_external=external), \
         patch.object(source_checkout.importlib, "import_module", return_value=external), \
         patch.object(source_checkout.sys, "path", list(source_checkout.sys.path)):
        engine = backend.make_wrappers(scratch, backend.names)
        assert engine["vision"]() == "cached"
        assert external.apply_rope is official
        assert engine["prefix"]() == "cached"
        assert external.apply_rope is official
