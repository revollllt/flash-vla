from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks import source


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
    package = current / source.BACKENDS.parent
    (package / "backends").mkdir(parents=True)
    (package / "backends/upstream.py").write_text("def make_wrappers(*args, **kwargs): return {}\n")
    (package / "__init__.py").write_text("from .backends import TARGET\n")
    (package / "backends/__init__.py").write_text(
        "from . import upstream\nfrom types import SimpleNamespace\nTARGET=SimpleNamespace(marker='old', state=[])\n")
    git(current, "add", ".")
    git(current, "commit", "-m", "old")
    old_revision = git(current, "rev-parse", "HEAD")
    old = tmp_path / "old"
    git(current, "worktree", "add", "--detach", str(old), old_revision)
    (package / "backends/__init__.py").write_text(
        "from . import upstream\nfrom types import SimpleNamespace\nTARGET=SimpleNamespace(marker='new', state=[])\n")
    git(current, "add", ".")
    git(current, "commit", "-m", "candidate")
    return current, old


def test_loads_actual_old_and_new_modules_with_independent_globals(checkouts):
    current, old = checkouts
    with patch.object(source, "ROOT", current):
        a, ap = source.lingbot_target(old)
        b, bp = source.lingbot_target(current)
    assert (a.marker, b.marker) == ("old", "new")
    a.state.append("control")
    assert b.state == []
    assert ap["revision"] == git(old, "rev-parse", "HEAD")
    assert bp["revision"] == git(current, "rev-parse", "HEAD")
    assert ap["module"] != bp["module"]
    assert Path(ap["target_package"]).is_relative_to(old)


def test_refuses_unisolated_shared_source_change(checkouts):
    current, old = checkouts
    (current / "src/flash_vla/runtime/identity.py").write_text("# different runtime\n")
    git(current, "add", ".")
    git(current, "commit", "-m", "shared change")
    with patch.object(source, "ROOT", current):
        with pytest.raises(ValueError, match="outside isolated LingBot"):
            source.lingbot_target(old)


def test_refuses_dirty_source(checkouts):
    current, old = checkouts
    (old / "src/flash_vla/runtime/identity.py").write_text("# uncommitted\n")
    with patch.object(source, "ROOT", current):
        with pytest.raises(ValueError, match="clean committed"):
            source.lingbot_target(old)

def test_source_wrappers_restore_rope_between_engines_and_after_error():
    official = lambda: "official"
    external = SimpleNamespace(apply_rope=official)

    def backend():
        original = None

        def make(scratch, selected_names=None, *, enabled=False):
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
        return SimpleNamespace(make_wrappers=make)

    first, second = backend(), backend()
    source.isolate_rope(first)
    source.isolate_rope(second)
    scratch = SimpleNamespace(assets={"upstream": "/test/upstream"})
    cached = first.make_wrappers(scratch, enabled=True)
    reference = second.make_wrappers(scratch, enabled=False)
    with patch.object(source.importlib, "import_module", return_value=external), \
         patch.object(source.sys, "path", list(source.sys.path)):
        assert cached["vision"]() == "cached"
        assert external.apply_rope is official
        assert reference["vision"]() == "official"
        for engine, expected in ((cached, "cached"), (reference, "official"), (cached, "cached")):
            assert engine["prefix"]() == expected
            assert external.apply_rope is official
        with pytest.raises(RuntimeError, match="execution failed"):
            cached["prefix"](fail=True)
        assert external.apply_rope is official


def test_qualification_rejects_explicit_source_unbound_to_receipt(tmp_path):
    from lab.optimize import runner
    checkout = str(tmp_path / "recorded")
    record = dict(target="hardware/nvidia/h100/lingbot_vla", repository=checkout,
                  spec=dict(qualification_sources=dict(incumbent={}, candidate={},
                      incumbent_checkout=checkout, candidate_checkout=checkout),
                      options={"source_checkout": str(tmp_path / "unrecorded")}))
    with patch.object(runner.store, "changed_inputs", return_value=[]):
        with pytest.raises(ValueError, match="candidate source option differs"):
            runner._command(record, "qualify")


def test_official_adapter_forwards_source_without_replacing_oracle(tmp_path):
    from eval.lingbot import parity
    oracle = tmp_path / "official-oracle"
    with patch.object(parity, "run", return_value={"passed": True}) as run:
        assert parity.main(["--oracle", str(oracle), "--seed", "42",
                            "--option", "asset_config=assets.json",
                            "--option", "source_checkout=/actual/source"]) == 0
    run.assert_called_once_with("reference", oracle, 42, 36, 10,
                                asset_config="assets.json", source_checkout="/actual/source")
