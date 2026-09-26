"""`Registry.resolve`: every candidate is checked, the first that supports the
shape runs, and a failure says which shape or fallback caused it."""
from __future__ import annotations

from typing import Mapping

import pytest

from flash_vla.runtime.binding import RouteConstraint
from flash_vla.runtime.registry import Backend, Registry
from flash_vla.runtime.workspace import Scratch

SITES = ("up", "down")


def no_wrappers(scratch: Scratch, selected: frozenset[str]) -> Mapping[str, object]:
    return {}


def long_prefix(shape: Mapping[str, int]) -> bool:
    return shape["prefix_len"] == 968


REGISTRY = Registry({
    "general": Backend(names=frozenset(SITES), make_wrappers=no_wrappers),
    "long-prefix": Backend(names=frozenset(SITES), make_wrappers=no_wrappers, supports=long_prefix),
    "paired": Backend(names=frozenset(SITES), make_wrappers=no_wrappers,
                      route_constraints=(RouteConstraint.atomic(SITES, "one scratch"),)),
    "up-only": Backend(names=frozenset({"up"}), make_wrappers=no_wrappers),
}, default="general")


def test_the_first_candidate_that_supports_the_shape_runs() -> None:
    plan = {"up": ("long-prefix", "general")}
    assert REGISTRY.resolve(plan, SITES, {"prefix_len": 968}) == {"up": "long-prefix",
                                                                  "down": "general"}
    assert REGISTRY.resolve(plan, SITES, {"prefix_len": 712}) == {"up": "general",
                                                                  "down": "general"}


@pytest.mark.parametrize("first,message", [("typo", "unknown backend 'typo'"),
                                           ("up-only", "does not implement call site 'down'")])
def test_every_candidate_is_checked_not_only_the_one_that_runs(first: str, message: str) -> None:
    with pytest.raises(KeyError, match=message):
        REGISTRY.resolve({"down": (first, "general")}, SITES, {"prefix_len": 968})


def test_a_plan_for_a_pruned_call_site_is_still_checked() -> None:
    with pytest.raises(KeyError, match="unknown backend 'typo'"):
        REGISTRY.resolve({"down": ("typo",)}, ("up",), {"prefix_len": 968})


def test_no_supporting_candidate_names_the_shape() -> None:
    with pytest.raises(ValueError, match="supports shape {'prefix_len': 712}"):
        REGISTRY.resolve({"up": ("long-prefix",)}, SITES, {"prefix_len": 712})


def test_a_fallback_that_splits_a_constrained_group_names_the_candidates() -> None:
    split = {"up": ("long-prefix", "paired"), "down": ("general",)}
    assert REGISTRY.resolve(split, SITES, {"prefix_len": 968}) == {"up": "long-prefix",
                                                                   "down": "general"}
    with pytest.raises(ValueError,
                       match=r"requires .*\(after shape fallback: \{'up': \('long-prefix', 'paired'\)\}"):
        REGISTRY.resolve(split, SITES, {"prefix_len": 712})
