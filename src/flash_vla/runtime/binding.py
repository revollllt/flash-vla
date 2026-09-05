"""Plan validation against backend-declared route constraints.

A plan maps call sites to backends and is resolved once, before capture. Some
call sites share a buffer contract that only holds when they resolve to the
same backend -- a Q that crosses two call sites in implementation-owned
scratch, a producer/consumer pair joined by a readiness counter. The backend
that owns such a contract declares it as a `RouteConstraint`; this module
checks a plan against every declaration and rejects a violation at engine
construction rather than at the first replay. No torch, no backend imports:
constraints are data, and the Target's backend registry hands them in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RouteConstraint:
    """If any call site in `anchors` routes to the declaring backend, every
    call site in `members` must route there too."""
    anchors: frozenset[str]
    members: frozenset[str]
    reason: str

    @classmethod
    def atomic(cls, names: Iterable[str], reason: str) -> "RouteConstraint":
        """The call sites in `names` move together or not at all."""
        group = frozenset(names)
        return cls(group, group, reason)

    @classmethod
    def requires(cls, anchor: str, needs: Iterable[str], reason: str) -> "RouteConstraint":
        """Routing `anchor` here requires `needs` here as well; not the converse."""
        return cls(frozenset({anchor}), frozenset(needs) | {anchor}, reason)


def resolve(plan: Mapping[str, str] | None, default: str,
            call_sites: Iterable[str]) -> dict[str, str]:
    """The backend of every call site under `plan`, `default` where unnamed."""
    plan = dict(plan or {})
    return {name: plan.get(name, default) for name in call_sites}


def validate(routes: Mapping[str, str],
             constraints: Mapping[str, Iterable[RouteConstraint]]) -> None:
    """Raise `ValueError` when `routes` violates any backend's constraints.

    `routes` is the resolved call-site -> backend map; `constraints` maps a
    backend name to the constraints it declares.
    """
    for backend, declared in constraints.items():
        for constraint in declared:
            if not any(routes.get(name) == backend for name in constraint.anchors):
                continue
            stray = {name: routes.get(name) for name in constraint.members
                     if routes.get(name) != backend}
            if stray:
                raise ValueError(
                    f"backend {backend!r} requires {sorted(constraint.members)} to route "
                    f"together ({constraint.reason}); got {stray}")


def check_backends_provide(routes: Mapping[str, str],
                           provided: Mapping[str, Iterable[str]]) -> None:
    """Raise `KeyError` when a route names an unknown backend or an unprovided call site."""
    for name, backend in routes.items():
        if backend not in provided:
            raise KeyError(f"plan names unknown backend {backend!r} for {name!r}; "
                           f"known: {sorted(provided)}")
        if name not in set(provided[backend]):
            raise KeyError(f"backend {backend!r} does not implement call site {name!r}; "
                           f"it provides {sorted(provided[backend])}")
