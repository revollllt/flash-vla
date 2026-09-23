"""Backend registry: how a Target's implementations are named, routed and built.

A backend is a `Backend` value, declared once next to its implementation:

    names              the call sites it implements
    make_wrappers      (scratch, selected) -> {call_site: wrapper} for the
                       selected call sites; `scratch` is the runner's workspace
                       allocator (`runtime/workspace.py`), which the backend
                       uses for any device memory that outlives one call, and
                       `scratch.assets` its read-only asset paths
    route_constraints  `RouteConstraint`s over its call sites (`runtime/binding.py`)
    graph_contract     the kernel-name patterns the captured program must and
                       must not contain, given the call sites routed to it

A variant of a backend (the same wrappers with a launch attribute armed) is
`dataclasses.replace(backend, make_wrappers=...)`: the registry never needs to
know that two names share an implementation, and a backend never needs to know
the name it is registered under.

The call sites themselves, standard or a model's extension ops, belong to the
model (`runtime/vla.py`); a backend only implements some of them. The
registry resolves a plan over a graph's call sites, validates it against
every backend's constraints, and builds the one op table an engine runs on. It
has no model, device or kernel knowledge of its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from . import binding
from .binding import RouteConstraint
from .workspace import Scratch

#: One call site's implementation: positional arguments in `OpSpec.params`
#: order, outputs written in place; the runner ignores the return value.
Wrapper = Callable[..., object]
#: Builds the wrappers of the selected call sites of one backend.
WrapperFactory = Callable[[Scratch, frozenset[str]], Mapping[str, Wrapper]]


@dataclass(frozen=True)
class GraphContract:
    """Kernel-name patterns the captured program must not contain (`forbid`)
    and must contain exactly one kernel of (`require_one`)."""
    forbid: tuple[str, ...] = ()
    require_one: tuple[str, ...] = ()

    def merged(self, other: GraphContract) -> GraphContract:
        """Both contracts at once, each pattern listed once in first-seen order."""
        return GraphContract(forbid=tuple(dict.fromkeys(self.forbid + other.forbid)),
                             require_one=tuple(dict.fromkeys(self.require_one + other.require_one)))


def no_graph_contract(routed_call_sites: frozenset[str]) -> GraphContract:
    """The contract of a backend that constrains no kernel names."""
    return GraphContract()


@dataclass(frozen=True)
class Backend:
    """One routable implementation of some call sites; see the module docstring."""
    names: frozenset[str]
    make_wrappers: WrapperFactory
    route_constraints: tuple[RouteConstraint, ...] = ()
    graph_contract: Callable[[frozenset[str]], GraphContract] = no_graph_contract


def call_sites_by_backend(routes: Mapping[str, str]) -> dict[str, frozenset[str]]:
    """The call sites each backend of `routes` is routed, in first-routed order."""
    grouped: dict[str, set[str]] = {}
    for call_site, backend in routes.items():
        grouped.setdefault(backend, set()).add(call_site)
    return {backend: frozenset(call_sites) for backend, call_sites in grouped.items()}


class Registry:
    """The backends of one Target and the default one a plan does not name."""

    def __init__(self, backends: Mapping[str, Backend], default: str) -> None:
        if default not in backends:
            raise KeyError(f"default backend {default!r} is not registered: {sorted(backends)}")
        self.backends: Mapping[str, Backend] = MappingProxyType(dict(backends))
        self.default = default

    def provided(self) -> dict[str, frozenset[str]]:
        """Call sites each backend implements."""
        return {name: backend.names for name, backend in self.backends.items()}

    def constraints(self) -> dict[str, tuple[RouteConstraint, ...]]:
        return {name: backend.route_constraints for name, backend in self.backends.items()}

    def resolve(self, plan: Mapping[str, str] | None,
                call_sites: Iterable[str]) -> dict[str, str]:
        """The backend of every call site under `plan`, validated."""
        provided = self.provided()
        binding.check_backends_provide(plan or {}, provided)
        routes = binding.resolve(plan, self.default, call_sites,
                                 known_call_sites=frozenset().union(*provided.values()))
        binding.check_backends_provide(routes, provided)
        binding.validate(routes, self.constraints())
        return routes

    def op_table(self, routes: Mapping[str, str], scratch: Scratch) -> Mapping[str, Wrapper]:
        """One wrapper per call site, built by the backend the route names."""
        table: dict[str, Wrapper] = {}
        for backend, call_sites in call_sites_by_backend(routes).items():
            built = self.backends[backend].make_wrappers(scratch, call_sites)
            missing = sorted(call_sites - set(built))
            if missing:
                raise KeyError(f"backend {backend!r} did not build {missing}")
            table.update({call_site: built[call_site] for call_site in call_sites})
        return MappingProxyType(table)

    def graph_contract(self, routes: Mapping[str, str]) -> GraphContract:
        """The union of every routed backend's contract over its routed call sites."""
        contract = GraphContract()
        for backend, call_sites in call_sites_by_backend(routes).items():
            contract = contract.merged(self.backends[backend].graph_contract(call_sites))
        return contract

    def atomic_groups(self, routes: Mapping[str, str]) -> tuple[frozenset[str], ...]:
        """Call sites that must be invoked together on `routes`: every constraint
        whose members all resolve to the declaring backend, overlapping groups merged."""
        groups = [frozenset(constraint.members)
                  for backend, constraints in self.constraints().items()
                  for constraint in constraints
                  if all(routes.get(name) == backend for name in constraint.members)]
        merged: list[frozenset[str]] = []
        for group in groups:
            overlapping = [existing for existing in merged if existing & group]
            merged = [existing for existing in merged if not existing & group]
            merged.append(group.union(*overlapping))
        return tuple(merged)


__all__ = ["Backend", "GraphContract", "Registry", "Wrapper"]
