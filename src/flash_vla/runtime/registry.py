"""Backend registry: how a Target's implementations are named, routed and built.

A backend is a `Backend` value, declared once next to its implementation:

    names              the call sites it implements
    make_wrappers      (scratch, selected) -> {call_site: wrapper} for the
                       selected call sites; `scratch` is the runner's workspace
                       allocator (`runtime/workspace.py`), which the backend
                       uses for any device memory that outlives one call,
                       `scratch.assets` its read-only asset paths, and
                       `scratch.on_replay` where it registers the hook that
                       plans a kernel at each inference's replay-time length
    route_constraints  `RouteConstraint`s over its call sites (`runtime/binding.py`)
    graph_contract     the kernel-name patterns the captured program must and
                       must not contain, given the call sites routed to it
    supports           whether it runs correctly at a model's shape numbers
                       (a kernel specialized to one prefix length); a plan names
                       candidates per call site, and the first that supports the
                       shape runs it. With replay buckets the shape carries the
                       replay axis at each bucket (`ReplayAxis.at`), and one
                       route must support every bucket

A variant of a backend (the same wrappers with a launch attribute armed) is
`dataclasses.replace(backend, make_wrappers=...)`: the registry never needs to
know that two names share an implementation, and a backend never needs to know
the name it is registered under.

The call sites themselves, standard or a model's extension ops, belong to the
model (`runtime/vla.py`); a backend only implements some of them. The
registry resolves a plan -- ordered candidate backends per call site -- over
a graph's call sites at its shape, validates the routes against every
backend's constraints, and builds the one op table an engine runs on. It has
no model, device or kernel knowledge of its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from . import binding
from .binding import Candidates, RouteConstraint
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


def supports_every_shape(shape: Mapping[str, int]) -> bool:
    """A backend that runs at any shape of its call sites."""
    return True


@dataclass(frozen=True)
class Backend:
    """One routable implementation of some call sites; see the module docstring."""
    names: frozenset[str]
    make_wrappers: WrapperFactory
    route_constraints: tuple[RouteConstraint, ...] = ()
    graph_contract: Callable[[frozenset[str]], GraphContract] = no_graph_contract
    supports: Callable[[Mapping[str, int]], bool] = supports_every_shape


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

    def resolve(self, plan: Candidates | None, call_sites: Iterable[str],
                shape: Mapping[str, int]) -> dict[str, str]:
        """The backend of every call site under `plan` at `shape`: the first of its
        candidates (the default backend where the plan names none) that supports
        the shape, validated against every backend's constraints. Selection is
        greedy per call site, not a search over the constraints: a fallback that
        splits a constrained group is an error that names the candidates."""
        provided = self.provided()
        binding.check_backends_provide(plan or {}, provided)
        candidates = binding.resolve(plan, (self.default,), call_sites,
                                     known_call_sites=frozenset().union(*provided.values()))
        binding.check_backends_provide(candidates, provided)
        routes: dict[str, str] = {}
        for call_site, names in candidates.items():
            supporting = [name for name in names if self.backends[name].supports(shape)]
            if not supporting:
                raise ValueError(f"no candidate of {call_site!r} supports shape {dict(shape)}: "
                                 f"{list(names)}")
            routes[call_site] = supporting[0]
        try:
            binding.validate(routes, self.constraints())
        except ValueError as error:
            fallbacks = {call_site: names for call_site, names in candidates.items()
                         if routes[call_site] != names[0]}
            raise ValueError(f"{error} (after shape fallback: {fallbacks})") from error
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
