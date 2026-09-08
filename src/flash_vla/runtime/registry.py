"""Backend registry: how a Target's implementations are named, routed and built.

A Target declares its backends as a mapping from backend name to a module (or
any object) with this contract:

    NAMES              the call sites the backend implements
    make_wrappers(scratch, selected_names=None) -> {call_site: callable}
                       build the wrappers; `scratch(role, shape, dtype, device)`
                       is the runner's workspace allocator, which the backend
                       must use for any device memory that outlives one call
    ROUTE_CONSTRAINTS  optional `RouteConstraint`s (`runtime/binding.py`)
    graph_contract(routes) -> {"forbid": [...], "require_one": [...]}
                       optional kernel-name patterns the captured program must
                       and must not contain when these routes are active
    OPS                optional extension `OpSpec`s beyond the standard vocabulary

The registry resolves a plan over a graph's call sites, validates it against
every backend's constraints, and builds the one op table an engine runs on.
It has no model, device or kernel knowledge of its own.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

from . import binding
from .binding import RouteConstraint
from .ops import OpSpec

Scratch = Callable[[str, tuple[int, ...], Any, Any], Any]


class Registry:
    """The backends of one Target and the default one a plan does not name."""

    def __init__(self, backends: Mapping[str, Any], default: str) -> None:
        if default not in backends:
            raise KeyError(f"default backend {default!r} is not registered: {sorted(backends)}")
        self.backends = dict(backends)
        self.default = default

    def provided(self) -> dict[str, set[str]]:
        """Call sites each backend implements."""
        return {name: set(module.NAMES) for name, module in self.backends.items()}

    def constraints(self) -> dict[str, tuple[RouteConstraint, ...]]:
        return {name: tuple(getattr(module, "ROUTE_CONSTRAINTS", ()))
                for name, module in self.backends.items()}

    def ops(self) -> tuple[OpSpec, ...]:
        """Extension op specs every backend declares, deduplicated by name."""
        seen: dict[str, OpSpec] = {}
        for module in self.backends.values():
            for spec in getattr(module, "OPS", ()):
                if spec.name in seen and seen[spec.name] is not spec:
                    raise ValueError(f"extension op {spec.name!r} declared twice")
                seen[spec.name] = spec
        return tuple(seen.values())

    def resolve(self, plan: Mapping[str, str] | None,
                call_sites: Iterable[str]) -> dict[str, str]:
        """The backend of every call site under `plan`, validated."""
        provided = self.provided()
        binding.check_backends_provide(plan or {}, provided)
        routes = binding.resolve(plan, self.default, call_sites,
                                 known_call_sites=set().union(*provided.values()))
        binding.check_backends_provide(routes, provided)
        binding.validate(routes, self.constraints())
        return routes

    def op_table(self, routes: Mapping[str, str], scratch: Scratch) -> SimpleNamespace:
        """One wrapper per call site, built by the backend the route names."""
        selected: dict[str, set[str]] = {}
        for call_site, backend in routes.items():
            selected.setdefault(backend, set()).add(call_site)
        table: dict[str, Callable] = {}
        for backend, names in selected.items():
            built = self.backends[backend].make_wrappers(scratch, selected_names=set(names))
            for call_site in names:
                if call_site not in built:
                    raise KeyError(f"backend {backend!r} did not build {call_site!r}")
                table[call_site] = built[call_site]
        return SimpleNamespace(**table)

    def graph_contract(self, routes: Mapping[str, str]) -> dict[str, list[str]]:
        """The union of every routed backend's graph contract."""
        merged: dict[str, list[str]] = {"forbid": [], "require_one": []}
        active = set(routes.values())
        for name, module in self.backends.items():
            declare = getattr(module, "graph_contract", None)
            if declare is None or name not in active:
                continue
            contract = declare(routes)
            for key in merged:
                merged[key] += [p for p in contract.get(key, ()) if p not in merged[key]]
        return merged

    def atomic_groups(self, routes: Mapping[str, str]) -> tuple[frozenset[str], ...]:
        """Call sites that must be invoked together on `routes`: every constraint
        whose members all resolve to the declaring backend, overlapping groups merged."""
        groups: list[frozenset[str]] = []
        for backend, constraints in self.constraints().items():
            for constraint in constraints:
                if all(routes.get(name) == backend for name in constraint.members):
                    groups.append(frozenset(constraint.members))
        merged: list[set[str]] = []
        for group in groups:
            hit = [g for g in merged if g & group]
            for g in hit:
                merged.remove(g)
                group = frozenset(group | g)
            merged.append(set(group))
        return tuple(frozenset(g) for g in merged)


__all__ = ["Registry", "Scratch"]
