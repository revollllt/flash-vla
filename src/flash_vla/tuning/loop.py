"""The sweep loop, with the backend-specific parts injected.

One candidate at a time: skip it if infeasible, build it, optionally check it,
time it inside a CUDA graph with cold reads, rank what survived. That skeleton
is the same for TileLang and for hand-written CUDA, so it lives here; the two
things that are not the same -- how a candidate becomes a callable, and how a
callable is launched -- arrive as `build` and `invoke`.

Timing is always `graph_time_cold`. Eager timing cannot resolve these kernels:
at Pi0's decoder shapes the launch overhead is several times the kernel, so an
eager sweep ranks launch noise. Three production configs were wrong in exactly
that way and survived an eager benchmark.

The module is `loop`, not `sweep`, so that the exported `sweep()` function does
not shadow its own submodule on the package -- `import flash_vla.tuning.sweep`
would otherwise hand back the function.

Failures are counted by category and the first exception of each is kept. That
matters most on a device this backend has never run on, where a whole axis can
be illegal and every candidate fails: a bare count of 200 compile failures says
nothing, while the first traceback usually says everything. Counting rather than
raising is still the right default, because some failures are expected -- the
fused dual-GEMM gate rejects the no-warp-specialization pipeline plan at compile
time, and a sweep over that flag has to tolerate it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from flash_vla.runtime.cuda import graph_time_cold

CATEGORIES = ("infeasible", "compile_failed", "incorrect", "timing_failed")


@dataclass
class SweepResult:
    """What a sweep measured, and what it could not."""

    label: str
    results: list[dict] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    errors: dict[str, BaseException] = field(default_factory=dict)

    @property
    def best(self) -> dict | None:
        """Fastest measured candidate, or None if nothing survived."""
        return self.results[0] if self.results else None


def sweep(candidates: list[dict], build: Callable[[dict], Any],
          invoke: Callable[[Any, int], Any], *, feasible: Callable[[dict], bool] | None = None,
          correct: Callable[[Any], bool] | None = None, label: str = "sweep",
          n_inner: int = 48, reps: int = 40, verbose: bool = True) -> SweepResult:
    """Measure every candidate, fastest first.

    candidates      one dict per point in the tuning space (see `space.grid`)
    build(c)        compile candidate `c`; raising counts as a compile failure
    invoke(k, i)    issue exactly one launch of built kernel `k` against the
                    i-th input set -- distinct sets keep the reads cold
    feasible(c)     optional pre-compile filter, e.g. a shared-memory bound.
                    Checked first so an impossible candidate costs nothing
    correct(k)      optional numerical check; failures are dropped, not ranked.
                    Never skip this on a re-tune: some tilings are wrong rather
                    than slow, and produce garbage instead of an error
    """
    results: list[dict] = []
    counts = {category: 0 for category in CATEGORIES}
    errors: dict[str, BaseException] = {}

    def failed(category: str, error: BaseException | None = None) -> None:
        counts[category] += 1
        if error is not None:
            errors.setdefault(category, error)

    for candidate in candidates:
        if feasible is not None and not feasible(candidate):
            failed("infeasible")
            continue
        try:
            built = build(candidate)
        except Exception as error:
            failed("compile_failed", error)
            continue
        if correct is not None:
            try:
                passed = correct(built)
            except Exception as error:
                failed("incorrect", error)
                continue
            if not passed:
                failed("incorrect")
                continue
        try:
            microseconds = graph_time_cold(lambda i: invoke(built, i), n_inner=n_inner, reps=reps)
        except Exception as error:
            failed("timing_failed", error)
            continue
        results.append({"us": round(microseconds, 3), **candidate})

    results.sort(key=lambda entry: entry["us"])
    outcome = SweepResult(label=label, results=results, counts=counts, errors=errors)
    if verbose:
        report(outcome)
    return outcome


def report(outcome: SweepResult, top: int = 6, by: str | None = None) -> None:
    """Print the ranking, then the first failure of each category.

    `by` breaks the winner down per value of one axis -- `by="warp_spec"` is how
    a TileLang sweep shows the best warp-specialized and non-warp-specialized
    config side by side, which is the comparison that decides the flag.
    """
    dropped = ", ".join(f"{count} {category.replace('_', ' ')}"
                        for category, count in outcome.counts.items() if count)
    print(f"[{outcome.label}] {len(outcome.results)} measured"
          + (f" ({dropped})" if dropped else ""))

    for entry in outcome.results[:top]:
        config = {key: value for key, value in entry.items() if key != "us"}
        print(f"    us={entry['us']:8.3f}  {config}")

    if by is not None and outcome.results:
        for value in sorted({entry[by] for entry in outcome.results if by in entry}, key=str):
            fastest = min(entry["us"] for entry in outcome.results if entry.get(by) == value)
            print(f"    best {by}={value}: {fastest}us")

    for category, error in outcome.errors.items():
        print(f"    first {category}: {type(error).__name__}: {error}")


__all__ = ["CATEGORIES", "SweepResult", "report", "sweep"]
