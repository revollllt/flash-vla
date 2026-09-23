"""Paired LIBERO task success of two policy variants run under one protocol.

Every episode of `eval.libero` is fixed by its suite, task and trial: the
initial state, the simulator seed and the policy noise all derive from
`seed + task_id * 1000 + trial`. Two variants run with the same protocol
therefore face identical episodes, and their difference is read per episode:
the pairs where only one variant succeeds, tested with the exact McNemar test.
A reference run (the official OpenPI model) is shown beside them for scale.

    python lab/quantization/libero_paired.py --baseline bf16 --candidate mxfp8 \\
        --dir artifacts/libero-mxfp8 --reference-dir /path/to/official --reference official \\
        --out results/pi05-rtx5090/mxfp8-llm-ffn/libero
"""
from __future__ import annotations

import argparse
import json
from math import comb, sqrt
from pathlib import Path
from typing import TypedDict

SUITES = {"spatial": "libero_spatial", "object": "libero_object", "goal": "libero_goal",
          "long": "libero_10"}
PROTOCOL_KEYS = ("suite", "seed", "trials_per_task", "task_ids", "replan_steps")


class Episode(TypedDict):
    task_id: int
    trial: int
    success: bool


class SuiteRow(TypedDict):
    suite: str
    episodes: int
    baseline: int
    candidate: int
    reference: int | None
    both: int
    baseline_only: int
    candidate_only: int
    mcnemar_p: float


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval of a success rate."""
    rate = successes / n
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def outcomes(path: Path, protocol: dict[str, object] | None) -> tuple[dict[tuple[int, int], bool],
                                                                      dict[str, object]]:
    """(task, trial) -> success of one complete run, and its protocol, which must
    equal `protocol` when one is given."""
    report = json.loads(path.read_text())
    if report["status"] != "complete":
        raise ValueError(f"{path} is {report['status']}, not complete")
    own = {key: report[key] for key in PROTOCOL_KEYS}
    if protocol is not None and own != protocol:
        raise ValueError(f"{path} ran another protocol: {own} against {protocol}")
    episodes: list[Episode] = report["episodes"]
    return {(e["task_id"], e["trial"]): e["success"] for e in episodes}, own


def compare(args: argparse.Namespace) -> list[SuiteRow]:
    rows: list[SuiteRow] = []
    for name, suite in SUITES.items():
        baseline, protocol = outcomes(args.dir / f"{name}-{args.baseline}.json", None)
        candidate, _ = outcomes(args.dir / f"{name}-{args.candidate}.json", protocol)
        reference_path = (args.reference_dir / f"{name}-{args.reference}.json"
                          if args.reference_dir is not None else None)
        reference = (outcomes(reference_path, protocol)[0] if reference_path is not None
                     else None)
        if set(baseline) != set(candidate):
            raise ValueError(f"{suite}: the two variants ran different episodes")
        both = sum(baseline[k] and candidate[k] for k in baseline)
        baseline_only = sum(baseline[k] and not candidate[k] for k in baseline)
        candidate_only = sum(candidate[k] and not baseline[k] for k in baseline)
        rows.append(SuiteRow(suite=suite, episodes=len(baseline),
                             baseline=sum(baseline.values()), candidate=sum(candidate.values()),
                             reference=sum(reference.values()) if reference is not None else None,
                             both=both, baseline_only=baseline_only,
                             candidate_only=candidate_only,
                             mcnemar_p=mcnemar_exact(baseline_only, candidate_only)))
    baseline_only = sum(r["baseline_only"] for r in rows)
    candidate_only = sum(r["candidate_only"] for r in rows)
    references = [r["reference"] for r in rows]
    total = SuiteRow(suite="all", episodes=sum(r["episodes"] for r in rows),
                     baseline=sum(r["baseline"] for r in rows),
                     candidate=sum(r["candidate"] for r in rows),
                     reference=(sum(x for x in references if x is not None)
                                if None not in references else None),
                     both=sum(r["both"] for r in rows), baseline_only=baseline_only,
                     candidate_only=candidate_only,
                     mcnemar_p=mcnemar_exact(baseline_only, candidate_only))
    return [*rows, total]


def render(rows: list[SuiteRow], args: argparse.Namespace) -> str:
    def cell(successes: int, n: int) -> str:
        low, high = wilson(successes, n)
        return f"{successes}/{n} = {100 * successes / n:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"

    lines = [f"| suite | {args.baseline} | {args.candidate} | Δ (pp) | both succeed "
             f"| only {args.baseline} | only {args.candidate} | McNemar p | {args.reference} |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        n = r["episodes"]
        delta = 100 * (r["candidate"] - r["baseline"]) / n
        reference = cell(r["reference"], n) if r["reference"] is not None else "—"
        lines.append(f"| {r['suite']} | {cell(r['baseline'], n)} | {cell(r['candidate'], n)} "
                     f"| {delta:+.1f} | {r['both']} | {r['baseline_only']} "
                     f"| {r['candidate_only']} | {r['mcnemar_p']:.3f} | {reference} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", type=Path, required=True, help="holds <suite>-<variant>.json")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--reference", default="official")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = compare(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "paired.json").write_text(json.dumps(rows, indent=1) + "\n")
    (args.out / "paired.md").write_text(render(rows, args))
    print(render(rows, args))


if __name__ == "__main__":
    main()
