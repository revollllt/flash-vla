"""Combine the survey's raw JSON into per-shape bests and per-model sums.

Usage: python summarize.py <results dir>; writes summary.md and summary.json there.
"""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from vla_shapes import SHAPES, floor_us

Row = dict[str, str | int | float | bool]


def backend_label(row: Row) -> str:
    """lib:backend, with +tuned and +splitK<S> when they apply."""
    return (f"{row['lib']}:{row['backend']}" + ("+tuned" if row.get("autotuned") else "")
            + (f"+splitK{row['split_k']}" if row.get("split_k", 1) > 1 else ""))


def shape_class(m: int, k: int, n: int) -> str:
    """The survey's shape classes: the first rule that applies, else mid-M."""
    rules = [(n * k <= 1536 * 2048, "small-weight"), (k > n, "down (K>N)"),
             (m <= 64, "small-M"), (m >= 900, "large-M")]
    return next((name for applies, name in rules if applies), "mid-M")


def main() -> None:
    results_dir = Path(sys.argv[1])
    rows: list[Row] = [row for path in sorted(results_dir.glob("*.json"))
                       if not path.name.startswith("summary")
                       for row in json.loads(path.read_text())["rows"]]
    valid_gemms = [row for row in rows if row["op"] == "gemm" and "us" in row and row.get("ok", True)]
    quantize_us: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        timed = ([row["us"]] if row["op"] == "quantize" and "us" in row else []) + (
            [row["quantize_us"]] if row.get("quantize_us") else [])
        quantize_us.setdefault((row["shape"], row["fmt"]), []).extend(timed)

    per_shape = []
    lines = []
    for fmt in ("mxfp8", "nvfp4", "fp8_1d2d"):
        lines += [f"\n### {fmt}\n",
                  "| shape | M×K×N | calls | class | floor µs | best µs | ×floor | best backend "
                  "| quant µs | BF16 cuBLAS µs |",
                  "|---|---|---:|---|---:|---:|---:|---|---:|---:|"]
        for site in SHAPES:
            candidates = [row for row in valid_gemms if row["shape"] == site.name and row["fmt"] == fmt]
            if not candidates:
                continue
            best = min(candidates, key=lambda row: row["us"])
            bf16_us = next((row["us"] for row in valid_gemms
                            if row["shape"] == site.name and row["fmt"] == "bf16"), math.nan)
            floor, bound = floor_us(fmt, site.m, site.k, site.n)
            fastest_quantize = min(quantize_us.get((site.name, fmt)) or [math.nan])
            record = dict(fmt=fmt, shape=site.name, model=site.model, m=site.m, k=site.k, n=site.n,
                          count=site.count, cls=shape_class(site.m, site.k, site.n), floor_us=floor,
                          bound=bound, best_us=best["us"], best=backend_label(best),
                          quant_us=fastest_quantize, bf16_us=bf16_us)
            per_shape.append(record)
            lines.append(f"| {site.name} | {site.m}×{site.k}×{site.n} | {site.count} "
                         f"| {record['cls']} | {floor:.2f} ({bound[0]}) | {best['us']:.2f} "
                         f"| {best['us'] / floor:.2f} | {record['best']} | {fastest_quantize:.2f} "
                         f"| {bf16_us:.2f} |")

    per_model = {}
    lines += ["\n### Per observation (sum of count × time, ms)\n",
              "| model | format | floor | best library | ×floor | + unfused quantize | BF16 cuBLAS "
              "| best vs BF16 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in ("groot", "pi05"):
        for fmt in ("mxfp8", "nvfp4", "fp8_1d2d"):
            records = [record for record in per_shape
                       if record["model"] == model and record["fmt"] == fmt]
            if not records:
                continue
            floor_ms = sum(record["count"] * record["floor_us"] for record in records) / 1e3
            best_ms = sum(record["count"] * record["best_us"] for record in records) / 1e3
            quantize_ms = sum(record["count"] * record["quant_us"] for record in records
                              if not math.isnan(record["quant_us"])) / 1e3
            bf16_ms = sum(record["count"] * record["bf16_us"] for record in records
                          if not math.isnan(record["bf16_us"])) / 1e3
            per_model[f"{model}/{fmt}"] = dict(floor_ms=floor_ms, best_ms=best_ms,
                                               quant_ms=quantize_ms, bf16_ms=bf16_ms)
            lines.append(f"| {model} | {fmt} | {floor_ms:.2f} | {best_ms:.2f} "
                         f"| {best_ms / floor_ms:.2f} | {quantize_ms:.2f} | {bf16_ms:.2f} "
                         f"| {bf16_ms / best_ms:.2f}× |")
        lines.append("")

    lines += ["\n### Gap by class (count-weighted best / floor)\n",
              "| class | mxfp8 | nvfp4 |", "|---|---:|---:|"]
    for cls in ("small-M", "down (K>N)", "small-weight", "mid-M", "large-M"):
        cells = []
        for fmt in ("mxfp8", "nvfp4"):
            records = [record for record in per_shape
                       if record["fmt"] == fmt and record["cls"] == cls]
            weighted_best = sum(record["count"] * record["best_us"] for record in records)
            weighted_floor = sum(record["count"] * record["floor_us"] for record in records)
            cells.append(f"{weighted_best / weighted_floor:.2f}" if records else "-")
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")

    text = "\n".join(lines)
    (results_dir / "summary.md").write_text(text + "\n")
    (results_dir / "summary.json").write_text(
        json.dumps(dict(per_shape=per_shape, per_model=per_model), indent=1))
    print(text)


if __name__ == "__main__":
    main()
