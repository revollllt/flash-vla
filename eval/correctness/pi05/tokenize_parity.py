"""Equivalence gate for the Pi0.5 table tokenizer against OpenPI's SentencePiece path.

`models.pi05.tokenize` replaces a per-inference `sp.encode` of the whole prompt
with a 257-entry table lookup. That is only legitimate if the two produce the
same token ids for every input, so this gate asserts exactly that -- not a
tolerance, an equality.

Two independent properties are checked, because they can fail for unrelated
reasons:

1. `discretize` must equal `np.digitize(state, linspace(-1,1,257)[:-1]) - 1`.
   This is a numerical claim about float arithmetic and is checked on a dense
   sweep that includes every bin edge and its neighbourhood, plus the non-finite
   values, in float32 and float64.
2. `encode` must equal `reference` token-for-token. This is a claim about
   SentencePiece segmentation not crossing value boundaries, and it is checked
   over a matrix of task strings crossed with state distributions -- including
   every one of the 257 bin values, and a prompt long enough to exercise the
   truncation branch.

No GPU and no OpenPI checkout are needed; only the PaliGemma SentencePiece model
(gs://big_vision/paligemma_tokenizer.model), via --tokenizer or
$PALIGEMMA_TOKENIZER.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from flash_vla.models.pi05.tokenize import BIN_MAX, BIN_MIN, Pi05Tokenizer, discretize

_BINS = np.linspace(-1.0, 1.0, 256 + 1)[:-1]

PROMPTS = [
    "fold the towel",
    "pick up the plate and put it in the sink",
    "",
    "a",
    "put_the_red_block_on_the_green_block",  # underscores, cleaned to spaces
    "line one\nline two",                    # newline, cleaned to a space
    "grasp the handle " * 60,                # long enough to force truncation
]


def check_discretize() -> dict[str, object]:
    """Compare the closed form against np.digitize on a sweep that hits every edge."""
    sweep = np.concatenate([
        np.linspace(-1.5, 1.5, 200001),
        _BINS, np.nextafter(_BINS, -np.inf), np.nextafter(_BINS, np.inf),
        _BINS - 1e-9, _BINS + 1e-9,
        np.array([-1.0, 0.0, -0.0, 1.0, 0.9921875, -1e-30, 1e-30,
                  np.inf, -np.inf, np.nan]),
        np.random.default_rng(0).standard_normal(200000),
    ])
    result: dict[str, object] = {"samples": int(sweep.size)}
    for name, dtype in (("float32", np.float32), ("float64", np.float64)):
        values = sweep.astype(dtype)
        reference = np.digitize(values, bins=_BINS) - 1
        mismatches = int(np.count_nonzero(reference != discretize(values)))
        result[name + "_mismatches"] = mismatches
    result["distinct_bins"] = int(np.unique(np.digitize(sweep, bins=_BINS) - 1).size)
    return result


def _states(dim: int, seed: int) -> list[tuple[str, np.ndarray]]:
    """Random, extreme, and exhaustive-per-value state vectors."""
    rng = np.random.default_rng(seed)
    cases = [(f"random[{index}]", (rng.standard_normal(dim)
                                   * rng.uniform(0.1, 1.5)).astype(np.float32))
             for index in range(200)]
    cases += [
        ("below_range", np.full(dim, -2.0, np.float32)),
        ("at_low_edge", np.full(dim, -1.0, np.float32)),
        ("zero", np.zeros(dim, np.float32)),
        ("at_high_edge", np.full(dim, 1.0, np.float32)),
        ("above_range", np.full(dim, 2.0, np.float32)),
        ("ramp", np.linspace(-1.2, 1.2, dim).astype(np.float32)),
        ("float64", rng.standard_normal(dim)),
    ]
    # Every bin value, broadcast to every slot: covers the 1-, 2- and 3-digit
    # token layouts and the negative-sign layout in all positions.
    for value in range(BIN_MIN, BIN_MAX + 1):
        x = -1.5 if value < 0 else (value - 128) / 128.0 + 1.0 / 512.0
        cases.append((f"all_bin_{value}", np.full(dim, x, np.float32)))
    return cases


def check_encode(tokenizer: Pi05Tokenizer, dim: int, seed: int) -> dict[str, object]:
    """Compare the table path against the SentencePiece path on the case matrix."""
    mismatches: list[str] = []
    lengths: list[int] = []
    truncated = 0
    cases = _states(dim, seed)
    for prompt in PROMPTS:
        tokenizer.set_task(prompt)
        for name, state in cases:
            tokens, mask = tokenizer.encode(state)
            tokens, mask = tokens.copy(), mask.copy()
            reference_tokens, reference_mask = tokenizer.reference(prompt, state)
            if not (np.array_equal(tokens, reference_tokens)
                    and np.array_equal(mask, reference_mask)):
                if len(mismatches) < 8:
                    mismatches.append(f"{prompt[:24]!r} / {name}")
            valid = int(mask.sum())
            lengths.append(valid)
            truncated += valid == tokenizer.max_token_len

    return {
        "cases": len(PROMPTS) * len(cases),
        "mismatches": len(mismatches),
        "first_mismatches": mismatches,
        "truncating_cases": truncated,
        "valid_tokens_min": min(lengths),
        "valid_tokens_max": max(lengths),
        "valid_tokens_mean": round(float(np.mean(lengths)), 1),
        "max_token_len": tokenizer.max_token_len,
    }


def measure(tokenizer: Pi05Tokenizer, dim: int, iterations: int = 5000) -> dict[str, float]:
    """Host latency of both paths, the reason the table exists."""
    prompt = PROMPTS[1]
    tokenizer.set_task(prompt)
    state = (np.random.default_rng(3).standard_normal(dim) * 0.5).astype(np.float32)

    timings = {}
    for name, call in (("table_us", lambda: tokenizer.encode(state)),
                       ("sentencepiece_us", lambda: tokenizer.reference(prompt, state))):
        for _ in range(200):
            call()
        start = time.perf_counter()
        for _ in range(iterations):
            call()
        timings[name] = round((time.perf_counter() - start) / iterations * 1e6, 2)
    return timings


def run(tokenizer_path: str | None, dim: int = 32, seed: int = 0) -> dict[str, object]:
    """Run both checks plus the timing, print the report, and return it."""
    # The long prompt deliberately overflows max_token_len on every state, so the
    # per-call truncation warning is expected here; the count is in the report.
    logging.getLogger("flash_vla.models.pi05.tokenize").setLevel(logging.ERROR)

    tokenizer = Pi05Tokenizer(tokenizer_path)
    report = {
        "discretize": check_discretize(),
        "encode": check_encode(tokenizer, dim, seed),
        "host_latency": measure(tokenizer, dim),
    }
    report["passed"] = (
        report["discretize"]["float32_mismatches"] == 0
        and report["discretize"]["float64_mismatches"] == 0
        and report["encode"]["mismatches"] == 0
    )
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=None,
                        help="paligemma_tokenizer.model (default: $PALIGEMMA_TOKENIZER)")
    parser.add_argument("--dim", type=int, default=32, help="state dimensionality")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    return 0 if run(args.tokenizer, dim=args.dim, seed=args.seed)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
