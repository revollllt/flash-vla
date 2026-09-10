"""Defaults for direct model and kernel measurements."""

LATENCY_METRICS = ("chunk_latency", "device_latency", "host_time", "segment_latency", "overhead")
STATISTICS = ("min", "median", "p99")
LATENCY_DEFAULTS = {
    "metrics": LATENCY_METRICS,
    "statistics": STATISTICS,
    "reps": 100,
    "warmup": 5,
    "soak_s": 0,
    "p99_min_reps": 100,
    "clocks": "unlocked",
    "deltas": "independent_versions",
    "primary_statistic": "median",
}
