# Engineering tests

Run the tests for the behavior being changed:

| Behavior | Tests |
| --- | --- |
| Runtime and workload identity | `test_binding.py`, `test_cuda_stream_graph.py`, `test_identity.py` |
| Checkpoint loading and asset isolation | `test_checkpoint.py`, `test_checkpoint_compatibility.py`, `test_target_assets.py`, `test_source_engines.py` |
| Latency protocol and device selection | `test_latency.py`, `test_environment.py` |
| Profiling and saved-result interpretation | `test_profiling.py`, `test_kernel_trace.py`, `test_results.py` |

```bash
python -m pytest -q tests/test_latency.py tests/test_environment.py
python -m tests.targets --target h100/pi05
```

`python -m pytest -q` runs the small CPU suite; CUDA-dependent cases skip when no
GPU is available. CI runs this suite. Kernel or capture changes need the affected
GPU check; documentation and test-only edits do not require model benchmarks.

```bash
python -m pytest -q tests/test_cuda_stream_graph.py
python -m tests.pi05.fold
python -m tests.pi05.tokenize --help
python -m tests.tile_sm90
```

The tile checks require a supported GPU. Model output accuracy and official
references belong to [eval](../eval/README.md); model and kernel timing belongs
to [benchmarks](../benchmarks/README.md). Historical Campaign tests were removed
with their controller.
