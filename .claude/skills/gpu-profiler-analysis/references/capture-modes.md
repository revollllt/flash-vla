# Capture modes

Use the least invasive mode that answers the question.

| Mode | Tool | Purpose | Expensive flags |
| --- | --- | --- | --- |
| summary | CUPTI/events | latency and regression numbers | none |
| timeline | Torch or Nsight Systems | graph, stream, copy, overlap | CPU + CUDA/NVTX tracing |
| mapping | Torch or Nsight Systems | CPU/Python correlation | `with_stack`, shapes, NVTX |
| kernel-detail | Nsight Compute | counters, roofline, occupancy | replay and multiple metric passes |

Warmup must complete before capture. Keep the active capture short and
representative. Do not compare profiler wall time with a benchmark baseline.

For CUDA Graph workloads, distinguish graph-level timeline evidence from
node-level evidence. Node-level tracing can carry substantially more overhead;
use it only when the graph as a whole does not identify the bottleneck.

Source mapping is optional. A formal graph-on trace is the performance truth;
an eager or lower-fusion mapping trace may be used later to recover Python
locations and then joined by kernel name, shape, and stage.
