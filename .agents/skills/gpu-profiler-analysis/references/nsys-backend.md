# Nsight Systems backend

Use Nsight Systems for application-level CUDA timelines. The default wrapper
captures CUDA, NVTX, and OS runtime events, keeps CUDA Graphs at graph
granularity, and exports the native `.nsys-rep` plus optional SQLite/JSON Lines.

Record the exact command and graph granularity in `manifest.json`. Add NVTX
ranges for stage boundaries when the workload has multiple graphs or phases.
Use node granularity only when graph-level evidence cannot distinguish the
problem; it can increase overhead substantially.

Nsight Systems artifacts are not converted to Chrome JSON. Use the Nsight UI or
its SQLite/JSON Lines exports for post-processing, while Torch traces remain the
Perfetto-compatible portable lane.
