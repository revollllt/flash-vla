# sm120 bring-up probes

Two probes behind the RTX 5090 hardware axis. Both answer a question that was
otherwise going to be assumed, and both are cheap enough to re-run whenever the
toolchain or driver moves. Results and their consequences live in
[the measured layer](../../src/flash_vla/hardware/nvidia/rtx5090/measured/README.md).

```bash
CUDA_HOME=/path/to/cuda python3 lab/sm120/ptx_support.py
nvcc -O2 -gencode arch=compute_120a,code=sm_120a -o /tmp/cluster_dsmem lab/sm120/cluster_dsmem.cu && /tmp/cluster_dsmem
```

**`ptx_support.py`** — assembles one minimal `.entry` per PTX instruction against
sm_90a, sm_120 and sm_120a and reports which target accepts it. `ptxas` is the
oracle. A case that fails for any reason other than a target refusal is `BROKEN`
and the table must not be read until the count is zero. Arch-independent: the
target list and the case list are both data.

**`cluster_dsmem.cu`** — launches a thread-block cluster, has every rank write
into rank 0's shared memory through `map_shared_rank`, barriers, and checks the
sum; then walks `cudaFuncSetAttribute` to find the real shared-memory opt-in
ceiling. Exists because ptxas accepting `barrier.cluster` says nothing about
whether a cluster can be placed, and the two answers were not the same as the
ones in circulation.

**`tilelang_sm120.py`** — the go/no-go for any TileLang-backed Target on this
device: compiles and runs a reduction kernel, a tensor-core matmul with warp
specialisation off, and the same matmul with it on, comparing each against
torch. The third is the one with something to prove, since warp specialisation
on Hopper uses `setmaxnreg` and ptxas refuses that on plain `sm_120`.

**`stack_check.py`** — bf16 matmul, CUDA graph capture/replay, and a CUPTI
capture. The graph check is the load-bearing one: the runtime replays a captured
graph, so capture failing would invalidate the execution model rather than one
kernel.
