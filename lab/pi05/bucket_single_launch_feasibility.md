# Single-launch prefix buckets: stop after CPU feasibility review

The retained 023 route uses two captured cfg0 bucket launches per GEMM.
One executes M896 or M968; the other returns uniformly before its mainloop.
This review inspected main `7869a5a` and the existing 023 native binary.
No candidate source, compilation, GPU execution, or heavy import was performed.

## Semantics: feasible for this fixed route

A single launch could carry both original Params, the mask pointer, and both
host-computed grid.x bounds. Launch max(short.x, long.x), select Params from
`mask[896]`, then return for raw `blockIdx.x >= selected_grid_x` before calling
the existing `GemmKernel::invoke`. Keep the selected bucket's disjoint workspace.

The relevant vendor paths are under `third_party/cutlass/include/cutlass/`:

- `gemm/threadblock/threadblock_swizzle_streamk.h:627`: the host grid is
  `(get_num_blocks(), 1, batch_count)`, including Stream-K padding/reduction.
- The same file at line 639 defines `device_num_blocks()` using `gridDim.x`,
  but searching all CUTLASS includes found no call to that method.
- Lines 654 and 704 derive batch and mapped block IDs from `blockIdx.z/x`.
  The fixed native arguments use kGemm with batch_count=1, so both grids have
  y=z=1; this finding does not generalize to arbitrary batched layouts.
- `gemm/kernel/gemm_universal_streamk.h:1003` selects DP, Stream-K, reduction,
  and padding work from the selected Params.block_mapping. Its loop advances
  tiles using that mapping's avail_sms, not the physical launch width.
- The same file at lines 809 and 885 waits/resets selected workspace barriers;
  line 1122 exposes the original invoke entry.

The guard must use the original physical grid bound before index remapping.
Trimming to logical tile count or testing a remapped block ID can remove
required Stream-K peers/reduction blocks. Every surviving CTA must select the
same Params, and the mask must remain stable throughout the launch.

A distinct combined entry still needs its own dynamic-shared attribute and
occupancy query. Both Params must be constructed using that entry's actual
occupancy; copying Params prepared for another entry assumes an unmeasured
property. `gemm/device/gemm_universal_base.h:129` initializes the exact Kernel2
symbol; lines 259 and 462 construct Params and launch it. The same cfg0 mainloop
and epilogue, including beta=1 C=D residual semantics, can remain intact.

## Parameter size is not the limiting evidence

CPU-only `cuobjdump --dump-elf` on the existing
`/home/ubuntu/flash-vla/.cache/cuda_ext/rtx5090_pi05_cutlass_backbone/libcutlass_backbone.so`
reported both KPARAM_INFO and CBANK_PARAM_SIZE:

| Existing entry | Parameter bytes |
| --- | ---: |
| Original cfg0 Kernel2 | 0x1e0 = 480 |
| Current BucketKernel | 0x1f0 = 496 |

Two original Params, one pointer, and two int32 bounds would total about 976
bytes before confirming a concrete layout. This is below even the older
4096-byte parameter limit. CUDA 12.1 expanded the limit to 32764 bytes on
Volta and later GPUs; this project uses CUDA 13.1/sm120a.
See [NVIDIA's parameter-limit announcement](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/).

Capacity does not imply free selection. Taking a dynamically selected address
inside a by-value kernel parameter may introduce local copies or additional
parameter loads. Existing 023 cfg0 and BucketKernel compile with 254 registers,
zero spills, 98304 bytes shared, and 128 threads. A combined entry's register,
spill, code-size, and occupancy effects remain unknown without compilation.
The overview's reported zero active blocks/SM is an invalid opt-in-shared-memory
estimate and is not used as occupancy evidence here.

## Decision

023's measured inactive launches total **82.850 microseconds over 51 calls**,
about 0.27% of the approximately 30.3 ms deployed runtime. This is a profiled
gross upper bound, not an expected end-to-end gain. Active short gate/up grids
are 850 blocks versus inactive long grids of 1020; down uses 170 for both.
A max-grid launch still leaves 170 extra early-return CTAs for each short
gate/up and adds Params selection to all 51 active GEMMs.

The small remaining gross opportunity and unmeasured costs do not justify
implementing this direction now. Stop at this CPU review; retain the accepted
two-launch route. No native candidate or extra measurement was produced.
