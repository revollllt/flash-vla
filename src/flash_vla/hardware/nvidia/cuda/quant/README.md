# Block quantization device functions (Blackwell)

`block_quant.cuh` quantizes one thread's eight values to MXFP8 (E4M3, UE8M0 per
32) or NVFP4 (E2M1, UE4M3 per 16, FP32 encode scale). It also gives the scale's
offset in the 128x4 swizzled layout that Blackwell's block-scaled GEMMs read.
The arithmetic is FlashInfer 0.7.0's default fast path, including its
fast-math flush of subnormals, so a producer that calls these functions writes
exactly what `flashinfer.mxfp8_quantize` / `fp4_quantize` would.

Include root: `src/flash_vla/hardware/nvidia/cuda` (`#include
"quant/block_quant.cuh"`). Namespace: `flash_vla_quant`. Needs a Blackwell
target (`sm_100f`, `sm_110f`, `sm_120f` or an arch-specific `sm_1xxa`), because
Hopper has no `cvt.e2m1x2`.

Caller contract: thread t owns elements [8j, 8j + 8) of a row. The 4 (MXFP8) or
2 (NVFP4) threads that share a scale block are consecutive, aligned lanes, and
every lane of the warp calls the function (the block max is a shuffle). One
lane per block passes the scale's address; the others pass `nullptr`.

Users: [`quant_ops`](../../quant_ops/README.md) (norm and activation producers).
Validation: `tests/test_quant_ops.py`, `lab/quantization/check_quant_ops.py`.
