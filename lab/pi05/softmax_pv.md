# Fixed full-row softmax + PV lab prototype

Only the CPU screen's 16x32x64 / four-warp / 200-CTA mapping is implemented.
The hand-written warp softmax computes each complete row from FP32 logits and
the finite runtime BF16 mask, then explicitly rounds P to BF16 in shared memory.
CuTe SM80 BF16/FP32 MMA consumes P with three K64 V stages and stores BF16 output.
No online normalization or deferred probability rounding is used.

The prototype follows the local CuTe sgemm_sm80 tutorial's partition/copy/MMA
interfaces, using the installed SM75 LDSM and SM80 cp.async zero-fill atoms.
It does not copy a new GEMM library or change production/vendor files. Shared P
uses the tutorial's swizzle, V has an N-contiguous swizzle, and all writers/readers
use the same layouts. Two CTA barriers protect the mask/P setup; V ring stages
are published after cp.async wait and recycled only after all MMA readers finish.
The intended budget is 46 KiB shared. Registers and actual dispatch are unmeasured
until compilation/check. A different row reduction tree is explicitly part of
this candidate, so P is not expected to be bitwise equal.

The existing nine Q/K/V/mask snapshots were captured at seed 42 from belt-cup,
before out=Q overwrote Q (captured revision bf2a9ba, retained metadata preserved).
The probe computes the current production QK once per pair outside timing.
No model reload is needed, and these older real snapshots are not represented
as a new capture from current production.

The fixed protocol is:
1. Compile this source with sm_120a and --fmad=false, inspect ptxas resources.
2. Run check on all nine pairs. Compare full BF16 diagnostic P and output from
   the actual timed branch against native softmax + torch.mm using existing
   shallow tolerances. The diagnostic P store is disabled in timing.
3. Only if those pass, run one ABBA over all nine pairs, 16 sequence repetitions
   per fresh graph, 5 warm repeats and 30 raw samples/leg. Both routes use the
   same logits, mask, V and output addresses, and both restore the saved Q into
   output before every call. QK, snapshot loading, allocations and compilation
   are outside the boundary. The working set is warm; no cold-cache claim.
4. Stop on implementation/numerical failure, negative gain, or gain not exceeding
   observed drift. No second tile or local parameter sweep follows.

The check's single resource trace runs in a separate process from timing.
The root owns subsequent model validation and deployed timing if warranted.

Commands from this worktree after the exclusive GPU/NVCC window is granted:

```sh
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
mkdir -p artifacts/rtx5090-pi05/softmax-pv
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 --shared -Xcompiler -fPIC \
  --expt-relaxed-constexpr --fmad=false -Xptxas=-v \
  -gencode arch=compute_120a,code=sm_120a \
  -I/home/ubuntu/flash-vla/third_party/cutlass/include \
  lab/pi05/softmax_pv.cu -o artifacts/rtx5090-pi05/softmax-pv/softmax_pv.so
/home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.softmax_pv \
  --phase check --snapshot artifacts/rtx5090-pi05/attention-qk-nine-pairs.safetensors \
  --library artifacts/rtx5090-pi05/softmax-pv/softmax_pv.so \
  --output results/rtx5090-pi05/gpt6-attention-softmax-pv-screen/check.json
/home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.softmax_pv \
  --phase time --snapshot artifacts/rtx5090-pi05/attention-qk-nine-pairs.safetensors \
  --library artifacts/rtx5090-pi05/softmax-pv/softmax_pv.so \
  --output results/rtx5090-pi05/gpt6-attention-softmax-pv-screen/abba.json
```
