# Fixed full-row softmax + PV lab prototype

Result: this fixed layout passes all nine real P/output checks but is slower
at the complete reset-inclusive softmax+PV boundary. The experiment stops;
no production route or vendor change was made.

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

## First build outcome: host-stub compilation failure

The fixed source was committed as 4aad6eb before compilation. Device ptxas
completed with 166 registers/thread, 47104 bytes shared, one barrier, and no
reported stack or spills. The host stub then failed: the anonymous namespace
containing Element became ambiguous with CuTe's imported anonymous namespace.
The compiler exited 1 and produced no loadable candidate library.

The serial GPU/NVCC window was released. No kernel launch, actual pair/P
comparison, resource trace, or ABBA followed. The error has not been retried.
It is a source/host-stub issue, not numerical or performance evidence about the
mapping. A named namespace is a possible mechanical repair, but has not been
applied or validated in this attempt.

The retained compile-failure.txt records the diagnostic. Production and vendor
files remain untouched.

## Completed fixed experiment: locally rejected

One mechanical namespace repair (88bd92a) resolved the host-stub ambiguity.
The kernel mapping, flags, numerical expressions and protocol stayed fixed.
The same source then compiled: 166 registers/thread, 47104 bytes shared, one
barrier, zero stack/spill stores/spill loads.

All nine actual pairs pass existing shallow tolerances. Diagnostic BF16 P:
worst rel_rms=3.9797604e-6, max_abs=3.0517578e-5,
minimum cosine=0.9999999999920808; one pair is exact. Timed-branch output:
worst rel_rms=0.0027467913, max_abs=0.0625,
minimum cosine=0.9999962292141567; none are bitwise equal.
All masks include the actual finite BF16 value -3.00405527047391e38 and zero.

The separate metadata trace confirms one fused kernel at grid(25,8,1),
block(128,1,1), with 166 registers and 47104 B shared. The occupancy query
returns two active CTAs/SM limited by shared memory (allocated shared/CTA
48128 B and allocated registers/CTA 21504). The profiler's grid/SM average
is not evidence of a physical CTA distribution or active-warps history.

| Leg | Median us per nine calls |
|---|---:|
| A1 native softmax + torch PV | 78.172002 |
| B1 fused softmax/PV | 94.750002 |
| B2 fused softmax/PV | 94.780002 |
| A2 native softmax + torch PV | 78.258000 |

Mean leg-median disadvantage is 16.550001 us/nine calls = 1.838889 us/call.
A/B drift is only 0.085998/0.030000 us per nine calls. Both routes used the
same immutable score/mask/V addresses and mutable output addresses, and both
copied the same saved Q into output before each call. This is one warm-cache
local sequence, not a full-model latency result.

The fixed candidate is locally rejected. The comparison does not isolate the
cost of duplicated softmax from shared layout, registers or V staging. No
additional tile, stage, timer series or end-to-end experiment followed.
The process exited 0 and the GPU/NVCC window was released immediately.

[check.json](../../results/rtx5090-pi05/gpt6-attention-softmax-pv-screen/check.json)
contains all nine numerical comparisons and actual kernel metadata.
[abba.json](../../results/rtx5090-pi05/gpt6-attention-softmax-pv-screen/abba.json)
contains all 120 raw samples and drift calculations.
[compile-success.txt](../../results/rtx5090-pi05/gpt6-attention-softmax-pv-screen/compile-success.txt)
preserves the successful compiler resource log. Raw trace, logs and the lab
library remain in their recorded worktree paths; the raw trace is not committed.
