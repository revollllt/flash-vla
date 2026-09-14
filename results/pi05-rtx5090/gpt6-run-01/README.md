# Pi0.5 on RTX 5090 — gpt6 run 01

Active optimization run on branch `gpt6-pi05-5090`, starting at `5ac75bc`.

Workload: converted kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha weights; bf16; batch 1; 3 x 224 x 224 images; 200 prompt slots; chunk 50; 18 layers; 10 denoising steps. Timing input seed 42. The existing official oracle uses its saved seed-0 fixture, so correctness and timing fixtures are identified separately.

Environment: RTX 5090 (170 SMs, 96 MiB L2), driver 580.142, torch 2.13.0+cu130, Triton 3.7.1; CUDA 13.1 for native builds. Clocks unlocked, unchanged. Machine paths are configured by the ignored `artifacts/rtx5090-pi05/gpt6-env.sh`. The previously absent tokenizer was obtained from OpenPI's documented PaliGemma asset before the first valid measurement; oracle token IDs agree.

Measurements use the workflow's existing 5 warmup / 100 repetitions in a fresh process per version, initial capture only. End-to-end wall time includes input staging, host processing, graph replay and final synchronization, excluding load/capture. Timing runs without a profiler, serially on this GPU. Xorg is present; there were no other compute processes before the initial run.

## Commands

Source the ignored environment file. CHECKPOINT resolves to the converted belt-cup directory. Each raw measurement records resolved options.

```sh
. artifacts/rtx5090-pi05/gpt6-env.sh
.venv/bin/python -m benchmarks latency --target rtx5090/pi05 --plan shipped --seed 42 --option converted_checkpoint="$CHECKPOINT" --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha --out results/pi05-rtx5090/gpt6-run-01/measurements/NNN.json
.venv/bin/python -m eval.pi05.parity compare --oracle artifacts/rtx5090-pi05/oracle-belt-cup --checkpoint "$CHECKPOINT" --checkpoint-id kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha --target rtx5090/pi05 --plan shipped
```

## Evidence and current uncertainty

- Initial median: 59.5779 ms, min 59.4828, p99 59.6741; raw samples in `measurements/000.json`.
- Initial full-depth official comparison passed; `correctness/000-official.json` contains prefix KV and action errors. Oracle provenance is a vendored OpenPI forward and records its dirty source status. This is numerical agreement, not robot task-success validation.
- The separate pi05_base checkpoint previously failed prefix KV tolerance. This run does not claim its correctness or performance.
- Profiling and the initial floor estimate are recorded below. The latest focused expert trace is after 019 and the latest whole-forward overview is after 021; both are diagnostic rather than end-to-end timing results.

![Optimization progress](progress.svg)

## Initial profiling and candidate selection

Full-forward GPU correlation groups map to vision 5.493 ms, backbone 25.458 ms, expert 28.592 ms (profiling overhead excluded from speedup claims). The expert's FFN is 8.559 ms / 3780 launches and QKV 7.768 ms / 4860 launches. Two independent candidate worktrees fuse their pointwise chains around unchanged GEMMs. The backbone FFN is 15.466 ms / 272 launches, so its norm/GELU/product fusion is a third candidate. GPU jobs remain serial.

A default native-SDPA screening on random actual-shape attention tensors was slower: 76.27 us vs 32.44 us warm-cache call time. Explicit efficient-attention dispatch agrees; this installed cuDNN attention route rejects head_dim 256. This rejects these tested dispatches, not fused attention in general. Raw records: measurements/sdpa-screen.json and sdpa-dispatch.json.

The existing cost model predicts a 25.143 ms datasheet floor and 25.202 ms measured-primitive estimate. It prices every call's traffic as cold DRAM and omits host work; cache-sensitive rows and changing unlocked clocks prevent treating this sum as proof of exhaustion. Individual expert FFN estimate is 2.647 ms and QKV 1.268 ms.

## 001 — Expert FFN pointwise fusion, retained

Hypothesis: remove 17 pointwise/cast launches per FFN call while retaining both torch.mm and all bf16 rounding points. Actual belt-cup invocation snapshots at seed42 gave exact outputs/factors for calls 0/17/90/179; local 180-call graph median 54.9115 -> 25.8879 us/call (5.224 ms sum-equivalent estimate, not a deployment measurement). Full-depth official parity passed after integration.

Deployed shipped median 59.5779 -> 55.4515 ms (-4.1264 ms, -6.93%). Raw reports 000/001 use matched conditions and independent processes. Observed clocks were 2872/2865 MHz and memory 13801 MHz, both ending at 52 C; candidate reports a software-power clock reason, so its smaller gain than the local estimate is not assigned solely to one cause. Large median separation versus within-run spread supports retaining it; no periodic control rerun was added. Source revision 23f3c8b.

## 002 — Expert QKV pointwise fusion, retained

Two native kernels around the unchanged BF16 GEMM remove repeated casts and elementwise launches. Synthetic actual-shape tests at magnitudes 1, 1e-3 and 1e3 are bitwise equal including the factor, preserving KV-cache prefix sentinels; cold rotating graph samples are saved in the QKV candidate record. Full-depth real-weight official comparison passed after integration and its action metrics equal 001.

Deployed median 55.4515 -> 49.8801 ms, source c8f5e15. It is compared to the already-retained FFN version, not the initial model. The complete loaded plan is in measurements/002.json.

## 003 — Backbone FFN pointwise fusion, retained

Hypothesis: remove full FP32 cast/intermediate traversals around the two unchanged large GEMMs. Seventeen real call snapshots passed existing shallow tolerance: worst output rel_rms 1.14e-4 and minimum cosine 0.9999999935. Local graph totals torch 15.698/15.975 ms vs fused 11.936/11.971 ms, with reference drift noted. Full-depth official comparison passed after deployment.

Deployed median 49.8801 -> 45.8682 ms, source 571b9b4. Raw local/official/end-to-end evidence is saved alongside this trial. The change affects prefix KV numerics within existing tolerances, not the model precision policy.

## 004 — Packed expert FFN, retained

Hypothesis: pay skinny-GEMM launch/scheduling overhead once for gate and up. One BF16 GEMM writes two contiguous halves; native activation preserves BF16 projection rounding. Source tensors and packed weights are retained in the wrapper; all GPU storage uses runner scratch. Additional packed weights: 288 MiB for 18 layers.

Selected real invocation outputs/factors were bitwise equal; local median 25.8104 -> 17.1574 us/call, 1.5575 ms estimated across180 calls. Full-depth official comparison passed. Deployed shipped median 45.8682 -> 44.4914 ms, source ebfde75.

## 005 — Expert masked attention, retained

Hypothesis: use tensor-core BF16 inputs with FP32 score output to remove Q/K casts and FP32 SIMT GEMM, then fuse scale/mask/softmax into one native kernel, preserving BF16 probabilities before the unchanged P@V GEMM. Synthetic real-shape tests include out=Q alias, runtime mask changes, and masked-V independence; worst output rel_rms 1.56e-4. Local graph medians 37.8 -> 15.2 us include identical Q reset copy on both sides. Full-depth real-weight official comparison passed.

Deployed shipped median 44.4914 -> 41.0824 ms, source 4d53ad7. This native chain replaces only expert attention; the rejected native-SDPA screen remains separately recorded.

## 006 — Expert gated residuals, retained

Hypothesis: each projection keeps the existing BF16 GEMM and fuses its seven cast/mul/add/store operations into one CUDA pass. Selected real inputs at calls0/17/90/179 for both sites match bitwise. Local timings include identical residual resets on both sides: output projection19.1919 ->7.7451us, FFN down22.8162 ->12.7092us, giving3.8797ms sum-equivalent potential. Scratch adds100KiB. Full-depth official comparison passed.

Deployed median41.0824 ->37.9806ms, source e98b74c. This comparison adds the same shared residual kernel at both expert projection sites; no GEMM or numerical tolerance changed.

The affected CPU Target declaration test initially found native library loading in the backbone wrapper factory. Loading now occurs only on first execution, allowing graph/route declarations without a GPU/compiler. CUDA arithmetic is unchanged; the scoped declaration/route checks pass (8+1 checks). This loader-only fix does not add a timing point.

## 007 — CUTLASS backbone gate/up GEMMs, retained

The two large BF16 projections now use the tested128x128x64 Stream-K tile; native RMSNorm/GELU and BF16 projection output stay unchanged. The vendor revision matches the original screening library (main's cb4247394dd82148787aed73e5dc7cef33cbf862); a different installed CUTLASS checkout was detected and avoided. The two best screened tiles differed by less than the control drift, so one simpler cfg0 is used. Native compiler reports254registers and0spills; model capture/replay and all17realFFN calls passed.

Local FFN total11.591/11.862 ->10.866/10.883ms; full-depth official comparison passed. Deployed median37.9806 ->37.0332ms, source594e697. Only the gate/up call site is routed to CUTLASS in this trial; down is next.

## 008 — CUTLASS backbone down projection, retained

The same cfg0 Stream-K tile now runs the down projection with C=D and alpha=beta=1, preserving the existing residual expression. All17 actual calls passed existing tolerance; local total5.827/5.869 ->5.038/5.039ms. Full-depth official comparison passed.

Deployed median37.0332 ->36.2217ms, source54b5946. This is the incremental gain after007. Native workspace and pointer-bound plans are owned per runner and initialized during warmup.

## 009 — Vision normalization and activation, retained

FP32 centered-variance LayerNorm now uses a single CUDA pass, retaining BF16 normalized inputs to unchanged bias-fused GEMMs. FFN GELU runs in place after the projection rounds to BF16. Selected real layers passed local tolerance; 27-layer ABBA estimates about 1.07 ms of headroom. Full-depth official comparison passed (action cosine 0.9999852922, rel_rms 0.00542377). CPU Target binding also passes without a CUDA compiler environment.

Deployed median 36.2217 -> 35.3236 ms, source cc5f0a8. Both measurements ended at 2865 MHz SM, 13801 MHz memory and 57 C with the same power clock reason; observed within-run variation is much smaller than the 0.8981 ms gain.

## 010 — Prefix QKV pointwise fusion, retained

Reuse the Target-local RMSNorm kernel, retain the BF16 QKV GEMM, and replace the cast/rotation/scatter chain with one native pass. Selected actual layers 0/9/17 matched bitwise locally; 18-call graph median decreased from 137.960 to 58.0524 us per call (1.4383 ms sum-equivalent estimate). Full-depth official comparison passed (action cosine 0.9999866853, rel_rms 0.00516067). CPU declarations pass without a compiler environment.

Deployed median 35.3236 -> 33.9966 ms, source 7085a4b. Ending SM/memory clocks match 009; temperatures were 57/55 C. The 1.3271 ms gain exceeds within-run spread. Total reduction from the initial 59.5779 ms deployment is 42.94%; BF16 precision, full 18 layers and 10 denoise steps remain the measured workload. The model has remaining GEMM headroom; the initial floor is guidance only.

## Next bottleneck after 010

The new whole-forward trace maps GPU kernel duration to vision 4.539 ms, backbone 18.386 ms and expert 10.717 ms. Focused backbone attribution has no sequence mismatch or unattributed launch. Its FFN is 10.508 ms, including 10.026 ms in 34 gate/up GEMMs and 0.482 ms in native RMSNorm/GELU. Down GEMMs add 4.835 ms. Profiler time is diagnostic, not an extra latency measurement.

A controlled 12-tile expert screen on 18 actual weight sets found a small candidate: cfg9 packed GEMM 13.392 us vs torch 13.774/13.787 us, down 9.561 us vs 10.140/10.137 us. All configurations passed local tolerance. The approximately 0.17 ms model sum-equivalent is unverified; cfg9 remains a candidate while the larger backbone hotspot is investigated. The probe initially hit a native-library initialization collision; standalone and dual-library probes isolated GNU-unique CUTLASS TLS sharing, then reusing the existing main cfg0 library resolved the diagnostic failure. See lab/pi05/cutlass_expert_screen.md and measurements/expert-cutlass-screen.json.

The focused NCU capture ran the deployed gate on saved actual layer-0 operands, with cache-control all, clock-control none, kernel replay and one NVTX-selected launch (5 passes). Tensor activity was 92.06% of sustained elapsed peak, versus DRAM 14.63% and L2 49.03%. The kernel uses 254 registers/thread, 96 KiB dynamic shared memory and one 128-thread CTA resident per SM. This supports tensor work as the dominant limit of this gate implementation; low occupancy alone is not an optimization target. Its 313.216 us profiler duration uses isolated cold replay and must not be compared as a deployment version. Only NCU used sudo; report export/read required no ownership change.

Focused expert attribution after 010 is 10.679 ms in 3080 launches. Packed FFN is 2.840 ms, attention 2.197 ms, QKV 2.131 ms, FFN down residual 1.870 ms, output-projection residual 1.375 ms and action output head 0.217 ms. Each residual still has cuBLAS GEMM, split-K reduction and native gate/add kernels; the native add itself costs about 1.03 us per call in this trace. Candidate estimates must use this current evidence rather than summing unrelated isolated timings.

## 011 — Action output pointwise fusion, retained

The existing Target-local QKV library adds two dedicated kernels for the action head: private BF16 RMS factor, unchanged BF16 GEMM, and sequential FP32 factor/bias/Euler residual update. The public norm_factor argument remains untouched. All 10 actual-step local outputs and preserved factor buffers matched bitwise. Reset-inclusive 10-call ABBA measured torch 0.234208/0.234160 ms versus fused 0.066256/0.066336 ms. This is a reused 1.745 MB working set, not a cold-weight test.

Full-depth official comparison passed with the same action metrics as 010. Because the estimated gain is small, four fresh-process end-to-end runs used A-B-B-A ordering: prior plan 33.956571/33.942453 ms, candidate 33.743207/33.766837 ms. Both A legs loaded the same complete route map. The mean of the two medians improves by 0.194490 ms; control drift is 0.014118 ms and candidate drift 0.023630 ms. SM clocks ended at 2865/2865/2872/2865 MHz, memory 13801 MHz, temperatures 58/56/57/59 C. This supports a small repeatable gain under the existing unlocked-clock conditions. The older 010 point is not used to attribute this incremental gain. Source 853d1fa.

A matching cold down capture (actual layer-0 A/B/initial residual, alpha=beta=1, C=D) reports tensor activity 95.90%, DRAM 20.36%, L2 50.24%, 303.168 us and effective SM frequency 2.725260 GHz. Useful work is 64.961 GFLOP, or 214.275 TFLOP/s; the measured unit-MMA limit of 512 FLOP/cycle/SM at that same frequency gives 237.207 TFLOP/s. The ratio is 90.33%. Counting the actual 128-row tiles pads M from 968 to 1024: 68.719 GFLOP / 303.168 us is 226.671 TFLOP/s, or 95.56% of the same-frequency limit, close to the tensor activity. This supports M-tail padding as much of the useful-work gap, but it is not guaranteed removable time; smaller tiles also change reuse and scheduling. Gate/down currently justify no repeat of the same 12-tile sweep; other GEMM shapes and fusion boundaries remain candidates.

## 012 — Expert down rounded gated epilogue, retained

The existing native CUTLASS library adds one cfg9 Stream-K broadcast epilogue. It reduces partial FP32 accumulators, rounds the GEMM result to BF16, then explicitly performs separate FP32 multiply and add before BF16 output. C=D preserves the in-place residual; no temporary projection is needed. All 180 actual calls passed existing shallow tolerance (worst rel_rms 0.00195104); selected same-mainloop BF16-projection decompositions match exactly. Local reset-inclusive ABBA is 12.72080/10.67627/10.68071/12.71182 us per call.

The first implementation failed numerical coverage and was not timed. Constant probes and comparison with the ordinary upstream epilogue isolated a missing destination fragment advance in the vendored broadcast reduce path. A Target-local adapter supplies that advance without changing the vendored header. All 50 rows of three constant probes then match exactly. The failed result remains recorded; see lab/pi05/rtx5090_expert_epilogue.md for the source evidence and fixed probe.

Full-depth official comparison passed (action cosine 0.9999849792, rel_rms 0.00548124). Four fresh-process A-B-B-A runs give prior plan 33.784778/33.769896 ms and candidate 33.499048/33.482750 ms. The mean of medians improves by 0.286438 ms; A/B drift is 0.014882/0.016298 ms. Only action_expert_ffn_down_residual differs between loaded plans. SM clocks ended 2865/2857/2865/2865 MHz, memory 13801 MHz, temperatures 58/57/58/58 C. Other compilation/model loading was paused during these end-to-end timings. Source df996b8.

## 013 — Shared vision FFN bias-GEMM tile, retained

Both vision FFN GEMMs use cfg10 (64x128x32, warp32x64x32, 5 stages) in the same Target-native library. The BF16 bias is read with ldc=0 and added to the FP32 accumulator before BF16 output. Up retains the existing norm and GELU; down retains the independent BF16 residual add. All 27 layers per site passed local tolerance, and repeated graph replay matched bitwise. Choosing separate cfg5/cfg10 tiles estimated only another 0.014 ms locally, so the production candidate uses one tile.

Complete-chain ABBA totals: up A1.398624/B1.298080/B1.297056/A1.425248 ms; down A1.394720/B1.116160/B1.112064/A1.384448 ms. Conservative local separation totals about 0.369 ms. Full-depth official comparison passed (action cosine 0.9999915368, rel_rms 0.00411417). The compiled new tile uses 160 registers and no spills.

Fresh-process end-to-end ABBA gives prior plan 33.459378/33.579306 ms and candidate 33.057243/33.062119 ms. A drift is 0.119928 ms versus B drift 0.004875 ms, so the observed improvement is reported as approximately 0.40–0.52 ms, not a falsely precise single kernel attribution. SM clocks ended 2857/2872/2865/2857 MHz, memory 13801 MHz, temperatures 59/60/59/58 C; unchanged power policy can couple workload to boost frequency. Only the two vision FFN routes differ. Every candidate/control median remains separated, and no extra repeat was added. Source 5f7ccf0.

## 014 — Expert output projection rounded gated epilogue, retained

Reuse the existing cfg9 rounded gated epilogue for K=2048 output projection by passing K through its workspace/plan ABI. The FFN down path remains K=4096; no second kernel is added. All 50 rows of the K=2048 constant probes match exactly. All 180 actual calls pass the existing shallow tolerance (worst rel_rms 0.00194467); selected same-mainloop BF16 projection decompositions match exactly. Local reset-inclusive ABBA is 9.08560/7.40080/7.40196/9.08836 us per call. Its 18 weight sets total 72 MiB and can fit in the 96 MiB L2, so this isolated result alone did not establish deployment benefit.

Full-depth official comparison passed (action cosine 0.9999916982, rel_rms 0.00407475). Fresh-process end-to-end ABBA gives prior plan 33.139814/33.110785 ms and candidate 32.845290/32.833674 ms. The mean of medians improves by 0.285817 ms; A/B drift is 0.029029/0.011616 ms. The loaded plans differ only at action_expert_out_proj_residual. All four ending SM clocks are 2865 MHz, memory 13801 MHz and temperatures 57/57/56/57 C; the same power clock reason remains active. Other GPU work, compilation and model loading stayed paused during measurement. Source 7669007.

## 015 — Reuse vision bias-GEMM tile for QKV, retained

A small Python wrapper reuses the deployed cfg10 native bias GEMM and existing LayerNorm for vision QKV. No new CUDA or ABI is introduced. All 27 actual-layer outputs match bitwise; local ABBA totals are 1.010784/0.962976/0.964240/1.026688 ms, with only 0.046544 ms minimum separation. Raw local evidence is in results/rtx5090-pi05/gpt6-vision-qkv-cfg10. Full-depth official comparison passed with action metrics equal to 014.

Fresh-process end-to-end ABBA gives prior plan 32.849472/32.834406 ms and candidate 32.807739/32.796503 ms. The mean of medians improves by 0.039818 ms; A/B drift is 0.015066/0.011236 ms. Both candidate medians are below both control medians, with the nearest separation 0.026667 ms. This supports retaining a small observed gain under the existing unlocked-clock conditions; its exact size is uncertain and no extra repeat was added. Only vision_encoder_norm_qkv differs between loaded plans. SM clocks ended 2872/2865/2865/2865 MHz, memory 13801 MHz, temperatures 58/58/59/58 C. Source 5772cf2.

## Profile and declaration check after 015

The refreshed whole-forward trace attributes GPU kernels by each segment's cudaGraphLaunch correlation: vision 4.169159 ms / 302 launches, backbone 18.379232 ms / 233 launches, expert 10.020473 ms / 2230 launches. The report is diagnostic; these sums do not replace end-to-end measurements or isolate the cause of changes from 010. The expert launch count decreased from 3080 to 2230. Raw trace/report remain under profiles/015-overview; the compact record is profile-015-summary.json.

The scoped CPU Target check passes 8 declarations and 1 route check with CUDA_HOME and TORCH_CUDA_ARCH_LIST unset. No other Target tests were rerun. Upcoming CPU-screened hypotheses are a BF16-rounded dual-dot FFN suffix, cfg10 reuse for vision output projection, and a padded PV dispatch check in expert attention. They are unmeasured candidates, not claimed gains.

## 016 — Rounded dual-dot expert FFN suffix, retained

One short Triton kernel replaces the packed BF16 GEMM plus bias/GELU/product kernel. Prepare and packed weight ownership are retained. The sole production tile is 16x64x32 with four warps and three stages; the other two experimental tiles are not deployed. Both FP32 accumulators explicitly round to BF16 and back before FP32 bias, GELU and product. The 819,200-byte projection scratch is removed. All 18 real layer snapshots match bitwise locally. The 288 MiB rotating weight set exceeds L2. PTX confirms BF16-input/FP32-accumulator MMA, both BF16 roundtrips before bias, and separate outer multiply/add operations. The tile uses 64 registers, 18,432 bytes shared memory and no spills.

Local suffix ABBA gives A16.0462/16.1280 us and B14.3022/14.3253 us per call, suggesting 0.31–0.33 ms over 180 calls. Full-depth official comparison passed with action metrics equal to 015. Fresh-process end-to-end ABBA gives prior plan 32.861063/32.857811 ms and candidate 32.743964/32.777847 ms. The mean of medians improves by 0.098532 ms; A/B drift is 0.003253/0.033883 ms. Only action_expert_norm_gated_ffn differs. The deployment gain is smaller than the local estimate; no additional repeats were used to seek a larger result.

Ending SM clocks are 2872/2865/2865/2865 MHz, memory 13801 MHz, temperatures 57/59/59/57 C and power 586.45/598.67/597.60/588.57 W. Power and thermal behavior may interact with the unlocked clock policy, but these snapshots do not establish why the local gain shrank. The shipped route is retained based on the separated end-to-end medians. CPU backend declaration/route checks passed 9/9 without CUDA initialization before integration. Source 78c8ca1.

## 017 — Reuse vision bias GEMM for output projection, retained

Reuse cfg10 for M768/K1152/N1152 with bias, then retain the separate BF16 residual add. The existing residual wrapper takes K from the weight shape and serves both vision output and FFN down projection. No new native kernel or ABI is added. All 27 actual layers pass existing shallow tolerance (worst rel_rms 0.00105418, minimum cosine 0.999999444); these outputs are not bitwise equal to cuBLAS, while repeated candidate replay is identical. Local complete-chain ABBA gives A0.587776/B0.454656/B0.454656/A0.587776 ms, or 0.133120 ms separation. The 68.34375 MiB weight set can fit L2, so deployment was measured independently.

Full-depth official comparison passed (action cosine 0.9999912284, rel_rms 0.00418872). Fresh-process end-to-end ABBA gives prior plan 32.771943/32.782361 ms and candidate 32.610786/32.613452 ms. The mean of medians improves by 0.165034 ms; A/B drift is 0.010418/0.002666 ms. Only vision_encoder_out_proj_residual differs between the loaded plans. Ending SM clocks are 2865/2872/2865/2865 MHz, memory 13801 MHz and temperatures 58/59/59/58 C. Source eb8c16a.

## Rejected PV padding dispatch screen

The 010 expert attention trace assigns 0.804084 ms to QK, 0.384005 ms to softmax, and 1.008733 ms to PV GEMM plus split-K reduction over 180 calls. There is no independent copy/memset in this chain. A single real step0/layer0 P/V pair tested whether padding K from 1018 to 1024 could improve the PV dispatch. Padding was excluded from timing, giving this candidate an optimistic screen.

The original path measured 5.583/5.774 us against padded 6.928/6.926 us. Padding removed split-K reduction but selected a slower 32x32 WMMA align8 GEMM. The padded result passed existing shallow tolerance and was not bitwise equal. This candidate is rejected before any full attention implementation; no production padding was introduced. Detailed evidence is in results/rtx5090-pi05/gpt6-attention-pv-padding.

The scoped CPU Target check after 017 passed all 8 declarations and 1 route check with CUDA_VISIBLE_DEVICES empty and CUDA_HOME/TORCH_CUDA_ARCH_LIST unset. This covers the new vision residual route without a CUDA compiler or device.

## 018 — Single-launch expert PV, withdrawn after inconclusive deployment timing

The candidate used only the fixed 16x32x64 Triton PV tile, retaining the existing FP32 QK scores, runtime-mask softmax, materialized BF16 probabilities and out=Q alias. Nine actual step/layer pairs passed existing shallow tolerance (worst rel_rms 0.0027468, minimum cosine 0.999996229); results were not bitwise equal. The nine-pair warm-set ABBA totals were torch 50.786/51.024 us and Triton 43.946/43.915 us. Trace confirmed one PV launch and PTX used BF16-input FP32-accumulator MMA. Full-depth official comparison passed (action cosine 0.9999909392, rel_rms 0.00425716).

The first fresh-process ABBA gave A32.596582/B32.499377/B32.533664/A32.561475 ms: mean-of-medians difference 0.062508 ms, with A/B drift 0.035107/0.034287 ms and nearest separation only 0.027812 ms. Because the difference was close to the drift scale, one additional reverse-order BAAB block was specified before collecting more data; all eight runs are retained.

The fixed BAAB returned B32.489375/A32.557318/A32.638460/B32.602458 ms. It has a positive mean-of-medians difference of 0.051972 ms, but candidate/control medians overlap; A/B drift is 0.081142/0.113082 ms. Ending clocks change from 2865 MHz in B1/A1 to 2857 MHz in A2/B2, with temperatures 60/62/62/61 C. These snapshots cannot isolate the clock contribution. All eight mean medians favor the candidate by 0.057240 ms, but the reverse-order check does not provide stable separation under the existing deployment policy.

The candidate is recorded as inconclusive and its production module, registry entry and shipped route were removed in 1542c05. The original source is reviewable at 34add1b; lab experiments and all latency/correctness reports remain. The curve marks this measured trial as withdrawn and retains 017 as the deployed version. No more repetitions were used to seek a favorable result.

## 019 — Expert QKV Triton GEMM, retained

The fixed 16x32x32 Triton tile replaces only the middle GEMM; the existing native prepare, public factor, bias/RoPE/scatter and BF16 projected scratch stay in use. The two losing experimental tiles are not deployed. All 18 captured real layer calls match bitwise for projected output, Q/K/V and factor. PTX confirms BF16-input FP32-accumulator MMA followed by BF16 round-to-nearest before the store. The winner uses 40 registers, 6 KiB shared memory and no spills.

Because the actual 18 weights total 90 MiB and can fit L2, the pure-GEMM screen explicitly rotates two independent copies of the same real weights (180 MiB), with identical addresses/order for A and B. This is a cache-pressure experiment, not the deployment sequence. Its ABBA gives A9.9929/9.9529 us and B8.5680/8.5698 us. The subsequent original-18-weight full-chain ABBA gives A11.6107/11.1396 us and B9.9236/10.2329 us. The latter control/candidate drift is 0.4711/0.3093 us; its conservative separation is 0.9067 us/call, and the drift is retained in the report.

Full-depth official comparison passed with action metrics equal to 017. Fresh-process deployment ABBA gives prior plan 32.565772/32.616870 ms and candidate 32.426883/32.429453 ms. The mean of medians improves by 0.163153 ms; A/B drift is 0.051099/0.002570 ms. Only action_expert_norm_qkv_rope differs. All ending SM clocks are 2865 MHz, memory 13801 MHz and temperatures 59/59/57/58 C. The separated medians support retaining the gain, while the control drift limits exact attribution. CPU declaration/route checks passed 9/9 without CUDA initialization before integration. Source 0dcb021.

## Rejected cfg0 reuse for backbone output projection

The current torch addmm already combines FP32 accumulation and the old residual before one BF16 store. Reusing the existing cfg0 backbone-down closure for M968/K2048/N2048 preserves those rounding locations and C=D aliasing; no extra residual kernel exists to remove. All 17 actual-layer outputs match bitwise. The 136 MiB weight set exceeds L2, and both paths use the same residual reset excluded from timing.

Single-tile ABBA totals A0.895008/B0.901120/B0.901120/A0.894976 ms across 17 calls. The existing cfg0 reuse is slower by 0.006128 ms versus the mean control, so its production wrapper change was removed and no additional tile was tried. This result rejects only that reuse candidate. Lab reproduction and all 60 samples remain in lab/pi05/cutlass_backbone_outproj.md and measurements/backbone-outproj-cfg0-rejected.json.

## Focused expert profile after 019

The refreshed positional mapping matches all 2050 launches, with no mismatched position or unattributed launch. Attributed kernel time is 9.655205 ms: FFN 2.735196 ms, attention 2.202637 ms, QKV 1.917469 ms, FFN down 1.526675 ms, output projection 1.174537 ms, action output 0.050338 ms and action input 0.048353 ms.

The current QKV split is prepare 0.223818 ms, GEMM 1.476047 ms, finish 0.217604 ms. Attention remains QK 0.803068 ms, softmax 0.383206 ms, PV GEMM 0.736895 ms and PV reduction 0.279468 ms. This supports continuing the already-prepared QK and QKV-finish experiments; it is a diagnostic snapshot, not another deployment measurement or an explanation of all differences from the older profile. The optional marker extension remains unavailable because Ninja is absent; all launches were nevertheless mapped by correlation and sequence. Compact evidence is profile-019-expert-summary.json.

## 020 — Expert FP32-score QK Triton tile, retained

The sole 32x32x64 tile replaces the QK GEMM. The runtime-mask native softmax, BF16 probability materialization, original torch PV/split-K reduction, output alias and scratch roles are retained. The current call remains one QK launch; this is a GEMM implementation change. All nine actual step/layer cases match bitwise for FP32 logits, BF16 probabilities and complete attention output. Reset-inclusive nine-call complete-chain ABBA gives control 118.314/118.315 us and candidate 110.126/110.133 us, about 0.91 us per call. Trace confirms the original later dispatches and the fixed QK kernel. Raw local evidence is in results/rtx5090-pi05/gpt6-attention-qk-triton.

Full-depth official comparison passed with action metrics equal to 019. Fresh-process end-to-end ABBA gives prior plan 32.406389/32.362382 ms and candidate 32.171719/32.211052 ms. The mean of medians improves by 0.193000 ms, with A/B drift 0.044007/0.039333 ms. Every candidate median remains below every control median; the observed separation ranges from approximately 0.151 to 0.235 ms. Only action_expert_attention differs.

Ending SM clocks are 2857/2865/2857/2865 MHz, memory 13801 MHz and temperatures 59/58/60/57 C. Both routes include one sample at each ending SM clock; these snapshots do not establish identical frequency histories. The gain is retained as deployment behavior under the existing policy without assigning the entire difference to isolated QK time. CPU wrapper declaration passed without CUDA initialization before integration. Source 76086d9.

A separate CPU-only layout hypothesis was dismissed: backbone QK and PV already use flat 2D GEMMs with Q/out(7744,256), K/V(968,256), and score/P(7744,968). There is no remaining batch dimension to remove by changing matmul to mm. No code, GPU screen or new softmax was introduced for that hypothesis.

## 021 — Expert QKV rounded finish fusion, retained

The existing 16x32x32 Triton GEMM tile now performs the original factor multiplication, bias, adjacent-pair RoPE and output scatter after an explicit BF16 roundtrip. Native prepare and the public factor stay unchanged. K/V use the caller-provided suffix views without adding a second prefix offset. The 256,000-byte projected scratch and one finish launch are removed. All 18 actual-layer Q/K/V/factor outputs match bitwise and prefix caches stay identical. Compiled resources remain 40 registers, 6 KiB shared memory and no spills. Local complete-chain ABBA is A10.807111/B9.637333/B9.630222/A10.775111 us/call; its same-weight local cache conditions differ from deployment.

Full-depth official comparison passed with action metrics equal to 020. The scoped CPU Target check passes all 8 declarations and 1 route check. Fresh-process end-to-end ABBA gives A32.194986/B32.041923/B32.142267/A32.234348 ms. Mean-of-medians gain is 0.122573 ms, but B drift is 0.100344 ms versus A drift 0.039362 ms. Because candidate drift is close to the gain, one fixed reverse-order BAAB block was declared before collecting further results.

That reverse block gives B32.139037/A32.250799/A32.248522/B32.162340 ms. Its mean-of-medians gain is 0.098972 ms; A/B drift is 0.002277/0.023303 ms and minimum separation 0.086182 ms. All eight mean medians favor the candidate by 0.110772 ms. Candidate/control medians stay separated in both orders, supporting retention. The curve uses the first candidate measurement by the same convention as earlier iterations; the four candidate medians span 32.041923–32.162340 ms, so that first point alone is not the incremental gain estimate.

The eight ending SM clocks are 2857/2865/2857/2865/2857/2865/2857/2857 MHz, memory 13801 MHz, with temperatures 57/56/59/59/60/61/58/59 C. These snapshots do not establish identical clock histories. Only action_expert_norm_qkv_rope changes across the verified complete plan maps. No GPU work, compilation or model loading overlapped these end-to-end runs. Source 4de57c9; all reports and the predeclared follow-up decision are retained.

## Rejected fixed-cfg0 backbone up/GELU epilogue

A same-library experiment reads the already rounded gate as matrix C and fuses GELU/product into the up GEMM epilogue. The original up BF16 roundtrip, exact tanhf expression and compiler precision flags are preserved. Both original and fused cfg0 compile at 254 registers with no spills; all 17 real layers match bitwise.

One fixed local ABBA across the 17 actual up weights gives A5.528512/B5.716608/B5.778464/A5.623680 ms. The candidate is slower by 0.171440 ms using mean medians, and even the faster candidate is 0.092928 ms slower than the slower control. It is rejected before deployment testing. Pre-materialized gate matrices change the local cache context, so the result does not isolate the cause or establish a deployment slowdown. Production native/ABI edits and the experimental binary were removed; the exact reproducible patch, resource excerpt, numerical errors and all 60 samples remain in lab/pi05/cutlass_gelu_up.md and measurements/backbone-gelu-up-cfg0-rejected.json.

## Whole-forward profile after 021

GPU events correlated with the three graph launches attribute vision 4.095025 ms / 302 launches, backbone 18.564488 ms / 233 launches and expert 9.383119 ms / 1870 launches. All 2405 kernel launches map to a segment. The latest expert count reflects the QKV finish launch removal. The overview reports approximately 0.518 ms of gaps in its GPU span, but profiling perturbs scheduling and this is not an unprofiled host-overhead estimate. No end-to-end improvement is inferred by subtracting old profiler sums.

The 27 independent vision GELU launches still total 90.916 us, close to the older 015 diagnostic value. This bounds the small upcoming vision fusion opportunity before replacement work. Compact evidence is profile-021-summary.json; raw trace/report remain under profiles/021-overview.

## Rejected fixed-tile Triton expert FFN down

A single 16x32x32, four-warp, three-stage Triton tile tested the existing rounded gated residual in one kernel, with K4096, 128 CTAs and no split-K. It compiles at 38 registers, 6 KiB shared memory and no spills; PTX retains the BF16 roundtrip and separate FP32 multiply/add. All 180 actual calls pass existing shallow tolerance (worst rel_rms 0.000269683128), and three same-mainloop decompositions match exactly. This is numerical agreement rather than bitwise equality to the deployed cfg9.

Reset-inclusive local ABBA gives A10.567289/B29.861511/B29.867199/A10.577244 us/call. The candidate is slower by 19.292088 us/call versus A/B drift 0.009955/0.005688 us, so it is rejected without deployment timing or another tile. The longer K loop, smaller grid and different cache requests remain possible explanations, not an established causal result. The actual 18 weights total 144 MiB, above L2 capacity. Production remains unchanged; lab/pi05/triton_expert_down.md links the raw 180-call errors, decomposition checks, 60 samples and PTX resources.

A separate CPU-only QK/softmax screen also stops the direct full-row CTA approach because of its limited parallelism and padded MMA work. A transposed four-query layout remains an untested boundary; its possible software support, cache traffic and reduction mapping were explicitly left uncertain. No kernel or GPU experiment was added for that direction. See results/rtx5090-pi05/gpt6-attention-qk-softmax-screen/README.md.
