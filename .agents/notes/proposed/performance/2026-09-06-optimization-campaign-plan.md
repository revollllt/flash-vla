# Flash-VLA 优化战役执行计划：三段全开的 kernel-design lanes（A 尾部归因、B 视觉编码器、C 骨干 GEMM、D 动作专家形态）

## Context

重构五个 PR 已合入本地 main（c7afd03）：显式计算图 + ModelRunner、单一部署配置 + lab/、
验收 = 可部署（`eval.gate`）、三列 floor（roofline / measured ceiling / measured）、
rel_rms + cosine 双门。飞轮 Profile → Analyze → Design → Implement → Validate → Deploy
的工具都在，ARCHITECTURE.md 记录了它。用户决定：在现有两个 Target（H100 上的 Pi0.5 与
Pi0）上放开手脚做 performance，三段（vision_encoder / llm_backbone / action_expert）都做，
用 kernel-design skill 的 wiki 与模板探索完整的优化空间，而不是只做 vision。

Analyze 已完成（job 598964 的 floor 报告，shipped plan，同一节点 ACD1-55）。按
"实测 − measured ceiling" 的绝对余量排序，两个 Target 的排行一致：

| 段 | Target | 实测 | measured ceiling | roofline | 上限收益 |
|---|---|---|---|---|---|
| vision_encoder | pi05 / pi0 | 2.05 / 2.54 ms | 1.01 / 1.01 | 0.71 | 1.04 / 1.53 ms |
| llm_backbone | pi05 / pi0 | 6.75 / 5.95 | 4.46 / 3.54 | 3.81 / 3.00 | 2.29 / 2.42 |
| action_expert | pi05 / pi0 | 8.13 / 7.70 | 4.41 / 4.46 | 2.04 / 2.04 | 3.72 / 3.24 |

三段合计上限 pi05 7.0 ms（16.9 → 9.9）、pi0 7.2 ms；现实目标 2–3 ms。

同时存在一个部署阻塞：Pi0.5 的 chunk p99 − min 间歇性达到 2.5–11.5 ms，超过注册表的
0.5 ms 上界，`eval.gate` 在 Pi0.5 上从未产出 `pass`。已有问题文档
`.agents/notes/proposed/performance/2026-09-06-pi05-chunk-tail-attribution.md`。
没有它，任何性能 lane 都拿不到 `pass` 记录。

## 执行中修正的前提（2026-09-07，协调者记录）

- **Pi0 骨干原不按 `layers` 二分**（lane C 发现）：`pi0/pipeline.py` 的骨干循环写死 18 层，
  `eval.correctness --layers 1` 在 Pi0 上曾是 18 层累积漂移对 shallow 容差的比较。已修
  （d537e01），全深度图不变。
- **Pi0 的官方基线层**（lane B 发现）：Libero Pi0 checkpoint 在机器上存在（`models/openpi/
  openpi-assets/checkpoints/pi0_libero_pytorch`），下文"机器上没有"是错的；且 gate 无参数调用
  `eval.pi0.reference` 而脚本要求 `--checkpoint`，Pi0 的门从结构上必 fail。已修（76b85eb，
  注册表 `OPENPI_PI0_CHECKPOINT`），Pi0 shipped 对 reference 的首次 `pass` 是 job 599777
  （ACD1-55）。Pi0 的 `pass` 因此是真实要求，不再需要"记 unavailable"的例外。
- **floor 报告的段 measured 是归因态数字**（lane D0 发现）：本文 Context 表的 measured 列取自
  `benchmarks floor`（kernel 归因下运行），高于 latency harness 同段同进程 A/B/A 的读数
  （action_expert：Pi0.5 8.13 对 7.40 ms，Pi0 7.70 对 7.05 ms）。lane D0 一节的"Pi0.5 的链是
  7.39"是 latency 数字，与表不同源。裁定：floor 报告只用于逐 site 归因与 ceiling；段级与
  chunk 级数字、以及一切 promotion 判断，用各 lane 作业里的 `benchmarks latency`，并标明仪器。
- **lane D0 的目标不成立**：按 latency 数字，Pi0.5 的链每层步仍比 Pi0 的 tilelang-fused 慢
  2.4–2.9 us（Pi0 的路由能把 RMS 折进 GEMM，AdaRMS 在 Pi0.5 上不允许；CUDA attention 把 Pi0
  的 819 个 key pad 到 1024）。裁定：J1 基线定价后按规则决定——明显更慢则只做组件包搬迁
  （bit-identity 判）并写 rejected note；接近则加候选 C3（key pad 1024→896）且只按 improve
  门 promotion；不用 no-regression 模式 ship 不提速的路由。D1 在 Pi0.5 上定价，不依赖移植。
- **候选顺序调整**：lane C 把 Pi0 的 cuBLAS 替换（原 C4）提前为 C0（预计 Pi0 骨干 −0.63 至
  −0.68 ms，无新 kernel）；lane B 新增 B0，Pi0 的两个 vision pre-norm GEMM 改走 Pi0.5 已 ship
  的 cuBLASLt 路由（两 Target 间 0.49 ms 差距的来源）。lane C 的 Pi0.5 骨干目标 5.5 ms 按其
  逐候选算术达不到（预计 −0.69 至 −0.86 ms），目标是方向，停止条件以注册表为准。
- 共享的 TileLang JIT 装饰器由 lane D0 提到 `hardware/nvidia/tilelang/jit.py`，两个 Target
  的 `base.py` 与组件包都从它导入。
- **lane A 已收口，过渡策略撤销**：尾部机制是 Pi0.5 的 `prompt` host slot 在 GPU 关键路径上做
  逐元素 torch 运算，torch 把它派到 intra-op 线程池，池屏障的等待没有上界（slot 耗时双峰：
  96.5% 的 forward 0.34 ms，其余 8–16 ms）。修法是把只依赖 token 数的三个向量在构造时制表
  （逐位等价），host slot 只做选行与拷贝。ACD1-8 上连续三次 A/B/A 两个 plan 的 p99 − min 全部
  在 0.5 ms 内，job 599815 产出 Pi0.5 首次 `pass`。此后 Pi0.5 的候选也必须拿到 `pass`。
- **lane B 结案**：SigLIP 组件包 `hardware/nvidia/h100/siglip/` 落地（`siglip-cublas`、`siglip-cuda`
  两个后端，两 Target 都注册）。Pi0 的 vision 改走 `siglip-cuda`（手写 LayerNorm + cuBLASLt 投影 +
  单 launch 融合 attention，head_dim 72→80 的 pad 只在 smem 里），chunk min −0.42 ms，gate `pass`
  含基线层（job 599893，ACD1-20）。Pi0.5 最佳候选 −0.094 ms 不过 0.10 ms 门槛（四次 gate 一致），
  shipped plan 不变。contract 的主候选 B1（LayerNorm 折进 GEMM A 侧）未花作业即否决：先前
  short-K GEMM 的消融表明每 SM 的 TMA 事务发射成本是瓶颈，折叠迫使 BLOCK_M ≤ 64 把 QKV 关键路径
  的 box 数从约 36 翻到约 72；"折叠不增加流量"对字节成立、对事务不成立。可复用的结论：短 K
  形状下 floor 的 measured ceiling 达不到，因为第三个常数（每 SM TMA 发射成本）先于带宽与
  tensor core 项绑定。用了 12 个作业中的 7 个。
- **lane D 结案**：D0 组件包搬迁完成（bit-identity），Pi0 移植在动手前被同节点 A/B/A 否决
  （链 40.8–41.0 us/层步对 Pi0 路由 38.0–38.4）；D1 定价上限 0.76 ms < 1 ms，未建原型。
  action_expert 段的余量归于依赖链延迟，见两份 rejected note。

## 探索结果（执行时的依据）

### 形状（两个 Target 的 vision 完全同形）

- vision：27 层，dim 1152，FFN 4304，16 头 × head_dim 72，256 patch/view × 3 view → M=768，
  QKV 打包宽 3456，bf16（`models/pi05/spec.py:37-40`、`models/pi0/spec.py:3`、
  `tilelang/kernels/attention.py:36-37`）。
- backbone：18 层，dim 2048，FFN 16384，QKV 宽 2560 = Q(8×256)+K(256)+V(256)，即 8 个
  query head / 1 个 KV head（MQA）、head_dim 256；prefix 行数 Pi0.5 = 3×256+200 = 968
  （`pi05/pipeline.py:83-84`），**Pi0 = 768**（`benchmarks/targets.py:45` 默认 `prompt_len=0`，
  `pi0/pipeline.py:58`；`DEFAULT_PROMPT` 只用于 pi05）；第 17 层只做 QKV。
- 所有权重与激活都是 bf16（`runtime/runner.py:123-124` 把权重统一 cast 到 `DTYPES["bf16"]`）；
  vision 的 `x` 是 `(3, 256, 1152)`、`qkv` 是 `(3, 256, 3456)`（三维，view 是 batch 维；
  `pi05/pipeline.py:112-116`）；QKV 打包是 Q|K|V 三个 1152 块、块内 head-major
  （`attention.py:45-46` 的 `view(-1, 256, 3, 16, 72)`）；view 之间无 attention。
- backbone 的 q/k/v 是 **token-major**（`Q.view(M, heads*head_dim)`，`pi05/.../wrappers.py:253-256`，
  `base.py:694-700`），K/V 是 padded KV cache 的前导行视图；`rope` 是 `(prefix_len, 256)` bf16
  交错 cos/sin（`pipeline.py:66-73,143`）；gated FFN 的激活是 **GELU-tanh**
  （`base.py:77-78,514`）。
- expert：见 lane D。

### vision / backbone 的现有实现（shipped plan）

Pi0.5 vision（`pi05/backends/tilelang/wrappers.py`，配置 :111-114）：
- `patch_embed` :129 → `tl_matmul_bias_res_mod`（64/128/64，stages 3，128 线程）+ 一次
  torch permute/contiguous。
- `norm_qkv` :145 → `tl_layer_norm`（base.py:369）+ **`torch.addmm`（cuBLASLt）**，2 launch。
- `attention` :192 → `attention.py:42` 的 **torch.compile SDPA**，不是 TileLang，3 launch。
- `out_proj_residual` :158 → `tl_matmul_bias_res`（64/128/128，st4，256 线程）。
- `norm_ffn_up` :170 → `tl_layer_norm` + **`torch._addmm_activation`（cuBLASLt GELU）**，2 launch。
- `ffn_down_residual` :182 → `tl_matmul_bias_res`（64/128/128，st3，128 线程）。

Pi0 vision（`pi0/backends/tilelang/wrappers.py`，配置 :256-261）：**文件是复制后分叉的**，
`norm_qkv` :292 用 `tl_matmul_bias_nows`（128/64/64 st2），`norm_ffn_up` :314 用
`tl_matmul_bias_gelu`（128/128/64 st4 256 线程）；其余四个与 Pi0.5 逻辑与配置相同。
`kernels/fused_norm.py`、`kernels/__init__.py` 是两边唯一逐字相同的文件；`base.py`
（903 vs 923 行）与 `attention.py`（61 vs 38 行）已分叉。

Pi0.5 backbone（配置 :212-219）：`projector` = `tl_layer_norm` + `tl_matmul_bias`（128/128/128
st3）；`embed_prompt` = `index_select` + `mul_`；`norm_qkv_rope` :236 = `tl_rms_norm` +
`tl_matmul_rope_scatter`（base.py:640；128/64/64 st3 128 线程，非 warp-specialized，
:216-217 注明是 load-bearing）；`attention` → **`cuda`** 后端（`backends/cuda/wrappers.py:223`
→ `enc_attn.py:140` → `kernels/enc_attn.cu`，1 launch；DH=256、BM=64、BKK=64、NWG=1、
KDEPTH=4、VDEPTH=2、256 线程）；`out_proj_residual` :274 与 `ffn_down_residual` :294 =
**`out.addmm_`（cuBLAS）**；`norm_gated_ffn` :285 = `tl_rms_norm` + `tl_matmul_gate`
（base.py:471；128/128/128，**stages 2**，256 线程，SWIZZLE=8）。

Pi0 backbone（配置 :344-352，全 TileLang）：`norm_qkv_rope` :360 是 **3** 个 kernel
（`tl_rms_norm` + `tl_matmul_ws` 128/128/64 st4 + `tl_rope_scatter_bf16`，经
`scratch("encoder_qkv")`）；`attention` 是 torch 链（每层 4 launch）；`out_proj_residual` =
`tl_matmul_res_ws` 128/128/64 st4 SWIZZLE=0；`ffn_down_residual` = `tl_matmul_res_ws`
128/128/128 st3 SWIZZLE=8；`norm_gated_ffn` 同 Pi0.5 的 `tl_matmul_gate` 配置。

### op 词汇（`runtime/ops.py`，新后端必须匹配的签名）

`vision_encoder_patch_embed` :134 (images, patch_w, patch_b, pos_emb, out)；
`vision_encoder_norm_qkv` :137 (x, norm_w, norm_b, qkv_w, qkv_b, out, x_norm)；
`vision_encoder_attention` :141 (qkv, out)；`vision_encoder_out_proj_residual` :143
(x, weight, bias, res, out)；`vision_encoder_norm_ffn_up` :146；`vision_encoder_ffn_down_residual`
:150；`llm_backbone_projector` :154；`llm_backbone_embed_prompt` :158；
`llm_backbone_norm_qkv_rope` :161 (x, weight_qkv, rope, q, k, v, x_norm)；
`llm_backbone_attention` :165 (q, k, v, scale, mask, out)；`llm_backbone_out_proj_residual`
:168 (x, weight, out，inout=out)；`llm_backbone_norm_gated_ffn` :171 (x, gate_w, up_w, out,
x_norm)；`llm_backbone_ffn_down_residual` :174。

### 注册与路由

`Registry`（`runtime/registry.py:33`）：`NAMES` → `provided()`；`resolve()` :60；`op_table()`
:68 逐后端调 `make_wrappers(scratch, selected_names=...)`；`graph_contract()` :82 只合并
活跃后端的 `{forbid, require_one}`；`atomic_groups()` :95 合并所有成员都路由到声明后端的
`RouteConstraint`（`runtime/binding.py:26-34`）。Pi0.5 `backends/__init__.py:35-41`：
`{tilelang, cuda, cuda-pdl}`，`cuda-pdl = partial(cuda.make_wrappers, pdl_chain=True)`；
Pi0 `backends/__init__.py:19-24`：`{tilelang, tilelang-fused}`。plan 文件：
`runtime/vla.py:166-186`，任何 harness 都接受 `--plan lab/plans/<name>.json`。workspace：
`runtime/runner.py:40-76` 的 `scratch(role, shape, dtype, device)`，warmup 后冻结，
未覆盖的申请直接抛错。

### 共享 SM90 tile 库（`hardware/nvidia/cuda/tile/`，README.md:16-28）

`common.cuh`（元素别名、Operand/Major、proxy fence、named barrier）、`barrier.cuh`
（Full/Empty 事务 barrier、PhaseRing）、`smem_layout.cuh`（swizzled tile）、`copy_g2s.cuh`
（`tma_load_2d/3d`、multicast、L2 prefetch、`cp_async_16`、`TmaTile2D` ring）、`copy_s2r.cuh`
（ldmatrix）、`copy_r2s.cuh`（stmatrix）、`copy_s2g.cuh`（`tma_store_2d/3d`、bulk store）、
`mma_sync.cuh`（m16n8k16 bf16、m16n8k32 fp8）、`wgmma.cuh`（64×N×16 bf16，N=8..256，SS+RS）、
`gemm.cuh`（`MmaSelector`、`gemm_ss/gemm_rs`）、`tma_host.cuh`（`encode_tensor_map_2d/3d`，
host-only，不可在 capture 内调用）。**没有 split-K 原语**。自检
`python -m eval.tile_sm90`（8 例）。

可复用的 CUDA kernel（`pi05/backends/cuda/kernels/`）：`enc_attn.cu`（MQA prefix attention，
TMA K/V ring + online softmax，唯一与 vision/backbone attention 同形的 CUDA kernel）；
`ffn_taskloop.cu`（**持久化 132-CTA task loop**：GatedUp `gelu(XFS@W1+b1)*(XFS@W2+b2)` 128
个 task、DownResidual 32 个 task、gmem 计数器排序）+ `sm90_ffn_gemm.cuh`
（`Gemm<M,N,K,MajA,MajB>`）、`sm90_ffn_task_desc.cuh`、`sm90_ffn_barriers.cuh`、
`sm90_ffn_warp_roles.cuh`（128 math + 2 producer warp，224 线程）；`attn_taskloop.cu`
（kQkvProj/kAttention/kOutProj 三类 task，132 worker，192 线程）+ `sm90_attn_task_desc.cuh`
（`PREFIX_LEN` 编译期常量）。

### 流程与工具（kernel-design skill、集群约定）

- kernel-design 骨架（`.claude/skills/kernel-design/SKILL.md`）：1 contract（填满每个字段，
  一屏，**交回一次等 ack**）→ 2 在候选 1 之前先测基线（生产路由、torch/SDPA、库 kernel）→
  3 torch 参考（`references/reference-tiers.md`：T2 ABI 镜像每个任务必做，跨 ≥2 段的融合还要
  T3 分解）→ 4 parity（`references/parity.md`：六指标 float64，rel_rms 与 cosine 双门，阈值
  只能来自 `eval/calibrate.py`）→ 5 候选循环（`references/loop.md`）→ 6 一个 PR 内 promotion。
- auto 模式：contract ack 后不再需要人，直到停止条件，回来时带证据包（ledger、最佳候选的
  parity 与 benchmark、还有预算时下一步想试什么）。
- contract 字段（`assets/contract-template.md`）：任务名、Mode、Date、Target、Objective、
  张量表（名字/命名维→固定数/dtype/角色/mutation-aliasing）+ 固定形状来源与可变维守卫、
  融合区域、正确性（参考层级、验证命令、gate/report 划分）、基线表（实现/命令/min ms/job）、
  Ceiling（tag 算术或声明 + job）、收益上限、Promotion（引用 registry 的 promotion_bar_ms 与
  尾部上界，`python -m eval.gate --candidate <plan>`）、允许的方案（后端、可 capture、replay
  路径不分配、`scratch` 分配器、PDL 点）、预算与停止。
- 工作区 `artifacts/ktasks/<task>/`（gitignored）：`contract.md`、`docs/draft.md`（写代码前必须有）、
  `docs/plan.md`、`candidates.jsonl`（id/parent/backend/thesis/status kept|revised|rejected+原因/
  parity/min_ms/artifact 路径）、`benchmark.csv`、`profile/`、`runs/`。一次一个候选；同进程 A/B/A；
  不锁频读 `min`；**编译/profile 作业进行中不改 kernel 源码**。
- 停止条件（先到者）：registry 候选规则 + 尾部上界经 `eval.gate` 判 pass；剩余阻塞已具名
  （GAP 常数、ceiling 已达、外部依赖）；预算用尽（contract 只能收窄不能放宽）。
- promotion 清单：parity 门过；基线配置下的 benchmark 证据；接入 op 表且
  `python -m eval.gate --candidate <plan>` 为 `pass`（blocked 只能重跑）；生产 kernel 加
  `benchmarks/kernels.py` 内置 case；note 更新并把证据摘要抄出工作区。
- wiki（`references/wiki/README.md`，按"你看到 → 读"索引）条目：bulk-store-publish、
  c7518-wgmma-serialization、cluster-barrier-placement、cold-burst-ceiling、
  epilogue-staging-short-k、ext-deepgemm-sm90、ext-fa3-pingpong、ext-flashmla-sm90、
  ext-mpk-megakernel、fusion-economics、layout-production-budget、measure-in-the-graph、
  measuring-persistent-kernels、pdl-placement、pdl-primary-is-a-resource、
  prefetch-across-dependency、price-the-direction、producer-fusion-pdl、reduction-own-task-kind、
  release-on-retirement、scale-on-register-fragment、serial-epilogue-owner、stacked-floors、
  tma-3d-box-row-major、wgmma-tile-n-floor。
- 模板（`references/templates/README.md`）：阶梯 01 tma_mbarrier_ring / 02 warp_specialization /
  03 wgmma_mainloop / 04 epilogue_persistent；原型 10 persistent_ws_gemm、11 fp8_two_level_accum、
  12 attention_online_softmax、13 mla_decode_split_kv、14 grouped_moe_gemm；胶水 30
  rmsnorm_residual、31 swiglu_fp8_quant、32 rope_layouts、33 softmax_rowwise；融合终局 40
  megakernel_interpreter、41 moe_align_finalize、42 hazy_llama_megakernel、43
  mpk_task_graph_runtime、44 megamoe_sm90、45 flag_barrier_megakernel。跑一个：
  `sbatch --export=ALL,TEMPLATE=<file>,RUN_ARGS="..." sbatch/kernel_template.sh`。
  检查器：`scripts/check_templates.py`（需 venv python）、`scripts/check_wiki.py`。
- benchmark-kernel：`python -m benchmarks kernels --target h100/pi05 --plan <plan> --segment <seg>
  --site <site> --timer cupti|cudagraph|events`；默认 cudagraph（48 次内循环成一图，40 次重复）；
  权重循环读取（冷）；一个计时声明必须写明 timer、plan、case、shape、median/std/min、
  TFLOP/s 与 TB/s、节点、不锁频。
- profiler：`python .claude/skills/gpu-profiler-analysis/scripts/run_local_profile.py --backend
  torch|nsys|ncu --output-dir artifacts/profile/<run> -- <command>`；ncu 只能在
  ACD1-10/20/21/31/40/62；解读归 `ncu-report` skill；profiler 时间只做诊断。
- 常数（`python .claude/skills/hardware-unit-test/scripts/constants.py --tag <tag>`，用之前先读
  `units`）：`ld.bw.dev.dram` 2.77 TB/s + 1.85 us 固定；`tma.bw.dev.burst` 16.8 MB 冷 1814 GB/s
  曲线；`wgmma.clock.sm` ~850 TFLOP/s；`launch.lat.dev.ramp` 1.24 us/launch；`ld.ctas.dev.knee`
  128 CTA 拐点（32 CTA 慢 1.63×）；`cluster.lat.sync` 0.65 us；`cluster.count.max` 15 个 8-cluster；
  `atom.lat.dev.hop` 651 ns；`coop.lat.dev.sync` 1.09 us/grid_sync；`coop.ratio.dev.relaunch`
  重启只比 grid_sync 贵 1.29×；`wgmma.issue.wg.ss`（N≥64）；`mma.xover.n.wgmma`（N<32 用 mma.sync）；
  `sched.ctas.sm.knee`（grid ≥ 3× SM）。
- 集群：只用 `sbatch -p acd_u --gres=gpu:1`；`sbatch/_common.sh` 负责 module/venv/compat 库规则/
  编译器/per-job 缓存；通用 `sbatch -J <name> --export=ALL,CMD="-m <module> ..." sbatch/run.sbatch`
  （含逗号的值改用环境变量传）；`sbatch sbatch/pi05_cuda.sh -m <module> ...`（带 CUTLASS 与
  tokenizer）；`PLAN=lab/plans/x.json sbatch sbatch/plan_e2e.sh`（correctness + A/B/A；PLAN 是
  环境变量，当参数传会被忽略而跑 shipped）；
  `sbatch/profile.sh`（latency + profile + floor）；ncu 节点见上。
- PR 规则（`.claude/rules/agent-notes-and-pr-workflow.md`）：一个 PR 一个决定；先看
  `git status --short --branch` 与完整 diff；最窄的可证伪验证；报告跑过的命令与拿不到的证据；
  note 同 PR 更新（`.agents/notes/{proposed,implemented,rejected}/{class}/YYYY-MM-DD-title.md`）。
- registry（`eval/acceptance.py`）：predbudget `{"candidates": 6, "non_improving": 3, "jobs": 12}`
  按 Target；`promotion_bar_ms` 0.10；`control_spread_max_ms` 0.10；`deployment.jitter_ms` 0.5；
  `stop.headroom_pct` 10；`soak_s` 10；shallow 容差 rel_rms_max 6.6e-2 / cosine_min 0.99978。
  `python -m eval.gate --target h100/pi05 --candidate lab/plans/<x>.json [--mode improve|no-regression]
  [--baseline]`，退出码 pass 0 / fail 1 / blocked 2；不带 `--baseline` 必 blocked。
- `benchmarks/targets.py`：`build(name, plan)`（seed 相同 → 权重逐位相同）、`declare(name, plan)`
  （只构图，登录节点可用）、`resolve`；plan 可以是 `shipped|reference|JSON|路径`。

### harness

`benchmarks/kernels.py`：`record_invocations` :47 记录一次 eager 段运行里 op 表的每次调用；
每个 call site（或 `atomic_groups` 成员元组 :110-115）成一个 case，标签
`"{segment}/" + "+".join(members)`；权重循环读取（冷）；计时器 `cudagraph`（一图 48 次内循环
/ 40 次重复）、`cupti`、`events`；`--site`、`--segment`、`--plan`、`--csv`；报告 median ms
与由 `engine.costs` 算出的 TFLOP/s、TB/s。`benchmarks/profile.py` `attribute()` :242：
每段 `valid, launches, total_us, wall_us, overlap_us, copy_us, call_sites{launches, dur_us,
names, ctas_min, under_one_wave, share}, unattributed`。`lab/pi05/enc_attn.py` 是 backbone
attention 的门；`lab/pi05/{kernels, attention_block, ffn_taskloop, ffn_full_chain_pdl,
xfs_producer, xfs_real_chain}.py` 全是 action expert；**没有任何 lab 脚本针对 vision**。

### action expert 的历史结论（lane D 的边界，不得重复的实验）

- 已交付：`implemented/architecture/2026-09-02-decoder-pdl-chain.md`（PDL 链，decoder 7.659 →
  7.391 ms；后续 kernel 的自计时含等待）；`2026-08-28-cooperative-xfs-pdl.md`（128-CTA 协作
  `tl_out_proj_residual_rms_xfs`，grid sync + 计数器复位 + PDL trigger）。
- 已否定（`rejected/architecture/`）：attention 持久化 task loop（1.01×，M=50 下 gmem join 太贵）；
  GU 冷 DRAM ceiling（9.25 us ≈ 冷 burst 上限 90%，ring 深度/事务数/CTA 数全部无效，L2 预热
  e2e 无效）；GU 会计（拷贝+等待占 9.6/11.6 us）；ring 深度 4/5 更慢；双权重流"免费"也只值
  2.39 us/层（0.43 ms 上限，关闭 warm/prefetch/retile/continuity 方向）；DR 链 10.5 us/层的
  link 价格（readiness poll 1.05、join 0.96、fold 1.37、residual 0.69 us），重排最好 0.041 ms；
  DR 宽 tile 更慢；DR L2 prefetch 更慢 2.2 us/层；rms→qkv 折叠 +0.083 ms、FFN 入口 trigger
  +0.562 ms；template 40 解释器型 megakernel 在每种深度都输给三次普通 launch；
  `coop.ratio.dev.relaunch` = 1.29（**不要用 grid barrier 换 launch**）。
- 待办：`.goal-task/pi05-latency-loop/state.md`：目标 14.5 ms 未达（16.07 ms），"megakernel
  跨层权重流连续性是命名的机制，当前超范围"；`todo#8` decoder attention split 7.65 us/约 1 MB
  是唯一未定价项，标为 megakernel 范围。`proposed/architecture/2026-09-03-ffn-gu-copy-column.md`：
  两个叠加的 floor（冷列 9.56 vs 机器 9.41；协议 floor ~7.7 us），evict-first L2 提示值 1.12 us/层。
- 代码：Pi0.5 `backends/cuda/`（`wrappers.py:179-412` 六个 call site；`attn_taskloop.py`
  `Workspace`/`launch_standalone`，ops 0-6；`kernels/attn_taskloop.cu` standalone kernels + PDL
  `cudaTriggerProgrammaticLaunchCompletion`/`cudaGridDependencySynchronize`，host
  `cudaLaunchKernelEx` + `ProgrammaticStreamSerialization`；`kernels/ffn_taskloop.cu` 持久化
  132 CTA，`reset_ffn_counters_kernel` 不得进图；`taskloop.py` 常量 `N_CTAS_FULL=132, FF=4096,
  D=1024, GATED_UP_CTAS=128, DOWN_RESIDUAL_TILES=32, DOWN_RESIDUAL_SPLIT=4, M_PAD=64`）；
  `cuda/__init__.py:25-42` ROUTE_CONSTRAINTS（qkv+attention 原子；GU+DR 原子；out_proj 依赖
  FFN 对）、`:45-68` graph_contract。`sm90_attn_task_desc.cuh:24-125`：M=50、M_PAD=64、
  H=8、DH=256、**PREFIX_LEN=968 编译期常量**、KEYS=1018/1024、N_CTAS=132、THREADS=192。
- Pi0 expert 与 Pi0.5 同宽（1024/4096/2560、8 头、head_dim 256、18 层）。差异（审阅核实，
  是 D0 的全部工作量）：行数 51（state token）；prefix 768，KEYS = 819，KEYS_PAD 仍 1024
  （`sm90_attn_task_desc.cuh:80-83`）；**Pi0 的 expert buffer 没有 padding**
  （`pi0/pipeline.py:140-148`：rows=51、cache_len 819、无 ROW_PAD、无 `mask_bias`），而 Pi0.5
  的 cuda wrapper 用 `_padded_base`（`cuda/wrappers.py:118-134`）拒绝未 pad 的 buffer，并硬检查
  `x.shape[0]==50`、`mask.shape[0]==KEYS`；Pi0 对同一个 STANDARD op 传
  `scale=None, bias=None, mask=None, gate=None`（`pi0/pipeline.py:161-175`；op 参数两边相同，
  `runtime/ops.py:182-197`，两个 Target 都没有扩展 OPS），而 cuda wrapper 会解引用这四个；
  XFS producer 是 Pi0.5 的 TileLang kernel（`cuda/wrappers.py:59`，`kernels/{xfs,adarms}.py`），
  写 K-major `(D, M_PAD)` 并复位 FFN 计数器，`graph_contract` 禁止 `reset_ffn_counters_kernel`
  进图（`cuda/__init__.py:57-60`），所以不能"退回普通 rms"；Pi0 另有扩展 op
  `action_expert_state_proj`、`action_expert_action_mlp`（`pi0/.../tilelang/wrappers.py:460-464`）。
  Pi0 shipped 是 `tilelang-fused`（`fused.py`：`tl_fused_rms_gate`、`tl_fused_rms_matmul_bias_res`、
  FlashDecoding `tl_fd_flat_split/combine`），无 PDL 链，无 ROUTE_CONSTRAINTS。
- `Registry.ops()`（`registry.py:51-58`）合并**所有已注册**后端的 OPS，不只是活跃后端。
- `eval/smoke.py:83-84` 从 plan 文件名前缀（`pi05-`/`pi0-`）推 Target，`_PI05_BACKENDS`（:37）
  硬编码，Pi0 的约束不检查：注册新后端时要同步改 smoke。
- 参考层级：T1 是 `src/flash_vla/models/<model>/reference.py`（pi05 只覆盖 expert attention
  block；pi0 没有），参考永远不是基线；parity 的 in-engine oracle 是 Target 的 reference plan
  （`eval.correctness`），kernel 级参考是 T2 ABI 镜像。
- 持久化基础设施：tile 库有 cluster multicast TMA、ClusterTransactionBarrier；没有协作/持久化/
  megakernel 脚手架；`sm90_attn_task_desc.cuh:168-243` 的 task-loop ABI（160 个 gmem 计数器，
  release→acquire 640 ns）；模板 04/10/40/42/43/44/45 可跑。

## 用户已确认的决定（2026-09-06）

1. **设备级组件包**：新增 `src/flash_vla/hardware/nvidia/h100/<component>/`（`siglip/`、
   `gemma_backbone/`、`gemma_expert/`），组件包持有 kernel 与 wrapper 工厂，Target 只注册路由；
   组件不 import Target。先出 PR0 改规则，再开 lane。
2. **过渡策略**：lane A 收口前，Pi0.5 候选满足候选规则 + 正确性门 + 基线层即可合入，尾部
   verdict 记为 blocked/fail 并引用尾部 lane 的 note；lane A 收口后统一补一次 `pass` 记录。
3. **lane D 范围**：D0 把 Pi0.5 的 cuda-pdl expert 链移植到 Pi0；D1 按 contract 给 megakernel
   定价并做单层原型，拿到数字后再决定是否投入整步 megakernel。

## 总体结构

```text
PR0  组件包规则（无 GPU，主会话，半天）
 ├─ lane A  Pi0.5 尾部归因           独立 worktree，立即开始，不依赖 PR0
 ├─ lane B  SigLIP 视觉编码器        依赖 PR0；两个 Target 共用
 ├─ lane C  Gemma 骨干 GEMM          依赖 PR0；两个 Target 共用
 └─ lane D0 expert 链移植到 Pi0      依赖 PR0；D1 依赖 D0 的数字与 lane A 的结论
```

- 每条 lane 一个 opus 子代理（`model: "opus"`），跑 kernel-design **auto** 模式：先交
  contract 给用户 ack，之后自主到停止条件，回来带证据包。主会话（协调者）只做：起 lane、
  审 contract、合并 promotion PR、维护 registry 与 note。
- 每条 lane 一个分支 `lane/<x>` 与一个 worktree `artifacts/worktrees/lane-<x>`（`git worktree
  add`），作业只从该 worktree 提交；lane 的工作区 `artifacts/ktasks/<task>/`。主树只做集成。
  **编译/profile 作业进行中不改该 worktree 的 kernel 源码**；要改就再开快照 worktree。
- 预算按 lane：registry 的 `budget {"candidates": 6, "non_improving": 3, "jobs": 12}` 在 PR0 里
  注明"按 lane（kernel-design task）计，不按 Target 计"；contract 只能收窄。
- 同节点 A/B/A；不挑节点，但 ACD1-1 的 control spread 在 8 次里 5 次超 0.10 ms，首跑 blocked
  就换节点重跑；ncu 只能在 ACD1-10/20/21/31/40/62。
- 合入顺序：每个 promotion 是一个 PR（一个决定），`eval.gate --candidate lab/plans/<x>.json
  --baseline` 的记录附在 note 里；Pi0 上要 `pass`；Pi0.5 上按过渡策略。合入本地 main，不 push。
- 汇报格式（每条 lane 结束）：跑过的命令与 job id、基线表、候选 ledger 摘要（kept/rejected +
  原因）、最佳候选的 parity 六指标与 A/B/A 三条 leg、gate verdict 与证据路径、拿不到的证据、
  还有预算时下一步。

## PR0 — 设备级组件包（主会话，先做）

- `ARCHITECTURE.md`：Dependency direction 加一行
  `hardware/<vendor>/<device>/<component>/ -> models/ + runtime/ + tile/；Target 可 import 组件包；
  组件包不 import 任何 Target`；"What a Target is" 补一句：Target 的 backends 可以由组件包的
  `make_wrappers(scratch, selected_names)` 工厂提供，Target 仍拥有路由、plan 与约束。
- `.agents/notes/implemented/architecture/2026-09-06-device-component-packages.md`：问题
  （SigLIP/Gemma 两个 Target 各一份且已分叉）、决定、替代（各自一份、只做 pi05）、后果
  （lane B/C/D0 的 kernel 只写一次；Pi0 的 vision tilelang 分叉版成为参考路由）、验证
  （`python -m eval.smoke`）。
- `eval/acceptance.py` budget 注释改为按 lane 计。
- `eval/smoke.py`：去掉硬编码的 `_PI05_BACKENDS`，改为从每个 Target 的 `registry.provided()`
  取后端集合，并对 Pi0 也做路由组合检查；plan 文件名规则（`<target>-<name>.json`）写进
  `lab/plans/README.md`。
- `runtime/ops.py`：`vision_encoder_norm_qkv` / `_norm_ffn_up` 的 `x_norm` 是 aux 输出且没有
  消费者（审阅核实），从词汇表删除，两边 tilelang wrapper 的签名同步（bit-identity 不变，
  `eval.correctness` 1×1 证明）。
- 不搬现有 kernel（搬迁与 lane 的 promotion 一起做，避免无收益的大 diff）。验证：smoke、
  `grep -rn "h100\.pi0\b\|h100\.pi05" src/flash_vla/hardware/nvidia/h100/<component>`（应为空；
  import 是点分路径，用 `hardware/nvidia/h100/pi0` 这种 grep 永远匹配不到）。

## lane A — Pi0.5 chunk 尾部归因（立即开始，不依赖 PR0）

问题文档：`.agents/notes/proposed/performance/2026-09-06-pi05-chunk-tail-attribution.md`（数据、
假设、实验、验收标准都在里面，执行者先读它）。要点：

1. 先加证据再猜：`benchmarks/latency.py` 每条 leg 记录逐次样本（不只 min/median/p99）、
   `/proc/self/status` 的 voluntary/nonvoluntary 上下文切换增量、`gc.callbacks` 带时间戳的
   collection 列表、10 Hz 的 `nvidia-smi --query-gpu=clocks.sm,clocks_event_reasons.active`
   与同卡进程采样；报告里新增 `attribution` 块。这是部署路径的 harness 改动，一个 PR。
2. 三个同节点决定性实验（`lab/pi05/tail_*.py` + `lab/sbatch/tail.sh`）：把 host slot 挪到
   `vision_encoder` replay 之前（`pi05/pipeline.py:136` 的 `g.host("prompt")` 位置，预计
   chunk min +0.15 ms）；`os.sched_setaffinity` 绑核；四个 pinned H2D 拷贝改成设备 staging
   的 D2D。每个实验一个 A/B/A（100 rep），同一作业内跑。
3. 验收：命名的机制 + 同节点 A/B；Pi0.5 两个 plan 连续三次 A/B/A `p99 − min ≤ 0.5 ms`，或写明
   部署要求；`eval.gate --target h100/pi05 --baseline` 出一次 `pass`；harness 保留归因记录。
4. 基线：在修掉 `gc.freeze()` 后的 runtime（main ≥ 678a136）上重新取，可比较的旧数据是
   598904–598959。

## lane B — SigLIP 视觉编码器（两个 Target 共用；依赖 PR0）

**目标**：`vision_encoder` 段 pi05 2.05 → 目标 1.3 ms、pi0 2.54 → 1.5 ms（上限 1.01）。
27 层 × 每层 9 launch（norm 2 + GEMM 4 + SDPA 3）≈ 250 launch，1.85 us 固定成本占 0.45 ms；
四个 GEMM 在 M=768 下只到实测 wgmma 上限的 33–47%，网格 108 CTA 不满一个 wave。

**contract 要点**（`artifacts/ktasks/siglip/contract.md`）：
- 张量（全部 bf16）：x (3, 256, 1152)，GEMM 视作 M=768；qkv_w (1152, 3456)、out_w (1152, 1152)、
  up_w (1152, 4304)、down_w (4304, 1152)，bias 各一；norm_w/norm_b (1152,)；attention 读打包
  qkv (3, 256, 3456)（Q|K|V 三块、块内 head-major）→ out (3, 256, 1152)，16 头 × 72，view 是
  batch 维、view 间无 attention；固定形状来自 `models/pi05/spec.py:37-40`。
- 融合区域：一个 call site 内（不改 op 词汇）：LayerNorm 折进 GEMM 的 A 侧——**BLOCK_M 上限 64**
  （64×1152×2 B = 144 KiB 驻留 smem，128 行 = 288 KB 放不下），M=768 只有 12 个 M tile，必须
  切 N，每个 N-CTA 各自重算它那 64 行的统计（多读一次 A，代价 147 KB/CTA，可接受）；
  bias/GELU/residual 折进 epilogue。
- 参考：T2 ABI 镜像 `siglip/backends/cuda/<kernel>_reference.py`（必做）；in-engine oracle 是
  reference plan（tilelang 路由）。parity：shallow 对 rel_rms_max 6.6e-2 / cosine_min 0.99978
  双门，用 `eval.correctness --target h100/pi05 --steps 1 --layers 1`（vision 完整跑）与
  `--target h100/pi0`；head_dim 72 pad 到 80 时 parity 必须覆盖 pad 区为 0。
- 基线（先测）：`python -m benchmarks kernels --target h100/pi05 --plan shipped --segment
  vision_encoder --timer cupti` 与 `--plan reference`，pi0 同；cuBLASLt（pi05 的 norm_qkv/
  norm_ffn_up）、TileLang（pi0）、torch.compile SDPA 三种基线都在表里。
- Ceiling：每个 call site 用 `benchmarks floor` 的 `ceiling_us_each`（tag `ld.bw.dev.dram` +
  `wgmma.clock.sm`），已在 job 598964 报告中；收益上限 pi05 1.04 / pi0 1.53 ms。
- Promotion：registry 规则；`eval.gate --candidate lab/plans/siglip-cuda.json`；Pi0 要 `pass`，
  Pi0.5 按过渡策略。

**候选阶梯**（一次一个，`candidates.jsonl` 记 parent/thesis/status）：
1. **B1 sm90 WS GEMM + 融合 prologue/epilogue**：模板 10_persistent_ws_gemm + 04_epilogue_persistent
   + 30_rmsnorm_residual（改 LayerNorm）；四个 GEMM 一个 kernel 模板四组实参（epilogue：bias /
   bias+residual / bias+GELU）；带 LN 折叠的两个 GEMM 用 64×128（或 64×256）tile，不带的两个
   可用 128×128；132 CTA 持久化调度；`wgmma.stages.wg.knee`（4 在飞）、`tma.stages.warp.knee`。
   预期 MFU 33–47% → 70%+，每 Target −0.5 ms。这一步单独就值得 promotion。
2. **B2 fused attention**：模板 12_attention_online_softmax；每 (view, head) 一个 CTA 组，
   seq 256、head_dim 72（wgmma K 步 16 → pad 到 80，或 `mma.sync` m16n8k16 直接吃 72 = 4.5×16
   不行，需 pad）；替换 torch.compile SDPA 的 3 launch（含 qkv 拆分转置）为 1 launch，直接从
   打包 qkv (768, 3456) 读。预期 11.2 → 4–5 us/层，−0.17 ms/Target。
3. **B3 层级融合**：norm_qkv + attention + out_proj 一个持久化 kernel（wiki `fusion-economics`、
   `pdl-placement` 先定价；launch 数 9 → 4）；只有 B1/B2 落地且 Analyze 显示 launch 固定成本
   仍占主导时才做。
4. **B4 PDL**：层内相邻 kernel 加 PDL 点（`producer-fusion-pdl`、`prefetch-across-dependency`），
   权重预取跨依赖；注意 `pdl-primary-is-a-resource`。

**交付**：`src/flash_vla/hardware/nvidia/h100/siglip/backends/cuda/{__init__.py (NAMES, OPS,
ROUTE_CONSTRAINTS=(), graph_contract), wrappers.py (make_wrappers(scratch, selected_names)),
kernels/*.cu, *_reference.py}`；Pi0.5 与 Pi0 的 `backends/__init__.py` 注册 `siglip-cuda`；
`lab/plans/{pi05,pi0}-siglip-cuda.json`；shipped plan 改路由（promotion PR）；
`benchmarks/kernels.py` 内置 case；note `implemented/architecture/2026-09-xx-siglip-cuda-backend.md`；
`eval/tile_sm90` 若新增原语要加 case。plan 文件名必须以 Target 前缀开头
（`pi05-siglip-cuda.json`、`pi0-siglip-cuda.json`），smoke 靠前缀识别 Target。
kernel-design 的每个任务在写代码前先有 `artifacts/ktasks/<task>/docs/draft.md`，跨两段的融合
（B3 若跨 stage）需要 T3 分解层参考。

## lane C — Gemma 骨干 GEMM（两个 Target 共用；依赖 PR0）

**目标**：`llm_backbone` pi05 6.75 → 5.5 ms、pi0 5.95 → 4.8 ms（上限 4.46 / 3.54）。
M=968（pi05）/ 768（pi0），K=2048，N=2560（QKV）/ 16384 gate+up / 2048 down。

**contract 要点**（`artifacts/ktasks/gemma_backbone/contract.md`）：
- call site 优先级（pi05 上限）：`llm_backbone_norm_gated_ffn` 205 → 153 us/层（0.89 ms；
  pi0 0.77）；`norm_qkv_rope` 32.7 → 11.9（0.37；pi0 0.26）；`attention` 22 → 9（0.22；pi0 0.33，
  pi0 每层 4 launch）；`ffn_down_residual` 83.8 → 76.4（0.13；pi0 0.51——pi0 用 tilelang
  `tl_matmul_res_ws`，pi05 用 cuBLAS `addmm_` 已到 91%）；`out_proj_residual` 0.06 / 0.15。
- 张量（全部 bf16）：x (968, 2048)；qkv_w (2048, 2560)；rope (968, 256) 交错 cos/sin；
  gate_w/up_w (2048, 16384)；down_w (16384, 2048)；q 输出 token-major `(M, 8*256)`，k/v 写进
  padded KV cache 的前导行视图，pair layout 与 scatter 见 `tl_matmul_rope_scatter`
  （base.py:640-700）。M 是 Target 的固定数（968 / 768），K、N 是模型常数。
- 融合区域：一个 call site 内；GU 的 gate 与 up 共享 A tile（dual-B），GELU-tanh·up 在 epilogue，
  绝不物化 2×hidden（63 MB 往返）；RMSNorm 折进 A 侧需要整行统计（K=2048，64 行 = 256 KB 超
  smem）→ 两遍或让上游 epilogue 产 row sumsq（跨 call site，先不做）。
- 基线：cuBLAS（`torch.matmul` 在 968×2048×16384 上先测；pi05 的 down/out_proj 已是 cuBLAS
  并到 91%）、TileLang 现有 kernel、`ffn_taskloop.cu` 的 `sm90_ffn_gemm.cuh` 原子。
- Ceiling 与上限：同 lane B 的来源；总上限 pi05 1.7 ms、pi0 2.0 ms。

**候选阶梯**：
1. **C1 GU dual-GEMM**：sm90 WS 持久化 GEMM（模板 10 + `sm90_ffn_gemm.cuh` 的 `Gemm<M,N,K>`），
   两个 B 流共享 A，epilogue GELU(gate)*up 写 bf16；tile 128×128×64、stages 4；968 行 → 8 个
   M tile × 128 个 N tile = 1024 tile 持久化调度；预期 75% → 90% 的 850 TFLOP/s，
   −0.6 ms/Target。若 cuBLAS 基线（两次 GEMM + 融合乘）已达 90%，先用 cuBLAS 路由做 promotion，
   C1 只在能赢 cuBLAS 时继续。
2. **C2 qkv + RoPE**：WS GEMM 版 `tl_matmul_rope_scatter`（非 WS 128/64/64 → WS 128/128/64 st4），
   epilogue 做 RoPE pair 旋转与 head-major scatter（模板 32_rope_layouts）；pi0 顺带把 3 launch
   合成 1（去掉 `scratch("encoder_qkv")` 往返）。
3. **C3 prefix attention**：`enc_attn.cu` 从 41% 提到 FA3 pingpong 水平（wiki `ext-fa3-pingpong`、
   模板 12；DH=256 时寄存器压力是主要约束）；pi0 同时从 torch 链换成同一 kernel（prefix 768，
   序列长度需模板化或运行期参数；pi0 的 backbone attention 也要 Pi0 pipeline 声明
   `mask_bias`/padding，与 D0 的 padding 改动合并）。
4. **C4 pi0 的 down/out_proj 换 cuBLAS**：零成本候选，pi0 上 0.5 ms 上限（先测 cuBLAS 基线即知）。

**交付**：`src/flash_vla/hardware/nvidia/h100/gemma_backbone/backends/cuda/…`（含 `enc_attn`
搬入时的 `PREFIX_LEN` 参数化）；两个 Target 注册 `gemma-cuda`；`lab/plans/*-gemma-cuda.json`；
note `implemented/architecture/2026-09-xx-gemma-backbone-cuda-backend.md`。

## lane D0 — Pi0.5 的 expert 链移植到 Pi0（依赖 PR0）

- 事实（见"action expert 的历史结论"最后一条）：同宽，但移植不是改一个常量。工作量：
  (a) Pi0 的 `pipeline.py` 为 expert 声明 padded buffer（rows 51 → M_PAD 64、cache 819 → 1024、
  `mask_bias`），图内声明 padding 是允许的不变量，reference 路由保持 bit-identity；
  (b) cuda wrapper 的 M / KEYS / PREFIX_LEN 从编译期常量改为按 Target 参数化（模板参数或按
  Target 编译两份 .so），`_padded_base` 与形状硬检查按参数走；
  (c) producer：Pi0 没有 AdaRMS，需要一个不带 scale/gate 的 xfs producer 变体（仍写 K-major
  `(D, M_PAD)`、复位 FFN 计数器、PDL trigger），wrapper 对 `scale/bias/mask/gate=None` 走该变体；
  (d) Pi0 每步的 `g.copy(ex[:1], est)` 与 `action_expert_action_mlp` 留在 tilelang，
  ROUTE_CONSTRAINTS 原样声明。
- 目标：Pi0 `action_expert` 7.70 → 7.4 ms（Pi0.5 的链是 7.39，且它还多做 AdaRMS）；
  上限 0.3–0.5 ms；预算 4 个候选、8 个作业，超出即停并写 rejected note。
- 步骤：组件包 `gemma_expert/backends/cuda/` 由 Pi0.5 的 `backends/cuda/` 搬入（一个 PR，
  bit-identity 用 `eval.correctness` 1×1 与 `benchmarks latency` 同节点 A/B/A 证明 Pi0.5 不变）；
  (a)–(d)；Pi0 注册 `expert-cuda-pdl`；`lab/plans/pi0-expert-cuda-pdl.json`；
  `eval.gate --target h100/pi0 --candidate … --baseline`（Pi0 基线层需要 checkpoint，机器上没有
  → 记 unavailable，Pi0 的 `pass` 用 in-engine 门 + 候选规则 + 尾部上界，note 写明）。

## lane D1 — expert megakernel 定价与单层原型（依赖 D0、lane A）

- 定价（contract 的 "Upper bound of gain"）：每层每步 45 us 实测；权重 32 MB/层 → 11.6 us
  （`ld.bw.dev.dram`）；约 10 个串行依赖 hop × 0.65–1.1 us（`atom.lat.dev.hop`、
  `coop.lat.dev.sync`）；`coop.ratio.dev.relaunch` 说明 grid barrier 换 launch 不赚；唯一未定价
  的机制是**权重流跨阶段/跨层连续**（producer warp 不等依赖就预取下一阶段/下一层的权重 tile，
  wiki `prefetch-across-dependency`、`stacked-floors`、`measuring-persistent-kernels`）。若定价
  后上限 < 1 ms，D1 到此为止，写 rejected note。
- 原型：单层五阶段一个持久化 kernel（模板 45_flag_barrier_megakernel + 43_mpk_task_graph_runtime
  的任务图；复用 `sm90_ffn_*`/`sm90_attn_task_desc` 的 task ABI 与 warp roles），只测一层的
  时间线（`lab/pi05/expert_layer_megakernel.py`），与 PDL 链的 45 us 同节点 A/B；不进 plan、
  不 promotion。
- 决策点：原型 ≥ 15% 快且路径清晰 → 立项整步 megakernel（18 层 × 10 步，新 lane，独立预算）；
  否则 rejected note，expert 的余量归于"依赖链延迟"，结案。

## 验证

登录节点（每条 lane 的每个 PR）：`python -m eval.smoke`；`python -c "import flash_vla"`；
`.venv/bin/python .claude/skills/kernel-design/scripts/check_wiki.py`（新增 wiki 条目时）与
`check_templates.py`（新增模板时）；`grep -rn "lab/" src eval benchmarks` 为空；组件包不
import Target（`grep -rn "h100/pi0\b\|h100/pi05" src/flash_vla/hardware/nvidia/h100/<component>`
为空）。

GPU（`sbatch -p acd_u --gres=gpu:1`，从 lane 的 worktree 提交，同节点 A/B/A）：
- 每个候选：parity（T2 镜像 + `eval.correctness --steps 1 --layers 1` 双门）→
  `benchmarks kernels --site <site> --timer cupti`（对基线表）→ `benchmarks latency --plan
  reference --plan lab/plans/<x>.json --plan reference --reps 100`（同进程 A/B/A，读 min）。
- promotion：`eval.gate --target h100/<t> --candidate lab/plans/<x>.json --baseline --reps 100`
  记录附 note；Pi0 要 `pass`；Pi0.5 按过渡策略；`benchmarks floor` 复跑一次记录新的
  within_ceiling。
- lane A 单独：三个实验的同节点 A/B/A + 连续三次 `p99 − min ≤ 0.5 ms` + 一次 `pass`。

## 风险

- ACD1-1 的 control spread 常超 0.10 ms：首跑 blocked 就换节点；不要为此放宽限制。
- lane B/C 的组件包搬迁若与 Pi0.5 现有 kernel 冲突（`enc_attn` 归 backbone 组件，
  `attn_taskloop`/`ffn_taskloop` 归 expert 组件），按 promotion PR 逐个搬，每次 bit-identity。
- head_dim 72（SigLIP）不是 16 的倍数：wgmma K 步需 pad 到 80，parity 必须覆盖 pad 区为 0。
- Pi0 的官方基线层没有 checkpoint：Pi0 的 `pass` 依赖 in-engine 门；note 写明。
- 多 lane 并行时 Slurm 队列可能排数小时：每个 lane 的作业合并成尽量少的 sbatch（一个作业跑
  parity + kernels + latency）。
