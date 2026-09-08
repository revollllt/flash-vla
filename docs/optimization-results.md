# 优化计划执行结果（2026-09-07–08）

执行仓库为 lab-H100 的 `/data/user/jzou521/codes/cuda/flash-vla`，起始
HEAD 为 `cc2289d`。工作覆盖 v0.2 的首版适用范围：现有 H100 / Pi0、Pi0.5、
一个组件的双版本原型和 CUDA 语义 tracing。原始计划及 starter 保留在
`artifacts/optimization_inputs/`。逐项状态见 `optimization-checklist.md`。

## 实验结论

| 实验 | 直接证据 | 结论及限制 |
|---|---|---|
| QKV 的 N tile 64→128 | 两个独立 job；每个版本每 job 1,200 个原始样本；Q/K/V 逐位一致 | 加宽方案回退，不晋升；没有把回退单独归因于寄存器、事务数或 wave 数 |
| TMA 等在途字节量对照 | 两个独立 job；相同 32 CTA、单 producer、L2 footprint、总传输字节量；现有 probe 的数据校验通过 | 字节量不足以解释成本；事务数是需要定价的变量，不能据此断言生产 vision kernel 的唯一瓶颈 |
| 真实 backbone attention tracing | 363 个 coarse / 126 个 focused 区间；实际第一层输入和输出；独立 graph replay buffer | 语义和存储正确，但插桩有明显扰动；区间不是硬件 active time |
| 同进程 A/A′/B | 三个独立动态库、编译几何、实际加载地址；Pi0 与 Pi0.5 两个 shape；A/A′ 逐位一致 | 支持这个 CUDA attention 组件的隔离原型；B 的小差异不足以支持优化收益 |
| 代表性 PDL 链 | job 602515 的同一条 replay trace 和另外测得的无 profiler stage 时间 | duration sum、interval union、makespan 分列；不同 capture 的差值不称为 overlap |

QKV 使用 `(M,N,K)=(968,2560,2048)`、bf16、相同 round-then-RoPE 语义、
warp specialization 关闭，只改变 N tile。十套权重共 100 MiB，按固定顺序
轮换；每组采用预先规定的 ABBA/BAAB，未按结果追加采样。

| QKV job | BN64 min / median（µs） | BN128 min / median（µs） |
|---|---:|---:|
| 602586 | 27.971 / 29.165 | 42.432 / 44.096 |
| 602588 | 28.397 / 29.254 | 43.341 / 44.470 |

这是 isolated、冷权重条件的反证，未转换成 E2E 收益。Pi0 当前 QKV 是不同的
实现路径，不能直接套用该结论。由于候选明显回退，没有必要花预算做晋升 gate。

TMA 的 32 KiB/CTA 等在途字节量子集，在 job 602552/602553 中从
`16 stages × 2 KiB` 改为 `2 stages × 16 KiB`，相同总字节的 CUPTI 时间
分别由 260.540→33.727 µs、262.523→33.920 µs；各点每事务约 254–265 ns。
这些点的总传输均为 64 MiB，L2 footprint 为 4 MiB。环深与 box 大小联动、
barrier 元数据略有变化，probe 排除了 WGMMA 和真实消费者；不把它当作完整
vision kernel 的加速实验，也不覆盖现有硬件 constants 表。

真实 attention 在 job 602528 的 off/coarse/focused 各 600 个样本中，
中位数分别为 26.048 / 29.984 / 32.992 µs。Focused 虽只观测 CTA 0，仍可能
影响整次 launch 的尾部，因此不能按记录数量推断扰动更小。独立 off 复验
job 602547/602559 中位数为 25.280 / 25.696 µs；不同 node/driver 的数值
不能相减为优化收益。插桩前后均为 240 registers、无 stack/local spill，
这同样说明“寄存器数相同”不足以证明无扰动。

TMA marker 记录 issue 调用前到消费者已有 wait 返回后的窗口。WGMMA marker
按现有每轮 score/P.V 两个 commit 的顺序编号；score 的 wait<0> 也覆盖前一
轮 P.V，末轮使用已有 drain。64 个配对包含 32 TMA 和 32 WGMMA；它们是
观测窗口，不是精确 commit-to-completion 活跃时间。未增加同步。

## 正确性与可视化

- 59 个不同 CPU 用例已分批通过，覆盖路由、时间线、失败早停、恢复、预算、
  并发预算预留、失效证据、alias、trace 配对和导出；没有修改 acceptance 门槛。
- Demo jobs 602410/602503：1/64/512 CTA parity、memcheck、off SASS 无 timer/SM
  读取；off 为 26 registers，trace 为 32，均无 spill。512 CTA 观测到 132 个 SM。
- Demo 观察到最小正 timer 差为 32 ns，同时存在零差；这不是保证精度。
  off/coarse/focused 各 600 个固定样本的中位数约为 61.677/61.744/61.699 µs。
- 真实 attention jobs 602528/602547/602559：逐位 parity、生产 fixture 输出
  复现、memcheck 与 synccheck 零错误，两次 graph replay 不覆盖先前记录。
- GPU job 602589：快照不受原 buffer 后续 mutation 影响，stride 和重叠 alias 保留。
- Perfetto v58.3 已实际导入 GPU `cta1.perfetto.json`：显示 scalar_math 和
  wmma_scope 两条角色轨道，后者持续 2.752 µs；UI 中 synthetic=false、
  hardware_active=not_measured。此前文件选择的工具阻塞已解除。

## 控制流程收益和成本

针对相同、预先给定的“正确性失败”和“官方证据缺失”两类输入，实际加载旧版
与新版 gate，在 CPU mock 下各运行六组。旧版每次继续调用 latency 和 floor，
新版均为零次；fail/blocked 结论一致。这里只证明避免了无意义调用，不估算
因此省下的 GPU 秒数，也不声称已经测得 Agent 的整体生产率提升。

另一个真实 CPU 小任务比较了直接执行与控制层的 start/source-copy/index/run。
六个配对区组中，直接执行约 15–21 ms；控制层首轮 3.835 s，后续 0.840–1.239 s。
源码复制、Git 状态和共享文件系统持久化的合计开销明显，不适合逐个包裹极短
CPU 查询。该结果包含全部样本，不扣除首轮，也没有归因某一个 I/O 原因。
廉价只读筛选可以直接调用现有工具；控制层用于需要持久阶段、GPU 归属和恢复的任务。

截至这些实验，Slurm 共记录 18 个 job，分配 GPU 时间合计 967 秒（16 分 7 秒），
包含六个失败 job 的 173 秒。它不是纯 kernel 时间，也不是开发总时长。
失败分别是 compute node 缺 git、缺 tokenizer、浅图路由校验过严、Slurm/可见
GPU ordinal 不一致、复制模块相对导入缺 package、TVM-FFI 新编译对象没有 libpath。
这些被归类为环境或实现问题，未写成机制失败。每任务 job 数均未超过其预算，
组件隔离任务的两次 job 也未越过其自设上限。

## 适用边界与后续重开条件

本轮没有合格的性能候选，不执行 shipped 晋升或组合候选的 GPU qualification。
这两项的执行前提不存在；已有适用性与共享 Target 覆盖检查仍已测试。
跨源码版本的正式 `eval.gate` qualification **尚不支持**，不会因为两个诊断
动态库可共存就假称 evaluator 已加载两个版本。当前支持的是同一源码树的 route
对照，并在评估前后检查显式源输入及验收值。源码版本、host 隔离和其他 backend
需另行接入；自动晋升始终关闭。

源输入只复制声明的依赖；manifest 明确不代表完整 runtime snapshot。未列出的
第三方、驱动和运行环境依赖仍在覆盖范围之外。未知身份不判成已知 no-op。
现有 NCU action 是 ffn_taskloop_kernel，已在上下文中标为不适用于 attention/QKV；
没有用它的计数器解释这里的 kernel。

重开 QKV 加宽方向需改变当前导致回退的组合条件，或提出能区分 epilogue/occupancy
与数据复用的新反证；重复旧条件应明确标为 replication。重开 tracing 的定量归因
需降低实际扰动或增加独立硬件证据。TMA 微观结论迁移到 vision 时仍需真实 mainloop
对照，不能靠历史 note 或该孤立 probe 替代。

完整原始数据分别位于 `artifacts/optimization/` 与 `artifacts/kernel_trace/`。
汇总入口为 `campaign-summary.json`、`job-accounting.txt`、`evidence-context.json`、
`workflow-comparison.json` 和 `controller-cost/comparison.json`。
