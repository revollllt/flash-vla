# 模型性能优化 workflow

目标是通过持续的 kernel 设计与优化，逼近目标硬件和实际 shape 下的 SOL（Speed of Light），并将收益落实到端到端延迟。优化循环为 **Profile → Analyze → Design → Implement → Validate → Integrate → Profile**。已有本轮同代码、同条件的测量、正确性结果和 profile 可以复用。

1. **准备模型和环境。** 从项目根目录运行以下命令，文中的项目文件路径均相对项目根目录。沿用模型已有的 Python 环境、真实 checkpoint、输入和运行参数，在目标 GPU 上执行；使用 Slurm 集群时通过作业分配 GPU。以下以 LingBot 为例，使用已有的 `FLASH_VLA_ASSETS` 配置；其他模型使用对应 Target 和资产参数。机器上的资产位置由本地配置提供。

2. **测当前版本的端到端延迟，确认输出正确。** 以本轮当前版本为比较起点，用已有数值参考和容差确认输出；已有结果就复用。`shipped` 表示当前发布方案，`reference` 表示数值参考实现。

   ```bash
   python -m benchmarks latency --target h100/lingbot_vla --plan shipped --seed 42 --out artifacts/before.json
   ```

   默认 warmup 5 次、测量 100 次，以 median 比较并保留原始样本，不做 soak。每个版本独立进程、只用首次 capture，每个 graph 始终在自己的 capture stream 上 replay。计时包含输入搬运、host 工作、replay 和末尾同步，不包含模型加载与 capture。

3. **理解模型的推理路径。** 沿着 `vision_encoder → llm_backbone → action_expert` 阅读实现，理清各部分的输入 shape、精度、调用次数和中间结果复用。确认当前已经用了哪些优化，尤其关注 denoising 循环中的重复计算。

4. **Profile：从整个模型定位到关键 kernel。** 保持与 benchmark 相同的输入、精度、执行路径和 CUDA Graph 设置，用 Torch Profiler 或 Nsight Systems 查看完整 forward：时间主要花在哪个模块，是否存在 host 调度、同步或 GPU 空隙。再对主要耗时模块做 call-site / kernel 分析，明确关键 kernel 的耗时、调用次数及相互依赖；需要硬件计数器定位原因时，再使用 Nsight Compute / `ncu-report`。

   ```bash
   python -m benchmarks profile --target h100/lingbot_vla --plan shipped --seed 42 --overview --trace-dir artifacts/profile/overview
   # 若整体 profile 指向 action_expert，再深入这个模块。
   python -m benchmarks profile --target h100/lingbot_vla --plan shipped --seed 42 --segment action_expert --trace-dir artifacts/profile/detail
   ```

   看 GPU 时间线判断耗时，CPU segment 标签仅表示提交范围。Profile 用于诊断，正式延迟另起无 profiler 的进程测量。

5. **Analyze：判断距离硬件 SOL 还有多少空间，选择优化假设。** 对关键 kernel 或相依算子链，结合 FLOPs、最低必要访存量和硬件算力/带宽估计理论下界 `max(FLOPs / 算力, bytes / 带宽)`，再用实际 shape 下的实测硬件能力和 profile 判断可达水平。优先复用已有 `benchmarks floor` 报告及 `src/flash_vla/hardware/<vendor>/<device>/measured/` 数据；缺少适用数据时，只针对当前问题估算或测量，并标明不确定性。

   区分计算、访存、launch、同步或流水线瓶颈，用“每次可节省时间 × 调用次数”粗估端到端价值，考虑重叠与关键路径。先查 Pi0、Pi0.5、其他 Target 和共享组件的已有实现，以及 `kernel-wiki` 中对应架构的经验，再选一个有依据、收益值得尝试的假设。已有 kernel 也要根据剩余空间继续优化；floor/ceiling 是参考，不能仅凭一个比例宣告达到 SOL。

6. **Design / Implement：手写 kernel，按瓶颈持续迭代。** 优先复用或适配已有 CUDA/TileLang 实现；融合优化通过手写融合 kernel 完成，不把 `torch.compile` 自动融合当作交付方案。现有编译融合路径可以作为替换前的数值和性能对照。

   计算瓶颈重点研究 tile 形状、Tensor Core 利用率和 warp/CTA 分工；访存瓶颈重点研究数据布局、合并访问、shared memory/寄存器复用及异步搬运；流水线瓶颈重点研究计算与搬运重叠、同步和资源占用。同时考虑将 normalization、activation、residual、epilogue 等相邻计算手工融合，消除中间张量、reshape/shuffle 和重复计算。根据瓶颈选择设计，不逐项机械尝试。

   每次先用最小改动验证一个假设，以相关数值检查和 kernel microbenchmark 筛选候选；计时对齐实际 shape、数据布局和缓存条件，需要时复核 in-graph 表现。只有有希望的候选进入完整模型验证。根据耗时和按需采集的 NCU 证据调整设计，直至该瓶颈没有值得继续尝试的方案，再转向下一个瓶颈。

7. **Validate / Integrate：确认端到端收益，接入当前最佳方案，再分析。** 沿用现有参考和容差，检查受影响的算子及模型输出，覆盖相关真实 shape，并按需要补充独立输入。共享 runtime / 同步改动做相关 GPU 集成检查，近似或语义改动补充任务质量评估。只验证本次改动涉及的部分，不放宽精度要求。

   独立测量修改后的完整模型：重复第 2 步命令，结果另存为 `artifacts/after.json`；使用候选 plan 时替换 `--plan`。前后保持同一物理 GPU、驱动/runtime、checkpoint、输入、shape、精度、计时范围和无竞争负载的条件。默认分别测 A、B；有漂移迹象才针对性复测，必要时用 A/B/A。收益接近已观察到的波动时先记为不确定，换环境后的差异不能算作代码收益。

   正确且端到端有收益就接入当前最佳 plan，否则撤销本次候选改动或记录待确认问题。比较 kernel、模块和端到端的收益是否一致：局部变快但整体未改善时，回到第 4 步检查关键路径、重叠、调度和测量条件。收益成立后重新判断剩余瓶颈与硬件能力的差距，再进入 Analyze → Design → Implement；执行路径或瓶颈变化时更新整体 profile，其余时候复用已有证据。

   留一条简短记录：假设与改动、代码版本或 diff、命令与环境、正确性及延迟结果、结论，并保存原始 JSON。只有主要热点已接近有证据支持的可达能力且没有值得尝试的新方案，或用户范围/时间预算要求结束时，才停止；一次优化成功或达到某个 floor 比例不代表循环完成。

日常迭代无需 Campaign 初始化、context activation、re-anchor 或完整 gate；不新增 hash、冻结 contract 或审批步骤。历史审查工具按明确需要使用，绘图和发布集中在有意义的阶段完成。
