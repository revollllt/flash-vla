# 模型性能优化 workflow

按下面 7 步迭代。已有本轮同代码、同条件的测量、正确性结果和 profile 可以复用。

1. **准备模型和环境。** 在 `lab-H100` 的 `/data/user/jzou521/codes/cuda/flash-vla` 中，沿用模型已有的 Python 环境、真实 checkpoint、输入和运行参数。GPU 命令在 Slurm 分配的 GPU 上运行。以下以 LingBot 为例，使用已有的 `FLASH_VLA_ASSETS` 配置；其他模型使用对应 Target 和资产参数，确认加载的是目标 checkpoint。

2. **测当前版本的端到端延迟，确认输出正确。** 以本轮当前版本为比较起点，用已有数值参考和容差确认输出；已有结果就复用。`shipped` 表示当前发布方案，`reference` 表示数值参考实现。

   ```bash
   python -m benchmarks latency --target h100/lingbot_vla --plan shipped --seed 42 --out artifacts/before.json
   ```

   默认 warmup 5 次、测量 100 次，以 median 比较并保留原始样本，不做 soak。每个版本独立进程、只用首次 capture，每个 graph 始终在自己的 capture stream 上 replay。计时包含输入搬运、host 工作、replay 和末尾同步，不包含模型加载与 capture。

3. **理解模型的推理路径。** 沿着 `vision_encoder → llm_backbone → action_expert` 阅读实现，理清各部分的输入 shape、精度、调用次数和中间结果复用。确认当前已经用了哪些优化，尤其关注 denoising 循环中的重复计算。

4. **先做端到端 profile，再深入主要瓶颈。** 保持与 benchmark 相同的输入、精度、执行路径和 CUDA Graph 设置，用 Torch Profiler 或 Nsight Systems 查看完整 forward：时间主要花在哪个模块，是否存在 host 调度、同步或 GPU 空隙。然后只对最值得优化的部分做 call-site / kernel 分析；需要硬件计数器时，再对关键 kernel 使用 Nsight Compute / `ncu-report`。用“该部分耗时 × 预计可降低比例”粗估端到端收益，选一个有依据、值得尝试的瓶颈。

   ```bash
   python -m benchmarks profile --target h100/lingbot_vla --plan shipped --seed 42 --overview --trace-dir artifacts/profile/overview
   # 若整体 profile 指向 action_expert，再深入这个模块。
   python -m benchmarks profile --target h100/lingbot_vla --plan shipped --seed 42 --segment action_expert --trace-dir artifacts/profile/detail
   ```

   看 GPU 时间线判断耗时，CPU segment 标签仅表示提交范围。Profile 用于诊断，正式延迟另起无 profiler 的进程测量。

5. **优先复用已有的高效 kernel。** 针对该瓶颈，先查 Pi0、Pi0.5、其他 Target 和共享组件已有的实现，确认 GPU 架构、shape、dtype 和算子语义是否适用。优先直接复用或小幅适配；没有合适实现时再设计新 kernel。每次围绕一个明确假设做最小改动，例如“复用已有 fused kernel，减少这组算子的显存往返”。

6. **寻找融合和消除重复计算的机会。** 检查热点中的相邻算子，优先尝试数学等价的融合，以及复用推理过程中不变的结果，减少 global memory 读写、中间张量、reshape / shuffle 和 kernel launch。第 5、6 步按瓶颈选择，不要求每轮两项都做。

7. **验证质量和速度，保留有效改动，继续迭代。** 沿用现有参考和容差，检查受影响的算子及模型输出，覆盖相关真实 shape，并按需要补充独立输入。共享 runtime / 同步改动做相关 GPU 集成检查，近似或语义改动补充任务质量评估。只验证本次改动涉及的部分，不放宽精度要求。

   独立测量修改后的完整模型：重复第 2 步命令，结果另存为 `artifacts/after.json`；使用候选 plan 时替换 `--plan`。前后保持同一物理 GPU、驱动/runtime、checkpoint、输入、shape、精度、计时范围和无竞争负载的条件。默认分别测 A、B；有漂移迹象才针对性复测，必要时用 A/B/A。收益接近已观察到的波动时先记为不确定，换环境后的差异不能算作代码收益。

   正确且端到端有收益就保留，否则撤销本次候选改动或记录待确认问题。留一条简短记录：假设与改动、代码版本或 diff、命令与环境、正确性及延迟结果、结论，并保存原始 JSON。瓶颈未变就继续第 5、6 步；瓶颈变化或局部收益未体现在端到端时回到第 4 步。按用户范围和时间预算结束。

日常迭代无需 Campaign 初始化、context activation、re-anchor 或完整 gate；不新增 hash、冻结 contract 或审批步骤。历史审查工具按明确需要使用，绘图和发布集中在有意义的阶段完成。
