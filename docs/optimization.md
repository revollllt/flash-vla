# 模型性能优化 workflow

通过手写 kernel 优化逼近实际 shape 下的硬件 SOL，并降低部署后的端到端延迟。采用双循环：

- **Kernel 内循环，可多 agent 并行：** Profile → Analyze → Design → Implement → Validate，按结果继续迭代。
- **Model 外循环，串行执行：** Profile → Analyze → Design → Implement → Validate → Deploy → Profile。

1. **准备模型与环境。** 从项目根目录运行命令，沿用目标模型的 Python 环境、真实 checkpoint、固定输入及运行参数，在目标 GPU 上执行。已有 Target 直接使用；新增模型时用 [target-onboarding](../.claude/skills/target-onboarding/SKILL.md)。以下命令以 LingBot 为例，需配置已有的 `FLASH_VLA_ASSETS`。

2. **测当前模型并确认输出。** 以当前 `shipped` 版本为比较起点，沿用已有数值参考和容差；本轮同代码、同条件的测量和正确性结果可以复用。

   ```bash
   python -m benchmarks latency --target h100/lingbot_vla --plan shipped --seed 42 --out artifacts/before.json
   ```

   前后保持同一物理 GPU、驱动/runtime、checkpoint、输入、shape、精度和计时范围。各版本独立进程、首次 capture、一个 graph 固定一个 stream；默认 warmup 5 次、测量 100 次、无 soak，以 median 比较并保留原始样本。计时包含输入搬运、host 工作、replay 和末尾同步，不含加载与 capture。A、B 分别测，有漂移迹象才针对性复测。

3. **理解模型结构。** 结合 [architecture](../ARCHITECTURE.md)，沿 `vision_encoder → llm_backbone → action_expert` 阅读推理路径，理清 shape、调用次数、已有优化和 denoising 循环中的重复计算。

4. **自顶向下 profile，选择值得优化的瓶颈。** 用 [gpu-profiler-analysis](../.claude/skills/gpu-profiler-analysis/SKILL.md) 看完整 forward 的 GPU 时间线，先定位耗时模块或 host/同步空隙，再深入该模块的 call site 和 kernel。按需用 [ncu-report](../.claude/skills/ncu-report/SKILL.md) 判断计算、访存和流水线瓶颈；结合已有 `tools.profiling.floor` 报告和 [hardware-unit-test](../.claude/skills/hardware-unit-test/SKILL.md) 的实测数据估计距可达 SOL 的空间、调用次数及端到端收益，缺数据才做针对性测量。

   ```bash
   python -m tools.profiling.model --target h100/lingbot_vla --plan shipped --seed 42 --overview --trace-dir artifacts/profile/overview
   # 仅当整体分析指向 action_expert 时，深入该模块。
   python -m tools.profiling.model --target h100/lingbot_vla --plan shipped --seed 42 --segment action_expert --trace-dir artifacts/profile/detail
   ```

   Profile 与 benchmark 使用相同输入和执行配置；看 GPU 耗时，CPU segment 标签仅表示提交范围。正式延迟另起无 profiler 的进程测量。

5. **并行设计和实现 kernel。** 将不同 kernel 或独立融合链分给多个 agent，在独立 worktree 中用 [kernel-design](../.claude/skills/kernel-design/SKILL.md) 迭代。先查 Pi0、Pi0.5、共享组件已有实现及 [kernel-wiki](../.claude/skills/kernel-wiki/SKILL.md)，再围绕一个有依据的假设做最小改动。按瓶颈优化 tile、数据布局、Tensor Core 利用率、访存和流水线；手写 CUDA/TileLang 融合 normalization、activation、residual、epilogue 等计算，减少中间张量和重复工作，不以 `torch.compile` 自动融合作为交付方案。

6. **Kernel Validate：确认正确性和局部收益。** 沿用现有参考与容差检查相关 shape/输入，用 [benchmark-kernel](../.claude/skills/benchmark-kernel/SKILL.md) 对齐实际数据布局、缓存和执行条件，比较 kernel 或完整融合链的耗时。失败或收益不确定就留在内循环分析、修改；**正确且有可信局部收益才交给 Model 外循环**，无需先优化到极限。同一 GPU 的性能测量串行；多 GPU 可各自做同卡前后对比，model 正式计时期间避免其他 agent 争用测量资源。

7. **Model 串行集成、验证、部署和测量。** 每次只把一个胜出候选接入当前最佳模型和 plan，做受影响的模型正确性检查；近似改动补充任务质量评估。验证通过后 Deploy 到实际推理路径，再运行第 2 步命令测部署性能，使用实际加载的 plan，结果另存 `artifacts/after.json`。有端到端收益就保留，否则回退或记录不确定，再处理下一个候选。接受 K1 后，K2 比较 `M+K1` 与 `M+K1+K2`，逐个确认增量收益。

   局部收益未传递到模型时，回第 4 步分析并反馈内循环；收益成立后由部署版本选下一轮热点。候选适配最新模型，相关条件变化才补验证；瓶颈未变可复用 profile。按用户预算持续迭代，主要热点接近有证据支持的可达能力且无值得尝试的新方案时结束。

每轮只留改动、代码版本或 diff、命令与环境、正确性及性能结果、结论，并保存原始 JSON。Skill 按需使用，不要求每轮全部执行；不新增 hash、冻结 contract 或 gate，不做无关全量审查。历史 Campaign 与发布流程仅在明确需要时使用。
