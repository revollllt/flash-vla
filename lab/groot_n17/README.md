# GR00T N1.7 / RTX 5090 Roofline 估计

[中文](README.md) | [English](README.en.md)

2026-09-17；参考实现为 `cf2d196`。本目录随主仓库 Git 管理，由四组的共同起点提供，计算采用项目现有 `Cost`、`gemm`、`attention` 和 `tools.profiling.floor.site_row`，未修改推理代码。

**当前 dense 计算路径的条件估计为 8.261 ms。达到该估计性能的 90%，对应完整模型延迟 ≤9.179 ms。** 这是优化目标，可达性尚未证实，预算耗尽仍停止。

| 阶段 | Dense FLOPs | 流式权重读取 | 理想延迟估计 |
|---|---:|---:|---:|
| 视觉 | 349.503 GFLOPs | 0.808 GB | 1.667 ms |
| 语言 backbone | 254.445 GFLOPs | 1.611 GB | 1.214 ms |
| 动作头 | 469.315 GFLOPs | 9.203 GB | 5.380 ms |
| 总计 | 1.073 TFLOPs | 11.622 GB | **8.261 ms** |

每个算子的估计是 `max(FLOPs / 209.6 TFLOP/s, 权重字节 / 1.792 TB/s)`，按实际调用次数求和。采用 5090 的 BF16 输入、FP32 累加 dense 峰值；硬件来源为项目 `rtx5090/spec.py`，未使用 FP8、稀疏或 FP16 累加的吞吐数字。

Workload 为 `libero_10`、BF16、batch 1、两路 256×256、156 tokens、16 层 backbone、32 个 DiT blocks、4 步去噪。内部输出 `[1,40,132]`。计算覆盖每个新观测的视觉、语言和完整动作路径。

主要假设：

- 激活能理想地在缓存/片上复用；权重矩阵每次调用流式读取一次，不假定跨调用常驻。四步去噪中的重复读取已计入。只读取当前 embodiment 的权重，不把 32 个 bank 全部计入。
- 不把整个词表作为 embedding lookup 的流量。lookup、bias、norm 等小算子、调度、同步和输入 staging 开销尚未计入；图像预处理、CPU→GPU 输入传输及动作解码仍在实验计时范围外。
- 96 MiB L2 中能保留哪些权重尚未实测。常量折叠、跨步 KV 复用、融合或执行顺序改变后，应重新评估计算量与访存假设。8.261 ms 不是所有等价实现的严格物理下限。

如果连中间激活也全部按 DRAM 读写定价，估计是 8.458 ms，作为访存假设的敏感性比较。套用现有实测 primitive 常数则得到 12.736 ms，其中包含逐调用冷读固定成本；它的计算吞吐是在不同频率条件下测得，也不是已测得的模型可达延迟。两者均不替代上述明确选定的目标口径。

直接运行通用 `tools.profiling.floor` 得到 5.154 ms 且 `valid=false`：三阶段声明过粗，未计完整循环流量；工具还将规格频率下的 209.6 TFLOP/s 与实测频率下的 253 TFLOP/s 按同条件校验。**未使用这个无效报告设置目标**，也未改门限让它通过。

展开后的矩阵/attention FLOPs 与三个阶段声明一致，并用一次真实 eager forward 的 `FlopCounterMode` 核对。语言阶段额外的 59,904 FP32 FLOPs 来自 RoPE 频率外积，已单独记录；最终 `[1,40,132]` 输出为有限值。访存量仍是模型假设，不是这次 FLOP 检查测出的 DRAM 字节。

复算（从任意 Flash-VLA checkout 根目录，使用 GR00T 环境；无需 GPU 或权重）：

```bash
PYTHONPATH=src:. python -m lab.groot_n17.roofline --out artifacts/groot-n17/roofline.json
```

共同数值记录为 [roofline.json](../../results/groot-n17-rtx5090/reference/roofline.json)，真实 forward 检查为 [roofline-execution-check.json](../../results/groot-n17-rtx5090/reference/roofline-execution-check.json)。脚本为 [roofline.py](roofline.py)。四组从同一共同提交继承这些文件；启动 prompt 统一引用本页，以 8.261 ms 为原始比较分母、≤9.179 ms 为目标。运行中若发现建模问题，应记录修正依据，汇总时统一处理，不能只替某一组更换比较分母。
