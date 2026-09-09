# LingBot Target onboarding retrospective

本复盘只依据 LingBot-VLA-4B 在 H100 上从 requirement freeze 到真实优化晋升的
证据。它不把一次接入中没有发生的问题外推成通用需求。

## 结论

现有 Target、plan、三段 ModelRunner、Identity V2、Campaign Ledger、gate 和
canonical trace 足以表达 LingBot。接入与首个优化没有遇到需要修改 generic
runtime 才能正确表达的 blocker，因此本轮不提取新的 shared runtime component，
也不实现通用 `flash-vla onboard` generator。

LingBot 的冻结输入、兼容性报告、official oracle、四层 baseline、floor/profile、
Campaign 与验收状态保存在
`artifacts/onboarding/lingbot-vla-4b-h100-bf16` 和
`artifacts/optimization/lingbot-vla-4b-h100-bf16`。这些持久化事实，而不是本复盘，
是后续 session 的恢复依据。

## 哪些现有 abstraction 完全够用

- Target identity 能完整表达 H100、冻结的 LingBot/Qwen revisions、Robotwin shape
  和 BF16，并把 plan 与 engine revision 排除在 workload identity 之外。不同模型
  revision、shape 或 precision 不能进入同一 Campaign lineage。
- Target-local route registry 能同时保留 `reference`、`shipped` 和候选 route。
  `eval.gate` 因此可以在同一源码树中完成 production path 的 A/B/A。
- 三段 ModelRunner 正好承载 vision、prefix KV 和 10-step action denoise。无需增加
  model-specific runtime stage 或 scheduler 分支。
- correctness ladder、baseline ladder、floor/profile、Campaign Ledger 和唯一
  Matplotlib renderer 都可直接复用。删除 derived state/trace/plot 后，ledger 仍可
  重建相同语义。
- 软件 range trace 的 identity 与保守解释规则已经足够用于诊断：launch/replay、
  CTA/warp role 和 async observation 各自保留；`smid` 只是属性，未观测区间不写成
  idle。其已测扰动使它不参与 promotion 数值归因。

## 哪些只需要 Target-local implementation

- checkpoint schema、权重映射、Qwen processor/tokenizer 接线与 Robotwin 的
  normalize/unapply 语义；这些都来自冻结 upstream，不属于 runtime invariant。
- LingBot 的 model package、shape declaration、三个 pipeline stage、official eager/
  compile adapters、fixture 与 acceptance tolerance。
- RoPE inverse-timescale cache。它依赖 LingBot upstream 的调用边界和共同加载模型，
  因而保持为三个 LingBot call site 的 atomic route。campaign iter-004 在 job 606866
  以 0.019 ms control spread 得到 113.724 ms 对 120.091 ms；晋升后的 `shipped`
  又在 job 606960 对冻结的 36-layer、10-step oracle 全部逐位一致。

## 哪些值得提取 shared component

本轮没有新增候选满足 generic runtime promotion rule。

| 候选 | Rule A：两个独立 Target 需要相同 invariant | Rule B：runtime 无法表达合法 Target | 决定 |
|---|---|---|---|
| LingBot schema/weight loader | 否 | 否 | 保持 Target-local |
| upstream processor 与 Robotwin transform | 否 | 否 | 保持 adapter-local |
| RoPE frequency cache | 只有 LingBot 实测需要 | 否 | 保持 LingBot route |
| 第四个 ModelRunner stage | 否 | 否；三段已完整表达 | 不新增 |
| source-revision 双 checkout qualification | 尚无两个 Target 的 production 需求 | 合法 route variant 已可表达 | 暂不通用化 |
| onboarding CLI generator | 只有一次真实第三模型样本 | 否 | 暂不实现 |

Identity、Campaign、trace 等 shared components 是 P0–P2 已由 Pi0、Pi0.5 与
LingBot 共同验证的既有 invariant，不是本轮临时抽象。

## 哪些 runtime boundary 真正成为 blocker

没有 generic runtime boundary 阻塞 LingBot bring-up 或首个性能晋升。

真实出现、但不属于 runtime 表达能力的摩擦有：official environment 与项目环境的
依赖隔离；upstream stdout 污染 machine-readable parity stdout；同进程重复构建完整
engine 导致显存生命周期和 A/A' 噪声；以及 qualification 对 source-version 双
checkout 尚不支持。前三项分别由固定 official environment、stdout 分流、同一 A
engine 复用和显式释放解决。最后一项在当前候选中不是 blocker，因为 reference 与
candidate 是同一已检查源码树里的 route；若未来合法候选必须跨源码 revision 加载，
届时才按 Rule B 重新评估。

## 哪些 onboarding 步骤仍需要 Human

Human 仍需声明或批准不可从代码唯一推出的意图：upstream/checkpoint、目标硬件、
deployment workload、precision、correctness requirement、deployment bound 和
optimization budget，并提供受限 checkpoint/集群的访问能力。Robotwin 的具体
batch、views、chunk、steps 与几何可以从冻结配置推导，但推导值和假设必须进入
machine-readable spec，不能静默猜测。

Human 不需要编写 model implementation、Target registration、benchmark、acceptance
或 Campaign 代码；本次 unseen-model workflow 已由 Agent 完成这些步骤。

## 哪些 Agent 工作可以 generator 化

在不解释模型语义的前提下，可以从 onboarding spec 生成低判断成本的 boilerplate：

- model package、Target/benchmark/acceptance registration 的文件骨架；
- compatibility JSON/Markdown 与 stage ledger 的固定字段；
- official/reference/baseline/floor/profile/parity Slurm wrapper 的公共头和 artifact
  路径；
- Campaign create/render/status 命令及最终 acceptance manifest 骨架。

不能安全生成的部分包括 checkpoint tensor 到 runtime buffer 的语义映射、processor
行为、model stage 分解、correctness observable 选择和性能 hypothesis。一次 LingBot
接入不足以稳定区分可生成 boilerplate 与模型特有语义，因此当前仍使用 skill
orchestration 加 Agent 手工生成；积累第二个独立新模型后再按 Rule A 复核。

## 实际摩擦与后续触发条件

| 摩擦 | 本轮处理 | 何时重新讨论 shared abstraction |
|---|---|---|
| official/project Python 依赖不同 | 冻结 official environment，作业显式选择解释器 | 第二个 Target 需要相同多环境编排 |
| parity stdout 被 upstream 日志污染 | upstream stdout 转 stderr，JSON stdout 保持纯净 | 已是通用 CLI hygiene，无需新 runtime 层 |
| A/A' 因重复 engine 构建产生约 14 ms 漂移 | 每个 unique plan 构建一次并复用 A engine | 其他 Target 复现同一生命周期 invariant |
| 长暂停后的环境漂移 | re-anchor 根据完整环境自动分配 measurement segment | 已属于 Campaign 通用 invariant |
| source-version qualification 不支持 | 当前使用同树 route，未绕过 gate | 合法 Target 无法以 route 表达时按 Rule B 提取 |

因此 P7 的动作是保留上述边界和重开条件，而不是立即增加 runtime surface。
