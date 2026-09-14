# Corrections

人判断进入 loop 的地方。技术事实不在这里:一条测错的常数、一个没查过的 ISA 支持、
一处带宽假设,补进 [kernel-wiki](../../.agents/skills/kernel-wiki/SKILL.md) 或硬件轴
的 `measured/` 就能避免重犯,那才是它们的归属。

- [reduce-intervention.md](reduce-intervention.md) — 下一轮的改造清单。run-01 的
  五次人为推动,逐条对应改哪里、拦住哪一次、怎么验证。
- [agent.md](agent.md) — agent 自查出来的纠正。**不自证**:产生错误的那套推理也在
  写关于这个错误的复盘,两者可以同错,所以需要独立复核才收录。

[workflow](../optimization.md) 的"结论复核"是这些的可执行形式;每轮读那六条就够。

## agent.md 的收录门槛

除了"是模式不是一次性 bug"之外,还要三条同时成立:

1. **可证伪的产物,不是叙述。** 一次测量、一个反转前次的提交、一个现在会失败的
   验证器。"回头看这里错了"不予收录。
2. **产物推翻先前结论,而不只是与之不同。** 换条件测出来的数字是另一次测量。
3. **独立确认**:由既没产生该结论、也没写该复盘的读者,在全新上下文里完成,按
   [correction-review](../../.agents/skills/correction-review/SKILL.md)。记下谁查
   的、重跑了什么。复述条目的说法不算确认。

过不了这三条的,写进该 run 的 README 当普通结果。
