# Agent Instructions

This repository keeps its collaboration rules in [`.claude/rules/`](.claude/rules/).
Read the rules relevant to the files you will change before editing. In
particular, all non-trivial work follows the Agent Note and PR workflow in
[`agent-notes-and-pr-workflow.md`](.claude/rules/agent-notes-and-pr-workflow.md).

Documentation records durable specifications, contracts, and evidence; it does
not restate or narrate implementation. Durable design decisions live in
[`Agent Notes`](.agents/notes/README.md), not in source-oriented documentation or PR prose.

此外有如下要求：
1.谋定而后动，通过建立低成本的假设筛选和验证机制，降低每次优化迭代成本
2.实验求新知，在迭代和对比实验中严格控制变量和噪声尺度，保证每次尝试带来信息增益
3.谨慎下结论，在归因性能变化前主动排除其他混杂因素，防止测量误差导致错误的优化结论
