# 当前项目文档入口

_事件中心盲区风险与主动验证项目的当前权威文档索引，更新于 2026-07-15_

---

## 📋 权威文档

后续 Agent 只应从以下三份文档开始工作：

1. [`event_centered_blind_spot_implementation_spec.md`](./event_centered_blind_spot_implementation_spec.md)  
   定义研究目标、方法语义、数据契约、风险与验证价值公式、评价指标及论文主张边界。
2. [`parallel_acceleration_implementation_plan.md`](./parallel_acceleration_implementation_plan.md)  
   定义 W0–W8 工作流、并行波次、最低/理想验收线、Go/No-Go 和降级策略。
3. [`event_centered_blind_spot_agent_sops.md`](./event_centered_blind_spot_agent_sops.md)  
   将前两份文档拆分为 SOP-00～SOP-16，可直接派发给实现 Agent。

权威顺序为：方法规格 → 并行计划 → Agent SOP。出现科学语义冲突时以方法规格为准；出现任务组织冲突时以 Agent SOP 中明确记录的冻结决策为准。

## 🔄 推荐工作方式

1. 总控 Agent 从 `event_centered_blind_spot_agent_sops.md` 选择一个依赖已满足的 SOP
2. 只把该 SOP 的允许文件、步骤和验收标准派发给执行 Agent
3. 执行 Agent 按 handoff 模板返回测试、产物、指标和限制
4. 通过对应 G0–G5 门禁后再启动下游任务

## 📦 历史文档

2026-07-14 及更早的设计、执行计划和交接记录已归档到 [`../documents_old/`](../documents_old/README.md)。这些文件用于追溯既有实验和决策，不再作为新实现的默认入口。

