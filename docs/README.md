# 当前项目文档入口

_Schema 4 Long40 与 SOP08 occupancy baseline 的现行文档索引，更新于 2026-07-27_

## 权威文档

1. [Long40 统一时间与数据契约](./long40_system_contract.md)：统一时间轴、数组形状、版本和跨 SOP 约束。
2. [M5 可见性引导放置设计](./superpowers/specs/2026-07-26-sop05r-m5-constructive-visibility-placement-design.md)：当前可见性引导放置设计。

## 当前实现边界

现行可执行入口覆盖记录切分、THOR 索引、Long40 snippet、base-state 提取、SOP05R lightweight-TEB 事件生成，以及消费已认证 Long40 risk shard/seal/sidecar 的 SOP08 occupancy baseline。SOP06/SOP07 生产器仍需按 Long40 重建，不得恢复已退役的短时域实现；SOP09 至 SOP13 仍退出活跃树。

实现、配置和测试必须以 Schema 4、8 帧历史、32 个未来端点和 6.4 秒未来窗口为准。已退役契约不得作为兼容回退或 fixture 重新引入。
