# Long40 统一时间与数据契约

_SOP-01 至 SOP-06 的现行生产契约，更新于 2026-07-27_

---

## 📌 权威范围

本文是动态对象轨迹、候选机器人轨迹和 SOP05R 事件产物的统一时间契约。
若现行 SOP、实现规格或里程碑文档与本文冲突，以本文为准。

现行生产链只接受 `40` 个动态对象样本和 `32` 个未来端点。任何其他时间布局
均不得作为生产输入、兼容回退或隐式转换来源。

当前实现和测试必须与本文一致；SOP06 Long40 handoff 尚待实现。

## ⏱️ 统一时间布局

```text
schema_version                    = 4.0.0
motion_snippet_layout_version     = history8_current7_future32_v1
history_steps                     = 8
current_index                     = 7
future_steps                      = 32
sample_dt_s                       = 0.2
history_time_offsets_s            = -1.4, -1.2, ..., 0.0
future_time_offsets_s             = 0.2, 0.4, ..., 6.4
sample_count                      = 40
snippet_duration_s                = 7.8
pose_time_layout_version          = future_endpoints_dt_to_horizon_v1
```

`history[7]` 是决策当前帧。未来数组不重复当前帧：
零基索引 `k` 对应物理时刻 `(k + 1) × 0.2 s`。因此：

- 动态对象完整轨迹为 `[40, ...]`
- 动态对象历史为 `[8, ...]`
- 动态对象未来为 `[32, ...]`
- `LocalTrajectory.poses` 为 `[32, 3]`
- `LocalTrajectory.controls` 为 `[32, 2]`
- 所有逐未来时刻的 footprint、collision 和 query-map 数组首个时间维均为 `32`
- `endpoint_times_s` 为 `[32]`，精确等于 `0.2, 0.4, ..., 6.4 s`

```mermaid
flowchart LR
    accTitle: Long40 Unified Timeline
    accDescr: Eight measured history samples end at the decision frame, followed by all 32 future endpoints through 6.4 seconds

    history[📥 History indices 0 to 6]
    decision[🔍 Current index 7 at 0.0 s]
    future[📊 Future indices 8 to 39]
    horizon[✅ Final endpoint at 6.4 s]

    history --> decision --> future --> horizon

    classDef input fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef current fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef output fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class history input
    class decision current
    class future,horizon output
```

## 🔀 SOP05R 双时间域

SOP05R lightweight TEB 额外保留一个规划器时间域，但不改变模型时间域：

| 属性 | 完整 TEB 路线 | 决策相对模型后缀 |
| --- | --- | --- |
| 类型 | `PlannedTebRoute` | `LocalTrajectory` |
| 时间范围 | 源状态后 `0.2–8.0 s` | 决策后 `0.2–6.4 s` |
| 端点数 | `40` | `32` |
| 坐标系 | 源世界坐标 | 决策局部坐标 |
| 用途 | 可达性、碰撞锚点、到达目标、溯源 | 查询图、碰撞证据和事件发布 |

完整 TEB 路线不是 `40` 样本动态对象 snippet，也不得被序列化为
`LocalTrajectory`。决策后缀必须从完整路线按决策时刻采样，并绑定同一目标和
同一任务代价。

SOP05R 正样本的首次连续碰撞必须满足：

```text
1.2 s <= t_collision - t_decision <= 6.4 s
```

碰撞 authority 必须扫描全部 `32` 个未来区间，包括 `6.2–6.4 s` 的最后区间。
不得截断到前三秒，也不得用离散端点重合代替连续 swept-footprint 判碰。

## 🔗 跨 SOP 约束

| SOP | Long40 约束 |
| --- | --- |
| SOP-01/02 | 记录和切分不改变；输出必须能支持连续 `7.8 s` snippet |
| SOP-03 | 只发布 `40` 样本、`8+32` 布局的真实动态对象 snippet |
| SOP-04 | 候选机器人 `LocalTrajectory` 固定为 `32` 个未来端点 |
| SOP-05/05R | 目标移植、可见性和碰撞检查覆盖完整 `40` 样本；标签覆盖全部 `32` 个未来区间 |
| SOP-06 | 新 handoff 必须共享同一 `32` 步轨迹、时间轴和 observed prefix |

## 🚫 禁止的兼容行为

- 禁止把其他时间布局补零、复制、外推或截断成当前布局
- 禁止在同一 collection 中混用不同时间布局
- 禁止根据数组 shape 猜测版本
- 禁止现行生产 loader 接受非 Schema `4.0.0` 的时间布局
- 禁止将 `8.0 s` TEB 完整路线误作模型未来

遇到非当前布局时，现行生产命令必须显式失败并报告版本/布局不匹配；活跃树
不提供兼容 loader 或隐式转换。

## ✅ 文档一致性检查

现行文档必须同时满足：

1. 未来端点数写为 `32`
2. 未来窗口写为 `0.2–6.4 s`
3. 完整 snippet 写为 `40` 样本和 `7.8 s`
4. 当前索引写为 `7`
5. 动态对象 Schema 写为 `4.0.0`
6. SOP05R 完整 TEB 路线单独写为 `40` 端点、最长 `8.0 s`
7. 不以文档更新冒充代码、测试或产物迁移完成
