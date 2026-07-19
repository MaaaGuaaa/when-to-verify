# 事件中心盲区样本生成、轨迹风险学习与主动验证：Codex 工程实现规格

> 项目暂定名：**Learning When to Verify: Decision-Calibrated Hidden Risk for Robot Local Planning**  
> 中文暂定名：**学习何时验证：面向机器人局部规划的决策校准隐藏风险**  
> 文档目的：把当前研究思路整理成可由 Codex/工程 Agent 直接拆解和实现的技术规格。  
> 推荐第一版范围：**2D BEV、差速轮式机器人、通用动态对象、程序化遮挡、短时局部规划**。

---

## 0. 一页式目标说明

### 0.1 要解决的问题

机器人在室内、仓储、园区、校园等环境中执行局部路径规划时，会遇到当前不可观测区域：

- 结构性盲区：后方/侧后方传感器死角、有限视场、自身结构遮挡；
- 环境遮挡：墙角、门口、货架、柱子、大型设备；
- 历史观测失效：过去看过，但当前不可见且信息已经陈旧。

我们不预测完整世界，也不直接从图像/点云做感知。系统从局部 BEV 信念出发，回答两个学习问题：

1. **轨迹风险学习**

\[
f_\theta(z_t,\xi)\rightarrow
\{Q_{0.5},Q_{0.8},Q_{0.9},Q_{0.95},p_{\mathrm{coll}}\}
\]

其中：

- \(z_t\)：机器人当前可获得的局部历史 BEV 信念；
- \(\xi\)：候选局部轨迹；
- 输出：候选轨迹穿过不可观测区域时的风险分布与碰撞概率。

2. **验证动作价值学习**

\[
g_\psi(z_t,\xi,v)\rightarrow G
\]

其中：

- \(v\)：短时验证动作，如原地偏航、短距离前探、侧探、停下扫描；
- \(G\)：做该验证动作后，后续执行/拒绝/重规划的经验决策损失下降量，已经扣除验证动作成本。

### 0.2 在线输出

系统不直接输出连续低层控制，而输出局部规划层决策：

```text
execute(ξ*)     # 执行候选轨迹前 0.2~0.5 s，随后滚动重规划
verify(v*)      # 执行 0.5~1.0 s 短时验证动作，更新 BEV 后重规划
reject          # 停止一轮、放弃当前候选或请求重新规划
```

### 0.3 核心数据策略

THÖR-MAGNI 等真实数据中，天然“轨迹进入盲区且与隐藏动态对象冲突”的事件很少。因此不依赖自然事件，而采用：

> **事件中心的半合成样本生成（Event-Centered Semi-Synthetic Generation）**

具体做法：

1. 从真实数据提取机器人状态和所有有效非机器人动态对象轨迹片段；
2. 从当前机器人状态程序化采样候选局部轨迹；
3. 在候选轨迹上选择潜在冲突点和冲突时刻；
4. 在冲突点附近放置程序化遮挡物，或施加结构性盲区；
5. 对真实 23 点 / 4.4 s 动态对象轨迹片段只做刚体变换；正式 schema 3
   SOP05 固定 `time_scale=1.0`，不为对齐事件而外推或补写轨迹；
6. 以 collision 母事件为可独立发布单元，再独立尝试近碰、时序安全、空间
   安全、无关隐藏对象和空盲区变体；未成功的变体记录稳定缺失原因，不会作废
   已合法的母事件或其他变体；
7. 用完整 oracle 世界自动计算风险 GT 和验证价值 GT。

---

## 1. 明确范围与非目标

### 1.1 第一版必须实现

- 2D/2.5D BEV 表示；
- 差速轮式机器人运动模型；
- 按 `human`、`carried_object`、`unknown_dynamic` 分型的真实动态对象轨迹片段库；
- 程序化矩形墙体/货架/柱子遮挡；
- 结构性 FOV 盲区；
- 局部轨迹 rollout；
- 风险 GT 自动生成；
- 轨迹风险网络；
- 反事实验证动作和净决策价值 GT；
- 验证价值网络；
- execute / verify / reject 离线或轻量闭环评估。

### 1.2 第一版明确不做

- 不从 RGB 图像或原始点云端到端学习；
- 不处理红绿灯、车道、人行道通行规则等自动驾驶语义；
- 不把 3D semantic occupancy prediction 作为主贡献；
- 不做端到端强化学习控制；
- 不声称 scenario bank 产生严格 Bayes ground truth；
- 不声称无条件安全保证；
- 不要求真实数据中天然存在大量遮挡碰撞事件；
- 不要求数据集中真实机器人未来轨迹作为唯一候选轨迹。

---

## 2. 数据源角色分工

### 2.1 主数据：THÖR-MAGNI

用途：

- 提取真实机器人历史状态；
- 提取所有有效非机器人 BODY 的轨迹、速度、加速度和 marker 几何分布；
- 建立按对象类型隔离的 trajectory snippet library；
- 提供真实室内/人机交互运动统计；
- 作为事件中心半合成场景的基础状态来源。

不依赖其天然提供：

- 遮挡冲突事件；
- 多条候选局部轨迹；
- 验证动作；
- 风险/验证价值标签。

参考：<https://arxiv.org/abs/2403.09285>

### 2.2 备选/增强：THUD++

用途：

- 提供额外动态室内数据；
- 提供行人轨迹与 2D/3D boxes；
- 若 Unity 平台可运行，用于反事实验证观测和闭环测试。

参考：<https://arxiv.org/abs/2412.08096>

### 2.3 泛化测试：JRDB

用途：

- 机器人第一视角动态人群；
- 完整 360°/3D 标注作为 oracle；
- 人为遮掉部分视场，测试模型跨数据集泛化。

参考：<https://arxiv.org/abs/1910.11792>

### 2.4 可选轨迹先验：SCAND

用途：

- 统计真实机器人线速度/角速度/遥操作动作分布；
- 约束候选轨迹 primitive 的范围；
- 不作为主要风险 GT 来源。

参考：<https://arxiv.org/abs/2203.15041>

### 2.5 闭环与反事实：Arena-Rosnav / Arena 4.0

用途：

- 程序化生成动态人群、墙角、货架、门口、结构性 FOV 盲区；
- 执行短时验证动作后真实更新观测；
- 闭环评估 execute / verify / reject。

参考：<https://arxiv.org/abs/2409.12471>

### 2.6 第一版推荐组合

```text
风险训练数据：THÖR-MAGNI 半合成事件
验证价值训练数据：同一个 2D oracle 生成器 + 部分 Arena 仿真
验证/测试：held-out THÖR 半合成 + Arena closed-loop
泛化补充：JRDB
```

---

## 3. 数据泄漏防护：先切分，再生成任何样本

这是实现中的硬约束。

### 3.1 禁止做法

禁止先从所有 recording 生成几百万 trajectory/event samples，再随机按 sample 切 train/test。相邻帧、相同参与者和相同轨迹片段会严重泄漏。

### 3.2 正确流程

```text
原始 recordings / sessions / participants
        ↓
group-level split
        ↓
train groups / calibration groups / val groups / test groups
        ↓
在每个 split 内独立：
    建 trajectory snippet library
    提取 base states
    生成候选轨迹
    生成遮挡事件
    生成风险 GT
    生成验证价值 GT
```

### 3.3 推荐切分键

优先顺序：

```text
recording_id
session_id
participant_id
scene_id
simulation_seed
```

建议比例：

```text
train       70%
calibration 10%
validation  10%
test        10%
```

上述是通用严格策略。THÖR-MAGNI 第一版采用显式的 recording 泛化例外：保留冻结的
`37/5/5/5` recording assignment，`recording_id` 跨 split 交集必须为 0；由官方
`FILE_ID` 得到的 recording-day `session_id` 可以跨 split，但必须完整填充并在 audit
中逐项列出；稳定 participant identity 不可获得，不得用重复 helmet label 伪造。
正式评测范围固定为 `unseen_recording_within_known_sessions`，不得宣称 unseen-day、
unseen-session 或 unseen-participant 泛化。通用数据集仍默认要求可用的
recording/session/participant 全部隔离。

### 3.4 轨迹片段库也必须分 split

训练集中的动态对象 snippet 不得出现在 test 生成器中。每个 split 和对象类型单独建立：

```text
snippets/train/<object_type>/
snippets/calibration/<object_type>/
snippets/val/<object_type>/
snippets/test/<object_type>/
```

---

## 4. 统一坐标系和基础数据结构

### 4.1 坐标系

对每个 base state，以当前机器人位姿为局部坐标原点：

```text
x 轴：机器人当前朝向前方
y 轴：机器人当前左侧
原点：机器人当前几何中心
单位：米
角度：弧度
```

全局点 \(p\) 转局部：

\[
p_{local}=R(-\theta_t)(p_{global}-p_t^{robot})
\]

### 4.2 时间统一

推荐：

```text
history_dt       = 0.2 s
history_steps K  = 8
history_span     = 1.4 s  # 8 个含当前点的采样：-1.4 ... 0.0 s
future_horizon   = 3.0 s
future_dt        = 0.2 s
future_steps T   = 15
```

未来数组统一采用 `future_endpoints_dt_to_horizon_v1`：当前位姿 `q0` 只作为积分种子，
不写入未来数组；`poses[0:15]=q1...q15`，零基索引 `k` 的物理时刻严格为
`(k+1)*future_dt`，即 `0.2...3.0 s`。旧 `q0...q14 / 0.0...2.8 s` 布局属于
schema 2，不得由 schema 3 loader 静默解释或改标签。

原始轨迹统一用线性插值/角度 unwrap 后插值到上述时间网格。

### 4.3 BEV 网格

推荐：

```text
range_x = [-8m, 8m]
range_y = [-8m, 8m]
resolution = 0.1m
H = W = 160
```

### 4.4 机器人和动态对象几何

第一版简化：

```text
robot footprint：矩形或圆形
    width  = 0.55m
    length = 0.70m
    safety inflation = 0.15m

human footprint：圆形
    ordinary radius = 0.30m
    THÖR Carrier role radius = 0.45m

carried_object / unknown_dynamic footprint：
    优先取 QTM marker 局部坐标 x/y extent 的 95 分位矩形
    至少 20 个有效 marker 帧，单边范围 [0.05m, 3.0m]
    marker 不足时使用按类型配置的 fallback
```

所有有效非机器人 BODY 都进入适配器，不得因名称为 storage、cart、carrier、LO、bin、box、bucket 或暂时静止而排除。类型只允许 `human`、`carried_object`、`unknown_dynamic`；原始名称和 role 保存在 provenance，不进入冻结 footprint spec。

---

## 5. 原始数据预处理

### 5.1 预处理输出

每个 recording 产生：

```python
RecordingIndex = {
    "recording_id": str,
    "timestamps": np.ndarray[N],
    "robot_pose": np.ndarray[N, 3],      # x, y, yaw
    "robot_twist": np.ndarray[N, 2],     # v, omega
    "dynamic_objects": {
        object_id: {                    # recording_id::body_name
            "object_type": str,
            "raw_body_name": str,
            "raw_role": str | None,
            "timestamps": np.ndarray[M],
            "poses": np.ndarray[M, 3], # x, y, yaw
            "velocities": np.ndarray[M, 2],
            "footprint": dict,
        }
    },
    "static_map": Optional[np.ndarray],
}
```

### 5.2 base state 提取

从每个 recording 每隔 0.5~1.0 秒取一个 base state：

```python
BaseState = {
    "state_id": str,
    "split": str,
    "recording_id": str,
    "dynamic_object_ids": tuple[str, ...],
    "timestamp": float,
    "robot_history": np.ndarray[K, 3],
    "robot_state": np.ndarray[D],
    "visible_dynamic_object_history": dict[str, np.ndarray[K, 3]],
    "visible_dynamic_object_specs": dict[str, dict],
    "static_map_local": np.ndarray[H, W] | None,
    "metadata": dict,
}

OracleContext = {
    "base_state_id": str,
    "dynamic_object_history": dict[str, np.ndarray[K, 3]],
    "dynamic_object_future": dict[str, np.ndarray[T, 3]],
    "dynamic_object_specs": dict[str, dict],
    "metadata": dict,
}
```

observed `BaseState` 与完整 `OracleContext` 必须写入不同对象/文件。对象 ID 排序且唯一；history/future/spec 的 key 必须严格对齐。`BaseState` 不得携带 future。全局 schema 1/2 产物不做兼容读取，升级到 schema 3 后统一重建。

筛选条件：

- 历史和未来窗口完整；
- 机器人当前状态有效；
- 机器人不在静态障碍内部；
- 局部范围内允许插入程序化遮挡物；
- 初期允许 base state 中没有自然动态体，因为后续会插入移植动态对象；但已有对象不得因暂时静止或非人语义被删除。

---

## 6. 真实动态对象轨迹片段库

### 6.1 目标

避免用纯随机直线动态体。真实对象轨迹片段保留速度变化、轻微弯曲和加速度统计；
当前正式 SOP05 只做 SE(2) 变换并固定 `time_scale=1.0`。小幅时间缩放只保留为
未来有独立版本的研究扩展，不属于当前发布契约。片段按 split 和 `object_type` 分库。

### 6.2 snippet 提取

从每个动态对象轨迹中滑窗提取：

```text
motion_snippet_layout_version = history8_current7_future15_v1
snippet_duration = 4.4 s
snippet_dt = 0.2 s
sample_count = 23
history_steps = 8
current_index = 7
future_steps = 15
human speed = 0.30~2.00 m/s
carried_object / unknown_dynamic speed = 0.05~2.00 m/s
max_acceleration = 2.5 m/s²（各类型可配置）
```

归一化为局部坐标：

```text
snippet 第一个位置为 (0, 0)
初始运动方向对齐 +x
```

结构：

```python
MotionSnippet = {
    "snippet_id": str,
    "source_recording_id": str,
    "source_session_id": str,
    "source_object_id": str,
    "object_type": str,
    "footprint": dict,
    "positions": np.ndarray[23, 2],       # float32
    "velocities": np.ndarray[23, 2],      # float32
    "headings": np.ndarray[23],           # float32 yaw
    "duration_s": 4.4,
    "mean_speed": float,
    "max_acc": float,
}
```

索引 `0:8` 是真实历史（`-1.4 ... 0.0 s`），索引 7 是当前，索引 `8:23` 是真实未来
（`0.2 ... 3.0 s`）。数组必须全部 finite。library NPZ metadata、summary 和 source
manifest 必须同时记录上述完整布局与新 `split_manifest_digest`。生产 loader 对缺少
布局字段、16 点、3.0 s 或 digest 不一致的旧产物明确失败；禁止重复 current pose、
反向外推、理想轨迹补历史或跨 gap 插值。4.4 s 连续窗口造成的短 segment 必须进入
`insufficient_contiguous_duration` 拒绝统计。

### 6.3 允许变换

\[
\tilde h_\tau=R(\phi)h_{a\tau}+t
\]

范围：

```text
rotation φ：任意
translation t：由冲突点决定
time scale a：片段库研究扩展可为 0.8~1.2；当前正式 SOP05 固定为 1.0
额外位置噪声：最多 0.05m，默认不用
```

### 6.4 变换后有效性检查

- 不穿过静态障碍或新插入遮挡物；
- 速度符合对象类型配置范围；正式路径不通过时间重采样改变真实片段；
- 加速度不超过阈值；
- 历史窗口内满足不可见约束；
- 未来窗口内至少部分进入局部地图；
- 不与机器人当前 footprint 重叠。

---

## 7. 候选局部轨迹生成

### 7.1 第一版：恒定速度-角速度 rollout

差速运动学：

\[
x_{k+1}=x_k+v\cos\theta_k\Delta t
\]

\[
y_{k+1}=y_k+v\sin\theta_k\Delta t
\]

\[
\theta_{k+1}=\theta_k+\omega\Delta t
\]

默认采样：

```yaml
linear_velocities:  [0.20, 0.40, 0.60, 0.80]
angular_velocities: [-0.80, -0.40, 0.00, 0.40, 0.80]
reverse_velocities: [-0.20, -0.40]   # 仅部分样本/结构盲区 stress test
horizon: 3.0
step_dt: 0.2
```

基础候选约 20 条，按概率加入 6 条倒车候选。

冻结的积分与序列化语义为：`q0=(0,0,0)` 只作为当前时刻的积分种子，不进入
`LocalTrajectory.poses`；15 个输出依次为 `q1…q15`，时间偏移严格是
`0.2,0.4,…,3.0 s`。因此零基 future index `k` 对应
`tau=(k+1)*0.2 s`，不得将 `poses[0]` 解释为当前时刻。

正式共享库必须声明
`trajectory_bank_version=sop04_audited_bank_v2` 与
`pose_time_layout_version=future_endpoints_dt_to_horizon_v1`，并由 v2 audit 证明
future-endpoint kinematics、shape/dtype/finite、query-map 以及 serial/parallel
determinism。SOP05 只接受三个 core payload 的精确 checksum envelope，并要求调用方从
产物目录外传入可信 `external_handoff_digest`；旧 v1、缺少时间字段或
`0.0…2.8 s` 布局一律拒绝，不做双版本兼容。

### 7.2 轨迹过滤

丢弃：

- 机器人 footprint 与基础静态地图相交；
- 超出 BEV 范围；
- 角速度/加速度超限；
- 长时间原地不动且不是 stop primitive；
- 被新插入遮挡物直接阻塞；
- 和局部目标方向完全无关的极端轨迹（若有局部目标）。

### 7.3 轨迹 query maps

对每条 \(\xi\) 生成：

```text
swept_volume_mask: [H, W]
time_to_arrival_map: [H, W], 未经过为 -1
braking_margin_map: [H, W]
centerline_map: [H, W]
```

扫掠体：

\[
S(\xi)=\bigcup_{\tau=1}^{T}B_r(q_\tau)
\]

---

## 8. 事件中心半合成生成器

### 8.1 为什么必须事件中心

随机放墙、随机放动态对象、随机采样轨迹会导致有效冲突事件极低。正确做法是：

> 先确定轨迹上的潜在冲突时刻和位置，再反向构造遮挡物和隐藏目标动态对象。

### 8.2 总流程

```text
BaseState z
    ↓
读取 schema 3 候选轨迹 ξ，缓存 canonical future sweep；history/seam 按 pair 构建
    ↓
在自由空间提议与 ξ 有因果关系、但不阻塞扫掠体的静态遮挡物
    ↓
用与正式 renderer 相同的栅格、ray casting 和遮挡几何得到当前 blind-region mask
    ↓
枚举对齐的冲突端点 τ*=(k+1)future_dt 和真实 23 点 MotionSnippet
    ↓
从 snippet 当前点到 p* 构造 SE(2) reachability arc，先查询盲区内可放置的
footprint-center mask
    ↓
用 chord fast path 做保守筛选；未能证明安全的候选进入有上限的 exact fallback
    ↓
仅做该候选已冻结的 SE(2) 移植，重做真实 footprint 静态/连续碰撞、
当前隐藏、未来连续出现和同索引 collision 校验
    ↓
collision 母事件独立发布；各 negative 变体独立生成与保留
    ↓
历史可见内容进入 RiskSample 输入，oracle future 只进入风险标签
```

正式生产算法标识为 `blind_reachability_quota_first_v3`（producer
`sop05_generation_run_v6`）。它沿用 v2 的全部几何权威，但将搜索次序改为由 pair
seed 稳定打散的 conflict/snippet/side/angle 序列；每个遮挡提案最多检查配置给出的
snippet 窗口，凑齐本 pair 请求的 exact-valid 母事件后立即停止。该升级只改变有限
搜索/完成策略，不放宽任何物理验收。“v5/v6”是 SOP05 流程代际，不是全局 schema
版本；所有产物仍必须标记
`schema_version=3.0.0`。
正式 loader 必须显式拒绝紧邻旧三元组
`blind_reachability_first_v2` / `sop05_generation_run_v5` /
`sop05_pair_generation_report_v3`，不得在加载时补算或升级解释。run manifest、summary
与 completion marker 的 v3 格式因字段 schema 未改变而刻意保留。

v3 沿用并冻结 `reachability_candidate_se2_v2`：`ReachabilityIdentity` / candidate
显式新增 `source_session_id`，并通过 `source_snippet_id` 在可信 snippet library
中间接定位 recording。transform payload 以及 generated event/world identity 则必须
显式同时绑定 `source_recording_id` 与 `source_session_id`。
任一字段缺失、不一致，或产物仍声明 `blind_reachability_first_v1` /
`reachability_candidate_se2_v1`，producer 和 loader 都必须 fail closed；禁止在加载时
补字段或升级解释旧 identity。

### 8.3 冲突点选择

从轨迹中选择：

```text
τ* ∈ [1.0s, 2.2s]
```

若冲突点由零基 future index `k` 选取，则
`τ*=(k+1)future_dt`；禁止使用 `k*future_dt` 造成一帧错位。

对应：

\[
p^*=q(\tau^*)
\]

计算轨迹切向 \(\mathbf t^*\) 和法向：

\[
\mathbf n^*=(-t_y^*,t_x^*)
\]

优先选择：

- 距离当前机器人不太近；
- 周边有足够空间放遮挡物；
- 不在已有墙体内部；
- 轨迹局部曲率不过大。

### 8.4 环境遮挡物放置

当前 schema 3 正式 SOP05 只发布 environment collision 母事件，算法固定为
`blind_reachability_quota_first_v3`。它不先生成目标曲线再反复尝试用单一正前方障碍遮住，
而是对候选轨迹的交互范围、bearing bin、遮挡物类型/尺寸/姿态构建稳定的
`causal_free_space_schedule_v1` 候选序列。每个遮挡物必须先通过基础静态图、当前
上下文占据、机器人完整 history+future 扫掠体与上下文对象扫掠体的净空检查。
运行前按 `trajectory_id` 稳定排序并对本次 schedule 中每条唯一候选轨迹只预热
一次 canonical robot future sweep；单进程直接复用，多进程通过 `fork` 继承已预热的
`robot_sweep_cache_v1` 内存页，以 copy-on-write 方式共享只读几何。base-specific
history 与 current-to-future seam 仍须按 pair 构建一次；不得在同一 pair 的每个遮挡
提议内重复构建这些扫掠体。

当前几何版本栈为：

```text
causal occluder schedule       = causal_occluder_schedule_v1
causal occluder proposal       = causal_occluder_proposal_v2
blind region                   = blind_region_causal_delta_v2
footprint center mask          = footprint_center_mask_causal_delta_v2
visibility                     = raycast_visibility_environment_v1
collision sweep preparation    = occluder_collision_sweep_preparation_v2
robot sweep cache              = robot_sweep_cache_v1
robot collision sweep bundle   = robot_collision_sweep_bundle_v1
```

`causal_occluder_proposal_v2` 将正式 interaction-region 摘要绑入 proposal identity；
blind region 只能是同一 renderer 下无遮挡基线与加入该遮挡后的 causal
visibility delta，并必须与 interaction region 精确取交。扩大、缩小或伪造
该区域均必须拒绝。`occluder_collision_sweep_preparation_v2` 下的连续碰撞
认证对 circle footprint 忽略无物理意义的 yaw-only 旋转加密，对 rectangle
仍保留姿态加密；旧 preparation 产物不得混用。

对每个通过的 causal occluder，生成器用与 SOP06 renderer 一致的栅格和 ray-casting
几何生成 current visibility 与 blind-region mask，再根据目标 footprint 及其 yaw 算出
“footprint 中心可以落入”的盲区 mask。真实 snippet 的 current 到所选冲突点形成
reachability arc；只有该 arc 查询到上述 mask 的候选才进入后续。弦线弯曲不超过
`chord_deviation_fastpath_m` 且弦线能被证明安全时走 fast path；其余候选进入每个
anchor 有上限的 exact fallback，超限后以稳定原因拒绝。

为避免 `64 × 全 split snippets × 7 × 30` 的无界穷举，snippet 先按 pair seed
稳定排序；第 `proposal_index` 个遮挡提案读取一个大小为
`snippet_candidates_per_proposal` 的循环窗口。conflict index、crossing side 与 angle
也按同一 pair identity 稳定排序。达到 `requested_event_count` 个 exact-valid 母事件
后只结束尚未发生的搜索；已执行阶段的计数、拒绝原因和 ID 证据仍完整保存。
相同 seed/input 必须给出相同搜索前缀和事件 identity，不允许依赖 worker 完成顺序。

运行时允许结果等价的批量几何与保守 broadphase：矩形 footprint 对闭网格 cell 的
相交可使用向量化 SAT，但数值边界必须回退既有 scalar authority；连续扫掠只可在
两者外接圆扩张 swept AABB 严格分离时快速判 clear，其余仍进入原递归权威。并行
pair 调度必须同时约束 running futures 与已完成待按 rank 输出的 reports，worker
完成顺序不得改变事件、stage 计数或选择结果。

正式移植只执行 reachability candidate 已冻结的 SE(2) 变换，`time_scale=1.0`；完整
23 点源轨迹、逐帧 yaw 与非初始点弯曲必须保留。候选随后使用真实
circle/rectangle footprint 做静态几何、动力学、当前隐藏、未来连续出现、
遮挡物与目标轨迹净空、以及目标与机器人在选定 future index 上的真实 collision
检查。传感器当前位姿只用于 LOS/FOV，不得隐式前插到碰撞 sweep。动态 sweep
仍需按不超过半个 BEV cell 的平移步长和 `5°` yaw 步长加密，并用 signed clearance
与保守运动上界认证帧间连续安全；无法证明正净空时保守拒绝。

每次运行必须保存可对账的候选阶段计数：

```text
obstacle_proposal_count
  = obstacle_proposal_rejected_count + obstacle_proposal_passed_count
transform_candidate_count
  = transform_rejected_count + chord_certified_count + chord_unresolved_count
exact_validation_count
  = exact_validation_rejected_count + exact_validation_accepted_count
```

这些是算法成本与失败位置的正式报告，不得只报最终成功率。历史
`joint_occluder_first_v4` 及其候选前缀仅用于读取旧实验的迁移说明；它不是当前
producer 版本，正式 schema 3 producer/loader 必须拒绝该标识，不得在加载时升级解释。

规模运行不得假设每个 pair 必然成功。2026-07-19 的 train smoke 中 100 个真实 pair
产生 95 个 exact-valid 母事件并稳定选出 90 个，生产申请应至少预留约 5–10% 的 pair
余量，同时保留失败 pair 的完整拒绝统计。

旧 v4 设计曾将遮挡物中心写成“冲突点加固定法向偏移”并配置
`normal_offset_range`。该块已废弃，不属于 v3 合法参数；当前位姿只能来自
`causal_free_space_schedule_v1` 的 interaction-range / bearing-bin / type-size-yaw
枚举与后续 causal-delta 验证。禁止根据法向偏移公式在 v3 中恢复
单一“正前方障碍”模式。

硬约束：

1. 遮挡物不得与机器人候选轨迹扫掠体相交；
2. 遮挡物不得与基础静态障碍大面积重叠；
3. 当前机器人到目标动态对象历史位置的视线需被遮挡物截断；
4. 目标对象未来从遮挡边界出现时应在物理上连续；
5. 不允许任何动态对象的真实 footprint 轨迹穿墙。

### 8.5 结构性盲区生成

本节保留为事件多样性扩展与诊断定义；当前
`blind_reachability_quota_first_v3` 正式 producer 的 `production_event_kind` 固定为
`environment`，不得为凑类型比例而伪称已产出 structural 事件。

不放实体障碍，直接定义传感器可见范围：

```yaml
structural_fov:
  forward_fov_deg: [160, 180, 220]
  range_m: [6.0, 8.0, 10.0]
  optional_blind_sectors:
    - center_deg: 150
      width_deg: 60
```

目标动态对象必须在当前及部分历史时刻落入 FOV 外，未来与候选轨迹发生交互。

### 8.6 混合盲区

同时存在 FOV 限制和实体遮挡。它与 structural 一样是当前 environment-only
正式 producer 之外的扩展/诊断类型。历史设计曾使用以下 proposal distribution：

```text
environment occlusion 60%
structural blind spot 30%
mixed 10%
```

上述 `60/30/10` 只能作为历史诊断参考，不是 schema 3 正式配置字段、
publication quota 或训练门槛。同样，任何 `6/3/1` 的 environment/structural/mixed
组成都不能决定发布资格。发布只看请求的总样本数及每个事件是否满足
collision、当前遮挡、未来出现和物理有效性契约；类型计数只作诊断。

每个 `complete` 产物必须由 producer CLI 额外交付一个版本化的
`publication_semantic_digest`。该值绑定 run manifest、checksum manifest、target-motion
manifest 以及完整 target-motion/`OracleWorld` 语义摘要；SOP06 加载时必须从目录外的
运行记录或 handoff 显式传入原值，不得从待验目录自身读取“期望值”。loader 同时重算
generated-event/world identity。这样即使整套目录及其内部 checksum/marker 被一致重封，
也不能在保留原 handoff digest 的条件下伪造另一组事件、世界内容或统计结果。

producer 在 staging 内写完 completion marker 后，必须以该目录外 handoff digest 调用
正式 SOP05 consumer loader，完成一次完整语义 round-trip，之后才允许原子 rename 到
最终输出目录。round-trip 的 schema、identity、selection、checksum、shard 或数量校验
任一失败都必须清理 staging，不能发布一个 producer 自称 complete、但正式 consumer
拒绝的产物。

---

## 9. 将真实 typed 动态对象轨迹移植到冲突事件

主论文事件分布默认只从 `human` snippet library 采样目标，以保持原始隐藏行人研究
口径。`carried_object` 和 `unknown_dynamic` 始终保留在场景中并参与观测、占据和
碰撞；将它们作为目标时必须通过显式 typed-sampler 配置启用，并单独报告结果。

`generator_train.yaml` 和 `generator_test.yaml` 必须提供完整 target-type policy：

```yaml
target_type_policy:
  whitelist: [human]
  weights:
    human: 1.0
    carried_object: 0.0
    unknown_dynamic: 0.0
```

解析时要求三类 key 齐全、权重有限且非负；白名单外权重归零，白名单内至少一项为
正，再按冻结类型顺序归一化。对解析后的 policy 做规范化序列化和稳定摘要，所得
`target_type_policy_digest` 必须写入事件 manifest，并由所有下游数据、checkpoint、
calibration 和评测产物逐级传播、严格校验。下游只使用 snippet/contract 冻结的
`object_type` 和 footprint spec，不得根据原始 body name、role 或文件名重新分类或
重估 footprint。

`reachability_candidate_se2_v2` 的源身份不能只用 recording 代替 session。
每个 transformed target 的 provenance 必须同时携带非空
`source_recording_id` 和 `source_session_id`，且与被选 `MotionSnippet` 逐字段
一致。这两个字段必须显式绑入 generated-event ID 和 world ID；
`EventTargetMotionRecord` 不新增这两个字段，而是通过它已绑定的
generated-event/world IDs 完成 join。SOP05 publication semantic digest 再绑定整个
已验证产物。任一显式 provenance 或 ID join 缺失/错配都是契约错误，
不得回退到 v1 identity。

### 9.1 对齐目标

SOP-05 读取 v2 snippet 时固定使用：

```text
history = transformed_poses[0:8]
current = transformed_poses[7]
future = transformed_poses[8:23]
source_anchor_time = 1.4 + conflict_time_s
```

`conflict_time_s` 从当前时刻计，而 snippet source time 从索引 0 计；直接令二者相等
会把所有事件提前 1.4 s，禁止继续使用。SOP-03 只做首点/初始运动方向归一化，下面的
事件级 SE(2) 移植仍由 SOP-05 完成。

选择 snippet 中某个索引 \(k^*\)，要求变换后：

\[
\tilde h_{k^*}\approx p^*
\]

并让该时刻对应：

\[
t^{source}_{k^*}-1.4\approx \tau^*
\]

### 9.2 横穿方向

初始目标方向通常接近轨迹法向：

\[
\angle(\dot h_{k^*},\mathbf n^*)<\theta_{max}
\]

默认 \(\theta_{max}=35^\circ\)。

也可以生成斜穿/同向交汇事件，但第一版以侧向横穿为主。

### 9.3 求变换

1. 以 snippet 真实 current `positions[7]` 与 source anchor 的差构造 reachability
   candidate；当前正式路径固定 `time_scale=1.0`；
2. 将 source anchor 平移到原点；
3. 将 anchor 速度旋转到候选横穿方向；
4. 平移到 \(p^*\)，并对完整 23 点位置/速度/yaw 应用同一 SE(2)；
5. 不再重采样该变换，对完整历史/未来轨迹做 exact 几何与可见性检查。

### 9.4 失败候选和续跑

以下任一失败时，当前 causal-occluder/reachability/exact 候选必须以稳定
reason 记录，然后继续冻结调度中的后续候选；不得改写已失败候选的身份或
用应当失败的候选凑数：

- 当前时刻目标对象可见；
- 过去历史中目标对象已被看到但不符合目标样本类型；
- 对象真实 circle/rectangle footprint 穿墙；
- 对象与机器人当前 footprint 重叠；
- 未来轨迹超出范围过多；
- 速度/加速度超限；
- 母事件无法在选定 future endpoint 形成 collision（near-miss/safe 由 SOP06 独立尝试）。

请求级 `accepted_count/requested_event_count` 只表示最终产出率。还必须分别报告
obstacle proposal、transform/chord 和 exact validation 三层的数量、守恒关系和拒绝
原因，不得用任一层的接受率替代其他层。当前 formal producer 只对
environment 事件进行正式统计；structural/mixed 扩展必须使用独立版本和产物目录。

---

## 10. 配对反事实样本生成

对同一 BaseState、候选轨迹、遮挡物和目标 `MotionSnippet`，生成多个只改变关键
变量的版本。所有带目标的变体固定目标 object type、footprint spec、
source object ID、`source_recording_id` 和 `source_session_id`。

### 10.1 样本类型

#### A. Collision

通过时间偏移/平移，使 footprint 在某一未来时刻相交。

#### B. Near miss

保持空间路径近似相交，但最小安全距离：

```text
0.05m ~ 0.35m
```

空间变体不能直接平移整条目标轨迹后忽略可见性。为保持同一遮挡骨架下“当前隐藏、
未来出现”，默认以目标当前隐藏位姿为枢轴，对完整 `MotionSnippet` 轨迹和逐帧 yaw 做
同一个刚性旋转；若单纯旋转不能达到所需间距，允许先沿传感器到当前目标的盲区射线
向后做配置上限内的刚性平移，再做枢轴旋转。每个候选仍须重新通过静态地图、遮挡物、
上下文对象、速度/加速度和可见性检查；禁止逐帧扭曲轨迹。

#### C. Temporal-safe hard negative

空间路径相交，但时间错开：

```text
±0.8s ~ ±1.5s
```

时间变体只能从独立配置给出的冻结偏移序列，在同一条 23 点真实 snippet 内重设冲突
锚点，不得外推；它必须同步重建 history/current/future，保持空间路径相交、同步
footprint 不相交，并重新满足当前隐藏和连续出现条件。某个冻结偏移若因 source 锚点、
静止片段或其他 typed-transplant 契约被拒绝，生成器必须记录稳定、低基数的
`temporal_transplant:<reason>` 并继续后续偏移；不能把可预期候选拒绝升级为整个 pair
任务异常，rejection key 也不得包含浮点偏移值造成高基数统计。
重建后的带目标 temporal variant 必须仍与母事件指向同一
`source_recording_id` / `source_session_id`，不得从同 recording 的其他 session
借用时间片段或只复制 recording 级 provenance。

正式 SOP06 使用 `independent_partial_pairs_v1`。它在已验收的 collision 母事件
及其遮挡骨架上独立尝试 temporal-safe；一个时间偏移失败只会记录该变体的
稳定原因，不会重新采样遮挡物，也不会删除母事件或其他已合法变体。历史
`joint_environment_pair_v2` 联合 multi-LOS 搜索仅作迁移记录；正式 v5 producer
和 consumer 必须以 `joint_environment_pair_v2_retired` 拒绝它，不得兼容解释。

#### D. Spatial-safe hard negative

时间接近，但目标动态对象横向偏移，使最小距离：

```text
0.5m ~ 1.0m
```

#### E. Irrelevant hidden dynamic object

盲区里有目标动态对象，但完全不接近当前候选轨迹；同步 signed clearance 至少为
`1.5 m`。

#### F. Empty blind spot

同一场景只移除 `target_dynamic_object_id`；其他动态对象必须保持不变。

配对组的唯一母事件必需位是 `collision`；训练所需额外 contrast 数可为 0。
near-miss、temporal-safe、spatial-safe、irrelevant-hidden 和 empty 都独立保留，
不允许某一种 negative 失败使整组失效。每组仍输出按六类冻结顺序的 coverage
mask 与每个缺位的低基数原因，便于后续分层采样和失败诊断。

consumer 必须显式区分三条路径：母事件渲染、部分组训练渲染、条件完整组审计。
部分组即使恰好包含六位，训练路径也固定 `audit_certified=false`；只有 coverage
mask 全为 true、`is_complete=true` 且 `eligible_for_strict_evaluation=true` 的组才能
进入 conditional complete audit。完整六位不是训练发布门槛。

为避免多对象上下文破坏 paired 因果对照，主训练和 paired evaluation 要求非目标
动态对象在所有 variant 中完全相同，且不形成 collision/near miss。违反者拒绝，
或标记为 `multi_object_context` 后仅进入自然分布/OOD 分析。保留
`empty_blind_spot` 事件名时，其严格含义是“target-empty”，不是删除场景中的全部
动态对象。

报告必须分开给出 SOP05 母事件生成成本、SOP06 各变体的 coverage/missing-reason
分布，以及条件完整组率；后两者不得替代 SOP05 的候选阶段接受率。

### 10.2 训练比例建议

```yaml
collision: 0.20
near_miss: 0.20
temporal_safe: 0.20
spatial_safe: 0.15
irrelevant_hidden: 0.15
empty: 0.10
```

上述只是下游对“已存在的合法变体”做重采样时的参考权重，不是 SOP06
生成 quota，也不得用于丢弃不完整组。训练集可在报告原始 coverage 后另存一份
平衡采样 manifest；calibration/test 应额外生成更自然的低碰撞先验，例如：

```text
collision prior = 1%, 5%, 10%, 20%
```

### 10.3 配对 ID

所有同源变体共享：

```text
pair_group_id
base_state_id
trajectory_id
occluder_geometry_id
dynamic_object_snippet_id
target_dynamic_object_id
target_object_type
target_footprint_spec_digest
source_recording_id
source_session_id
```

`target_dynamic_object_id` 必须从生成世界的稳定输入确定性派生，并与该世界已有上下文
对象 ID 无冲突；原始 snippet 的 `source_object_id` 作为独立 provenance 字段保留，
不得复用它来覆盖上下文对象。

便于做 paired evaluation 和防止生成器捷径。

---

## 11. 历史观测与 BEV 信念渲染

### 11.1 模型可见输入与 oracle 世界必须分离

模型输入只能由模拟传感器可见内容生成；完整隐藏动态对象轨迹和 specs 只用于标签。

### 11.2 每个历史时刻渲染

对于 \(t-K+1,\ldots,t\)：

1. 读取真实机器人历史位姿；
2. 应用同一个程序化遮挡物/FOV 模型；
3. 按每个动态对象的 pose/yaw 和 circle/rectangle spec 放置历史 footprint；
4. 用 ray casting 计算 visible cells；
5. 写入 visible free / visible occupied；
6. 不可见区域写入 unknown；
7. 更新 last-seen occupancy；
8. 更新 occlusion age。

### 11.3 输入通道

第一版推荐：

```text
past_dynamic_occupancy[K]
past_visible_mask[K]
current_visible_free
current_visible_occupied
current_unobservable_mask
last_seen_occupancy
occlusion_age_map
static_obstacle_map
robot_footprint
robot_velocity_channel
robot_yaw_rate_channel
```

轨迹相关通道另行拼接：

```text
swept_volume_mask
time_to_arrival_map
braking_margin_map
centerline_map
```

### 11.4 age map

\[
A_t(x)=\min(t-t_{last-seen}(x),A_{max})/A_{max}
\]

推荐：

```text
A_max = 5.0s
从未看见的 cell = 1.0
当前可见 cell = 0.0
```

---

## 12. 风险 Ground Truth

### 12.1 只统计隐藏风险

对于动态体 \(j\)，若当前时刻不在模型可见区域，记为 hidden actor：

\[
j\in\mathcal H_t
\]

主标签只统计 \(\mathcal H_t\) 中动态体和候选轨迹的未来关系，避免混入普通可见动态避障。

### 12.2 二值碰撞标签

\[
Y_{coll}(\xi)=\mathbf1\left[
\exists j\in\mathcal H_t,\tau:\
B_r(q_\tau)\cap A_j(t+\tau)\neq\emptyset
\right]
\]

### 12.3 最近距离和 TTC 辅助标签

保存：

```text
min_clearance
first_collision_time
time_to_min_clearance
near_miss_label
```

### 12.4 连续风险严重度

注意：它是 oracle 定义的危险严重度，不声称为真实概率。

\[
Y_{risk}(\xi)=
\max_{j,\tau}
\exp\left(-\frac{\max(d_{j,\tau},0)}{\sigma_d}\right)
\exp\left(-\frac{\tau}{\sigma_t}\right)
\]

若碰撞：

\[
Y_{risk}=1
\]

推荐：

```yaml
sigma_distance: 0.5
sigma_time: 2.0
near_miss_distance: 0.35
```

启动时从解析后的 BEV 物理宽高计算
`no_object_clearance_sentinel_m = hypot(bev.width_m, bev.height_m)`，并写入 resolved
config 与 manifest。若没有相关隐藏动态对象，`risk_severity=0`、
`critical_object_id=None`，`min_clearance` 使用该有限哨兵；不得写入 `Inf/NaN`。
这类行仍计入 collision/severity 指标，但必须从 clearance 分布聚合中排除，避免把
哨兵解释为真实测量值。

### 12.5 风险标签伪代码

```python
def compute_risk_gt(robot_traj, hidden_object_trajectories, dynamic_object_specs, cfg):
    collision = False
    min_clearance = float("inf")
    first_collision_time = None
    max_severity = 0.0
    critical_object_id = None

    for k, robot_pose in enumerate(robot_traj.poses):
        tau = (k + 1) * cfg.future_dt
        robot_shape = inflated_robot_shape(robot_pose, cfg.robot_inflation)

        for object_id, object_traj in hidden_object_trajectories.items():
            object_shape = footprint_at_pose(
                object_traj.pose_at(k),
                dynamic_object_specs[object_id],
            )
            clearance = shape_distance(robot_shape, object_shape)
            if clearance < min_clearance:
                min_clearance = clearance
                critical_object_id = object_id

            if shapes_intersect(robot_shape, object_shape):
                collision = True
                if first_collision_time is None:
                    first_collision_time = tau
                severity = 1.0
            else:
                severity = math.exp(-max(clearance, 0.0) / cfg.sigma_distance)
                severity *= math.exp(-tau / cfg.sigma_time)

            max_severity = max(max_severity, severity)

    if critical_object_id is None:
        min_clearance = float(cfg.no_object_clearance_sentinel_m)

    return {
        "collision": int(collision),
        "risk_severity": float(max_severity),
        "min_clearance": float(min_clearance),
        "first_collision_time": first_collision_time,
        "near_miss": int((not collision) and min_clearance < cfg.near_miss_distance),
        "critical_object_id": critical_object_id,
        "critical_object_type": (
            dynamic_object_specs[critical_object_id]["object_type"]
            if critical_object_id is not None
            else None
        ),
    }
```

---

## 13. 风险训练数据格式

推荐使用 Zarr；依赖不方便时使用压缩 NPZ shards + JSONL metadata。

```python
RiskSample = {
    "sample_id": str,
    "split": str,
    "base_state_id": str,
    "pair_group_id": str,
    "event_type": str,
    "bev_history": float32[K, C, H, W],
    "state_channels": float32[C2, H, W],
    "trajectory_channels": float32[C3, H, W],
    "robot_state": float32[D],
    "collision_label": int,
    "risk_severity": float,
    "min_clearance": float,
    "near_miss": int,
    "first_collision_time": Optional[float],
    "metadata": dict,
}
```

metadata 至少保存：

```text
source_recording_id
source_session_id
source_object_id
source_participant_id（可用时，仅用于 split/leakage 审计）
dynamic_object_snippet_id
target_object_type / target_footprint_kind
target_type_policy_digest
trajectory primitive (v, omega)
occluder type / size / pose
blind spot type
random seed
```

当前 schema 3 SOP07 的已实现边界如下：

- `hidden_risk_gt_schema3_v1` 只遍历由调用方显式声明且经 renderer 验证当前
  不可见的 object IDs；world 中其他对象不能影响主标签。
- `RiskSample` 的 history/state channels 由 history-only renderer 构建；候选轨迹的
  四个 query maps 按冻结顺序拼接。`OracleWorld` future 只在独立标签分支内读取，
  不得写入模型输入或 provenance metadata。
- 每个 `risk_shard_npz_jsonl_v2` immutable shard 目录精确包含 `samples.npz`、`metadata.jsonl` 和
  `summary.json`。样本按 `sample_id` 稳定排序，固定 expected count 与单 split；
  写入 staging 后必须用正式 loader 重读并校验 schema/layout、shape/dtype/finite、
  manifest/semantic digest 和 split leakage，通过后才原子暴露。目标目录已存在时
  直接失败，禁止覆盖。v1 shard 不是可兼容输入，必须 fail closed。
- `scripts/04_generate_risk_dataset.py` 已实现正式编排入口。它对一个 formal
  `independent_partial_pairs_v1` group 的所有已成功 variants 做一次原子
  `RiskSample` 组装/发布；缺位仍由 coverage + stable reason 记录，不会为
  凑齐六类而丢弃合法部分组。只有 conditional complete audit 路径要求六位
  全齐。CLI 的 exact `expected_sample_count` 指整个 shard 的最终样本数，不是
  “每个 group 必须是 six-pack”。
- shard 作者可以通过 `split_audit_records` 对跨 shard 的 base/source recording
  泄漏执行 fail-closed 校验，并完整报告允许的 THÖR session overlap；当前 CLI
  未提供额外 `split_audit_records`，审计上下文只包含本 shard 自动生成的 manifest
  rows。全局跨 shard audit 必须由 collection 调用方提供全部 split 记录后执行，
  不得用单 shard 通过代替 recording 隔离证明或 session overlap 报告。

这些模块、CLI 和分片契约目前只有 unit/toy-fixture 及确定性验证，
不等于整条真实数据集生成任务已完成。10–100 真实样本 smoke、
全局跨 shard recording 隔离/session overlap 审计与 50k/240k 目标规模运行仍待执行；未验证前
不得宣称完整交付或达到该规模。

---

## 14. 占据预测在本项目中的位置

### 14.1 主方法不依赖先预测完整 occupancy

主模型：

\[
(z,\xi)\rightarrow trajectory\ risk
\]

不是：

\[
z\rightarrow occupancy\ map\rightarrow risk
\]

### 14.2 占据预测 baseline

至少实现一个：

```text
历史 BEV → future occupancy probability
future occupancy + swept volume → 手写概率聚合 → trajectory risk
```

可选复用：

- SOGMP-style stochastic occupancy prediction；
- Trajectron++/Social-STGCNN 预测行人轨迹后栅格化；
- 轻量 ConvGRU occupancy baseline。

### 14.3 可选辅助任务

主模型使用共享 encoder：

```text
BEV encoder
  ├── trajectory risk head     # 主任务
  └── hidden occupancy head    # 辅助监督，可选
```

推理时风险 head 不应强制经过 occupancy 输出。

---

## 15. 轨迹风险模型

### 15.1 最小结构

```text
Input = BEV history/state channels + trajectory channels
  ↓
Small ResNet / CNN
  ↓
Global pooling
  ↓
MLP
  ↓
Q50, Q80, Q90, Q95, p_collision
```

### 15.2 推荐增强结构

```text
Temporal BEV Encoder: ConvGRU or temporal CNN
Trajectory Query Encoder: CNN on swept/TTA/braking maps
Fusion: cross-attention or feature-wise modulation
Risk Heads:
  quantile head
  collision head
Optional:
  auxiliary occupancy head
```

Attention 设计：

```text
Query = trajectory token / trajectory map features
Key   = BEV latent features
Value = BEV latent features
```

不要直接把 predicted occupancy map 当唯一 Value。

### 15.3 Loss

\[
L_{risk}=L_{quantile}+\lambda_{coll}L_{BCE}+\lambda_{occ}L_{occ(aux)}
\]

Pinball：

\[
\rho_\beta(u)=\max(\beta u,(\beta-1)u)
\]

默认：

```yaml
quantiles: [0.5, 0.8, 0.9, 0.95]
lambda_collision: 1.0
lambda_occupancy_aux: 0.2
optimizer: AdamW
learning_rate: 0.0001
batch_size: 64
max_epochs: 40
```

### 15.4 风险校准

calibration split 上：

\[
s_i^+=\max(0,Y_i-Q_{0.9}(z_i,\xi_i))
\]

\[
q_{cal}=Quantile_{1-\alpha}(s_i^+)
\]

\[
U_{1-\alpha}=Q_{1-\alpha}+q_{cal}
\]

可按以下组别分别校准：

- blind spot type；
- critical blind area \(|U_t\cap S(\xi)|\)；
- occlusion age；
- dynamic object density；
- target object type / footprint kind（样本足够时）。

---

## 16. Scenario Bank：验证价值标签的隐藏世界集合

### 16.1 命名约束

论文和代码中称为：

```text
scenario-bank empirical decision risk
simulator-defined decision-value target
counterfactual verification-value target
```

不要称其为严格 Bayes ground truth。

### 16.2 每个当前观测生成 M 个可能世界

\[
\Omega_z=\{W^1,\ldots,W^M\}
\]

推荐：

```text
M = 16（起步）
M = 32（敏感性实验）
```

### 16.3 世界组成

建议固定配额：

```text
1 个真实/当前半合成 oracle 世界
2 个空盲区世界
5 个时间偏移变体
4 个空间偏移变体
2 个速度缩放变体
2 个无关隐藏动态对象变体
```

每个世界必须和当前模型输入一致：

- 当前 visible cells 的 occupancy 一致；
- 差异仅位于不可观测区域或未来状态；
- 不违反静态几何；
- 不产生当前时刻直接可见的新增动态对象；
- 每个 world 内 `dynamic_object_trajectories/specs` key 对齐；跨 variant 只允许计划定义
  的 target 缺失/状态变化，非目标对象不会被删除或改型。

### 16.4 世界先验权重

第一版均匀：

\[
w_m=1/M
\]

增强版可按训练数据事件频率或生成概率设置权重，但必须做敏感性实验。

---

## 17. 验证动作库

### 17.1 Motion primitives

```yaml
verification_actions:
  - name: yaw_left_10
    delta_yaw_deg: 10
    duration: 0.4
  - name: yaw_right_10
    delta_yaw_deg: -10
    duration: 0.4
  - name: yaw_left_20
    delta_yaw_deg: 20
    duration: 0.7
  - name: yaw_right_20
    delta_yaw_deg: -20
    duration: 0.7
  - name: forward_peek
    delta_forward_m: 0.30
    duration: 0.8
  - name: stop_scan
    duration: 0.6
```

侧移只在机器人底盘支持时加入。差速底盘不直接侧移。

### 17.2 验证动作成本

\[
c(v)=\lambda_t\Delta t+\lambda_l\Delta l+\lambda_\theta|\Delta\theta|
\]

默认：

```yaml
lambda_time: 0.04
lambda_distance: 0.05
lambda_yaw: 0.0015  # 每度
```

成本在价值 GT 中只扣一次。

---

## 18. 验证后的重新规划

在线系统执行 verify 后会更新观测并重新规划，因此 GT 也必须体现重规划。

### 18.1 以原始 nominal trajectory 为任务锚点

验证模型输入 \((z,\xi,v)\) 中，\(\xi\) 表示当前 nominal local plan/局部意图。

执行 \(v\) 后，从新位姿生成：

\[
\Xi^v(z,\xi)
\]

它是围绕原始 \(\xi\) 的局部目标/终点方向重新采样的候选集合。

### 18.2 重规划候选集

生成方式：

- 新机器人位姿作为起点；
- 原始 \(\xi\) 的 horizon 终点或参考方向作为局部目标；
- 使用相同 \(v,\omega\) rollout；
- 过滤静态碰撞；
- 可保留 stop/reject。

---

## 19. 验证价值 Ground Truth

### 19.1 当前不验证的经验决策风险

当前 nominal 轨迹的执行成本：

\[
C_E(\xi,W^m)=c_{task}(\xi)+\lambda_c I_{coll}+\lambda_n I_{near}
\]

拒绝成本：

\[
C_R=c_{reject}
\]

当前经验风险：

\[
\widehat{BR}(z,\xi)=
\min\left(
\frac1M\sum_m C_E(\xi,W^m),
C_R
\right)
\]

### 19.2 反事实验证观测

对每个 \(v,W^m\)：

1. 应用验证 primitive 得到新机器人位姿；
2. 对完整 oracle 世界 ray cast；
3. 得到新增可见 occupancy、visible mask、updated age map；
4. 得到 observation signature \(\phi(o_v^m)\)；
5. 生成重规划候选集 \(\Xi^v(z,\xi)\)。

### 19.3 Soft posterior

signature 推荐：

```text
new_visible_area
new_visible_area ∩ original swept volume
new_visible_area ∩ replanned swept volumes
number_of_new_visible_occupied_cells
minimum_visible_actor_distance_to_local_goal_corridor
whether_dynamic_actor_seen
critical_region_age_reduction
```

\[
w_{j|m}^{v}=\frac{\exp(-D(\phi_j,\phi_m)/\tau_o)}{\sum_k\exp(-D(\phi_k,\phi_m)/\tau_o)}
\]

默认：

```text
tau_o = 0.2
signature 每维先按 train statistics 标准化
```

### 19.4 观察后重新规划的经验风险

假设在真实世界 \(m\) 下看到观测 \(o_v^m\)，对所有新候选轨迹取最小后验经验损失：

\[
\widehat{BR}_v^m=
\min\left[
C_R,
\min_{\xi'\in\Xi^v(z,\xi)}
\sum_j w_{j|m}^{v} C_E(\xi',W^j)
\right]
\]

### 19.5 验证后期望风险

\[
PostRisk(z,\xi,v)=c(v)+\frac1M\sum_m\widehat{BR}_v^m
\]

### 19.6 净验证价值

\[
G^*(z,\xi,v)=\widehat{BR}(z,\xi)-PostRisk(z,\xi,v)
\]

解释：

- \(G^*>0\)：验证动作在 scenario-bank 目标下净收益为正；
- \(G^*<0\)：验证成本超过其决策收益；
- \(G^*\) 已经扣除 \(c(v)\)，在线不能重复加验证成本。

### 19.7 伪代码

```python
def verification_value_target(state, nominal_traj, action, worlds, cfg):
    # 1. 不验证时：执行 nominal trajectory 或拒绝
    execute_losses = [
        task_cost(nominal_traj) + oracle_safety_loss(nominal_traj, world, cfg)
        for world in worlds
    ]
    br_before = min(np.mean(execute_losses), cfg.reject_cost)

    # 2. 每个 world 产生反事实观测
    observations = []
    signatures = []
    replanned_sets = []

    for world in worlds:
        new_state, observation = simulate_verify(state, action, world, cfg)
        observations.append(observation)
        signatures.append(make_signature(observation, nominal_traj, cfg))
        replanned_sets.append(
            generate_replanned_candidates(new_state, nominal_traj, cfg)
        )

    # 验证动作后的机器人位姿相同，通常候选集可共用；
    # 若环境过滤依赖 world，则保留逐 world 候选集。
    post_risks = []

    for m, observed_signature in enumerate(signatures):
        posterior = soft_posterior(signatures, observed_signature, cfg.tau_o)
        candidate_set = replanned_sets[m]

        candidate_risks = []
        for traj_prime in candidate_set:
            expected_loss = 0.0
            for weight, world in zip(posterior, worlds):
                expected_loss += weight * (
                    task_cost(traj_prime)
                    + oracle_safety_loss(traj_prime, world, cfg)
                )
            candidate_risks.append(expected_loss)

        best_replanned = min(candidate_risks) if candidate_risks else float("inf")
        post_risks.append(min(best_replanned, cfg.reject_cost))

    post_risk = verification_cost(action, cfg) + np.mean(post_risks)
    value = br_before - post_risk

    return {
        "value": float(value),
        "useful": int(value > 0.0),
        "br_before": float(br_before),
        "post_risk": float(post_risk),
    }
```

---

## 20. 验证价值训练数据格式

```python
VerificationSample = {
    "sample_id": str,
    "split": str,
    "base_state_id": str,
    "nominal_trajectory_id": str,
    "verification_action_id": str,
    "bev_history": float32[K, C, H, W],
    "state_channels": float32[C2, H, W],
    "trajectory_channels": float32[C3, H, W],
    "verification_fov_mask": float32[1, H, W],
    "verification_action_vector": float32[Dv],
    "value_target": float,
    "useful_target": int,
    "br_before": float,
    "post_risk": float,
    "metadata": dict,
}
```

只对以下轨迹优先生成验证价值标签：

```text
候选轨迹扫掠体与不可观测区域有显著交集；
风险处于边界区间，如 0.05~0.7；
或者当前 execute 与 reject 代价接近。
```

---

## 21. 验证价值网络

### 21.1 输入

\[
(z,\xi,v)
\]

具体：

- BEV/history latent；
- trajectory query maps；
- action token；
- 执行验证动作后的几何 FOV mask（不包含 oracle occupancy）；
- action duration/distance/yaw。

注意：测试时只能输入“预计可见区域几何”，不能输入验证后真实看到的 hidden actor。

### 21.2 输出

```text
G_pred：净验证价值
P_useful：验证是否有正价值
```

### 21.3 结构

```text
Shared or frozen BEV encoder
Trajectory encoder
Verification geometry/action encoder
Fusion
  ├── value regression head
  └── useful classification head
```

### 21.4 Loss

\[
L_{verify}=L_{Huber}+\lambda_{cls}L_{BCE}+\lambda_{rank}L_{rank}
\]

同一 \((z,\xi)\) 下对动作做 pairwise ranking：

\[
L_{rank}=\max(0,m-G(v^+)+G(v^-))
\]

默认：

```yaml
lambda_classification: 1.0
lambda_ranking: 0.5
ranking_margin: 0.05
optimizer: AdamW
learning_rate: 0.0001
batch_size: 128
max_epochs: 30
```

---

## 22. 在线决策：避免重复扣验证成本

### 22.1 先计算不验证决策

\[
C_E(\xi)=c_{task}(\xi)+\lambda_U U(\xi)
\]

\[
C_R=c_{reject}
\]

\[
C_0=\min(C_E(\xi),C_R)
\]

### 22.2 验证后估计代价

因为 \(G\) 是净价值：

\[
\widehat C_V(v)=C_0-G_\psi(z,\xi,v)
\]

不再额外加 \(c(v)\)。

### 22.3 决策规则

```python
C_execute = task_cost(xi) + lambda_risk * calibrated_risk(z, xi)
C_reject = reject_cost
C_no_verify = min(C_execute, C_reject)

values = {v: value_model(z, xi, v) for v in actions}
best_v = max(values, key=values.get)
C_verify = C_no_verify - values[best_v]

if C_verify + verify_margin < C_no_verify:
    decision = ("verify", best_v)
elif C_execute <= C_reject:
    decision = ("execute", xi)
else:
    decision = ("reject", None)
```

执行 verify 后必须重新构建 BEV 并重新规划，不拼接原轨迹剩余部分。

---

## 23. 数据集规模建议

本节全部是后续运行目标，不是当前实现状态或已达成证据。风险样本数必须从
immutable shard summary 汇总，不得由请求数或理论乘法推定。

### 23.1 最小可行版

```text
base states: 5,000
候选轨迹/base: 12
risk samples/trajectory: 按实际合法 coverage 统计，不固定要求 4 或 6 个变体
risk samples: 240,000
```

对边界轨迹：

```text
nominal trajectories for verification: 2/base
verification actions: 6
verification samples: 60,000
scenario bank M: 16
```

### 23.2 推荐论文版

```text
base states: 10,000~20,000
candidate trajectories/base: 12~20
risk samples: 0.5M~1.5M
verification samples: 0.2M~0.8M
scenario bank: M=16
```

---

## 24. 防止半合成生成器捷径

### 24.1 配对样本

相同几何只改变目标动态对象的时间偏移、横向偏移或是否存在；其他动态对象不变。

### 24.2 随机化

- 遮挡物尺寸、朝向、类型；
- FOV 角度和 range；
- 目标动态对象速度和时间偏移；
- 轨迹 primitive；
- event time；
- background static map。

### 24.3 split 隔离

- recording 隔离；
- 通用数据集隔离可用的 session/participant；THÖR 的 recording-day session 允许重叠
  但必须报告，participant 明确标记 unavailable；
- snippet 隔离；
- generator seed 隔离；
- test 使用部分未见参数范围。

### 24.4 关键 controlled tests

1. 同样盲区面积，不同隐藏动态对象位置；
2. 同样路径空间交叉，不同到达时间；
3. 同样动态对象轨迹，不同候选机器人轨迹；
4. 同样验证可见面积，但对关键扫掠区域覆盖不同；
5. 空盲区与有无关动态对象的对照。

---

## 25. Baselines

### 25.1 风险

```text
B1 Last observation hold
B2 Age-decay heuristic
B3 Occupancy predictor + hand aggregation
B4 Occupancy predictor + learned aggregator
B5 Risk-only model without quantile
B6 Risk model without conformal calibration
Ours Trajectory-query quantile risk + calibration
Ours+Aux optional occupancy auxiliary supervision
```

### 25.2 验证

```text
V1 Never verify
V2 Always verify
V3 Max visible-area gain
V4 Max critical-swept-area coverage
V5 Max occupancy entropy reduction
V6 Learned value without ranking loss
Ours Counterfactual net decision value
```

---

## 26. 评价指标

### 26.1 风险模型

- Collision AUROC / AUPRC；
- Brier score；
- NLL；
- trajectory-level ECE；
- quantile coverage；
- upper-bound tightness；
- false-safe rate；
- pairwise risk ordering accuracy。

### 26.2 验证价值

- Huber/MSE；
- useful action F1；
- pairwise ranking accuracy；
- Spearman/Kendall；
- top-1 verification regret：

\[
Regret=G^*(v^*)-G^*(\hat v)
\]

### 26.3 闭环

- collision rate；
- near-miss rate；
- false-safe execution rate；
- verification count；
- unnecessary verification rate；
- reject rate；
- success rate；
- completion time；
- extra path/time cost；
- safety-efficiency Pareto。

---

## 27. 必须实现的敏感性实验

因为 scenario bank 和 soft posterior 是模拟器定义的近似目标，必须报告：

```text
M ∈ {8, 16, 32}
tau_o ∈ {0.1, 0.2, 0.5}
scenario composition variants
signature feature ablation
uniform vs non-uniform world prior
verification cost scale
```

结论应该关注：验证动作排序和闭环决策是否稳定，而不是声称目标具有严格概率意义。

---

## 28. 工程目录结构

```text
project/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data_thor.yaml
│   ├── generator_train.yaml
│   ├── generator_test.yaml
│   ├── risk_model.yaml
│   └── verify_model.yaml
├── src/
│   ├── datasets/
│   │   ├── thor_adapter.py
│   │   ├── jrdb_adapter.py
│   │   ├── split_manager.py
│   │   ├── base_state_index.py
│   │   └── snippet_library.py
│   ├── geometry/
│   │   ├── transforms.py
│   │   ├── footprints.py
│   │   ├── rasterization.py
│   │   ├── raycasting.py
│   │   └── collision.py
│   ├── planning/
│   │   ├── differential_drive.py
│   │   ├── trajectory_sampler.py
│   │   ├── trajectory_filters.py
│   │   └── query_maps.py
│   ├── generation/
│   │   ├── event_sampler.py
│   │   ├── occluder_sampler.py
│   │   ├── structural_blindspot.py
│   │   ├── dynamic_object_transplant.py
│   │   ├── paired_variants.py
│   │   ├── observation_renderer.py
│   │   ├── risk_gt.py
│   │   ├── scenario_bank.py
│   │   ├── counterfactual_verify.py
│   │   └── verification_gt.py
│   ├── models/
│   │   ├── bev_encoder.py
│   │   ├── risk_model.py
│   │   ├── occupancy_baseline.py
│   │   ├── verification_model.py
│   │   └── losses.py
│   ├── calibration/
│   │   ├── split_conformal.py
│   │   └── grouped_calibration.py
│   ├── evaluation/
│   │   ├── risk_metrics.py
│   │   ├── verification_metrics.py
│   │   ├── closed_loop.py
│   │   └── plots.py
│   └── utils/
│       ├── config.py
│       ├── seeding.py
│       └── logging.py
├── scripts/
│   ├── 00_make_splits.py
│   ├── 01_index_recordings.py
│   ├── 02_build_snippet_library.py
│   ├── 03_extract_base_states.py
│   ├── 04_generate_risk_dataset.py
│   ├── 05_train_occupancy_baseline.py
│   ├── 06_train_risk_model.py
│   ├── 07_calibrate_risk.py
│   ├── 08_generate_verification_dataset.py
│   ├── 09_train_verification_model.py
│   ├── 10_eval_offline.py
│   └── 11_eval_closed_loop.py
└── tests/
    ├── test_transforms.py
    ├── test_raycasting.py
    ├── test_trajectory_rollout.py
    ├── test_occluder_visibility.py
    ├── test_dynamic_object_transplant.py
    ├── test_pair_variants.py
    ├── test_risk_gt.py
    └── test_verification_gt.py
```

---

## 29. 推荐配置文件示例

```yaml
seed: 42
schema_version: "3.0.0"

bev:
  range_m: 16.0
  resolution_m: 0.1
  size: 160
  history_steps: 8
  history_dt: 0.2
  future_steps: 15
  future_dt: 0.2

robot:
  model: differential_drive
  length_m: 0.70
  width_m: 0.55
  inflation_m: 0.15
  max_linear_speed: 0.9
  max_angular_speed: 0.8

dynamic_objects:
  human:
    radius_m: 0.30
    carrier_radius_m: 0.45
    min_speed_mps: 0.30
    max_speed_mps: 2.00
    max_acceleration_mps2: 2.50
  carried_object:
    fallback_length_m: 0.80
    fallback_width_m: 0.20
    min_speed_mps: 0.05
    max_speed_mps: 2.00
    max_acceleration_mps2: 2.50
  unknown_dynamic:
    fallback_radius_m: 0.50
    min_speed_mps: 0.05
    max_speed_mps: 2.00
    max_acceleration_mps2: 2.50
  marker_geometry:
    extent_quantile: 0.95
    minimum_valid_frames: 20
    min_extent_m: 0.05
    max_extent_m: 3.0

trajectories:
  linear_velocities: [0.2, 0.4, 0.6, 0.8]
  angular_velocities: [-0.8, -0.4, 0.0, 0.4, 0.8]
  reverse_velocities: [-0.2, -0.4]
  reverse_probability: 0.2
  horizon_s: 3.0
  dt_s: 0.2

production_event_kind: environment
target_type_policy:
  whitelist: [human]
  weights: {human: 1.0, carried_object: 0.0, unknown_dynamic: 0.0}
conflict_time_range_s: [1.0, 2.2]
max_local_curvature_per_m: 1.0
crossing_angle_max_deg: 35.0
time_scale_range: [1.0, 1.0]
min_contiguous_visible_frames: 2

occluders:
  types: [wall, shelf, pillar]
  wall: {length_range_m: [1.0, 3.0], width_range_m: [0.2, 0.5]}
  shelf: {length_range_m: [1.0, 2.5], width_range_m: [0.4, 0.8]}
  pillar: {length_range_m: [0.4, 0.8], width_range_m: [0.4, 0.8]}

blind_reachability:
  algorithm_version: blind_reachability_quota_first_v3
  obstacle_proposals_per_trajectory: 64
  snippet_candidates_per_proposal: 64
  interaction_range_m: [1.0, 4.0]
  bearing_bin_count: 12
  yaw_step_deg: 30.0
  crossing_angle_step_deg: 5.0
  minimum_shadow_center_cells: 32
  chord_deviation_fastpath_m: 0.15
  unresolved_exact_fallback_per_anchor: 16

risk_gt:
  sigma_distance_m: 0.5
  sigma_time_s: 2.0
  near_miss_distance_m: 0.35

scenario_bank:
  size: 16
  posterior_temperature: 0.2
  reject_cost: 0.20

verification:
  useful_margin: 0.0
  decision_margin: 0.01
```

---

## 30. CLI 运行顺序

下列命令是目标编排顺序，不表示已完成真实数据规模验证。当前 SOP07 的
GT、`RiskSample`、`risk_shard_npz_jsonl_v2` API 和正式
`scripts/04_generate_risk_dataset.py` 编排入口已实现，但只完成
unit/toy-fixture 验证；10–100 真实样本 smoke、全局跨 shard audit 与
目标规模仍待执行。

```bash
# 1. 按 recording / participant 切分
python scripts/00_make_splits.py --config configs/data_thor.yaml

# 2. 解析原始记录
python scripts/01_index_recordings.py --split train
python scripts/01_index_recordings.py --split calibration
python scripts/01_index_recordings.py --split val
python scripts/01_index_recordings.py --split test

# 3. 每个 split 和对象类型单独建立 MotionSnippet library
python scripts/02_build_snippet_library.py --split train
python scripts/02_build_snippet_library.py --split calibration
python scripts/02_build_snippet_library.py --split val
python scripts/02_build_snippet_library.py --split test

# 4. 提取 base states
python scripts/03_extract_base_states.py --all-splits

# 5. 生成事件中心风险数据（CLI 已实现；尚未做真实 smoke/规模验证）
python scripts/04_generate_risk_dataset.py \
  --sop03-root <schema3-sop03-root> \
  --sop04-root <schema3-sop04-root> \
  --sop04-handoff-digest <64-hex> \
  --sop05-root <schema3-sop05-root> \
  --sop05-publication-digest <64-hex> \
  --config configs/base.yaml \
  --paired-config configs/paired_variants.yaml \
  --split train \
  --seed 42 \
  --output-dir outputs/event_centered_blind_spot/schema-v3/risk-data/main-seed42-v1/train/shard-00000 \
  --shard-index <nonnegative-int> \
  --expected-event-count <positive-int> \
  --expected-sample-count <positive-int> \
  --checksum-workers 8

# 6. 训练 occupancy baseline（可并行/可后做）
python scripts/05_train_occupancy_baseline.py --config configs/risk_model.yaml

# 7. 训练主风险模型
python scripts/06_train_risk_model.py --config configs/risk_model.yaml

# 8. 独立 calibration groups 做校准
python scripts/07_calibrate_risk.py \
  --checkpoint outputs/event_centered_blind_spot/schema-v3/risk-model/main-seed42-v1/best.pt

# 9. 生成验证价值目标
python scripts/08_generate_verification_dataset.py \
  --split train \
  --scenario-bank-size 16

# 10. 训练验证价值模型
python scripts/09_train_verification_model.py --config configs/verify_model.yaml

# 11. 离线评估
python scripts/10_eval_offline.py --split test

# 12. 闭环/仿真评估
python scripts/11_eval_closed_loop.py --benchmark arena
```

---

## 31. 单元测试与验收条件

### 31.1 几何测试

- SE(2) 变换前后距离保持；
- 轨迹 rollout 与解析圆弧一致；
- swept volume 覆盖所有 footprint；
- collision 几何结果正确。

### 31.2 遮挡测试

- 遮挡物后 cell 被 ray casting 标为不可见；
- 遮挡物本身不阻塞候选轨迹；
- 结构性 FOV 外所有 cell 不可见；
- 目标动态对象当前不可见，未来可从边缘出现；
- circle/rectangle footprint 与逐帧 yaw 的遮挡和碰撞结果正确。

### 31.3 配对样本测试

同一 pair group（不要求每个组必须完整）：

- collision variant 的 min clearance ≤ 0；
- temporal-safe 空间路径相交但时间不相交；
- empty variant 只移除目标动态对象，其他动态对象不变；
- 除指定变量外几何相同。
- collision 母事件可独立通过；各 negative 缺失时 coverage mask 与 reason 精确对齐；
- 部分组渲染永不设置 `audit_certified=true`，完整组只用于条件审计。

### 31.4 风险 GT 测试

- 碰撞时 `collision=1` 且 `risk_severity=1`；
- 距离越近风险不降低；
- 同距离下越早出现风险越高；
- visible actor 不计入 hidden-risk 主标签。
- circle-circle、circle-rectangle、rectangle-rectangle 和多对象最小 clearance
  与手算一致。

### 31.5 验证价值测试

构造人工小场景：

- 一个动作正好看到冲突动态对象（至少包含矩形对象用例），价值应高；
- 一个动作只看到无关区域，价值应低/负；
- 增大验证成本后价值应下降；
- 空盲区场景验证通常不应有正价值；
- `PostRisk` 已含 `c(v)`，在线不可重复加成本；
- verify 后候选集必须从新位姿重生成。

### 31.6 最小工程验收

```text
✓ 能从一个 recording 生成 base states
✓ 能建立独立 train/test snippet libraries
✓ 能从一条直线候选轨迹生成墙后横穿 human-target 事件
✓ 能独立发布 collision 母事件，并对各 paired negative 报告 coverage/缺失原因
✓ 能渲染 K 帧不完整 BEV 和完整 oracle
✓ 能计算 collision/risk GT
✓ 能训练过拟合 1000 个 risk samples
✓ 能生成至少两个验证动作价值不同的样本
✓ 能训练 verification model 过拟合小数据
✓ 无 recording/sample leakage
```

---

## 32. 推荐实现顺序

### Phase A：最小生成器（最高优先级）

1. 解析 THÖR robot 与所有非机器人 dynamic-object trajectories/specs；
2. 坐标转换和 resampling；
3. 候选轨迹 rollout；
4. 按 split/type 建立 MotionSnippet libraries；
5. 在直线轨迹旁放矩形墙体；
6. 移植横穿 snippet；
7. 发布 collision 母事件，独立尝试 temporal-safe 及其他 negative；
8. 渲染 BEV；
9. 风险 GT。

### Phase B：风险模型

1. risk-only CNN；
2. BCE + continuous severity；
3. quantile heads；
4. occupancy baseline；
5. calibration。

### Phase C：完整事件多样性

1. 提高各 paired variant 的独立 coverage，并对完整六位组做条件审计；
2. 结构性盲区；
3. 混合盲区；
4. hard negatives；
5. 多先验测试。

### Phase D：验证价值

1. verification primitive geometry；
2. scenario bank；
3. counterfactual ray casting；
4. replan candidate set；
5. G* 生成；
6. value model。

### Phase E：闭环

1. execute / verify / reject controller wrapper；
2. 自建 2D simulator；
3. Arena-Rosnav 接入；
4. safety-efficiency Pareto。

---

## 33. 预期结果与失败判据

### 33.1 期望结果

风险模型：

- 比 `occupancy + hand aggregation` 更低 trajectory-level ECE；
- 更低 false-safe rate；
- paired hard negatives 上不只看路径空间交叉；
- conformal 后 coverage 接近目标值。

验证模型：

- 比 visible-area gain 更低 top-1 regret；
- 比 always verify 少做验证；
- 比 never verify 更低碰撞/false-safe；
- 对 M、\(\tau_o\)、scenario composition 具有可接受稳定性。

### 33.2 失败判据

若发生以下情况，需要调整研究设定：

- risk-only 模型明显不如 occupancy baseline；
- 模型仅根据盲区面积或 swept-volume 面积预测风险；
- verification value 对 scenario bank 参数极度敏感；
- 验证动作排序和闭环收益无相关性；
- paired temporal-safe 样本大量误判；
- train/test snippet 或 recording 泄漏；
- 半合成 test 好，但 Arena/JRDB 完全失效。

---

## 34. 论文中的正确表述

### 可以主张

1. 提出事件中心的反事实盲区场景生成器，保留所有 typed 动态对象，并以真实 human
   轨迹先验作为主目标分布构造大规模 collision/near-miss/hard-negative 样本；
2. 直接学习 trajectory-conditioned hidden risk，优于 cell-wise occupancy + aggregation；
3. 学习 simulator-defined counterfactual verification value，优于可见面积/信息增益式验证策略；
4. 在统计校准和闭环实验中减少 false-safe execution 和不必要验证。

### 不应主张

1. scenario bank 是真实后验；
2. G* 是严格 Bayes ground truth；
3. 连续 risk severity 是真实碰撞概率；
4. 模型提供无条件安全保证；
5. 半合成生成器等价于真实世界分布。

---

## 35. 最终一句话

> 本项目从真实机器人和动态对象轨迹中提取运动先验，但不依赖自然数据中稀少的遮挡冲突事件；它围绕候选局部轨迹上的潜在冲突点，程序化构造遮挡物和隐藏动态对象，生成成对的碰撞、近碰和 hard-negative 场景，用完整 oracle 世界训练轨迹相关风险模型，并通过反事实验证动作与重规划计算模拟器定义的净决策价值，最终在 execute、verify 和 reject 之间做局部规划决策。
