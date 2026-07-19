# 盲区轨迹风险与主动验证：并行加速实施计划

> 项目暂定名：**Learning When to Verify: Decision-Calibrated Hidden Risk for Robot Local Planning**  
> 中文暂定名：**学习何时验证：面向机器人局部规划的决策校准隐藏风险**  
> 计划目标：在不牺牲数据隔离、标签一致性和实验可信度的前提下，尽可能并行推进数据、模型、验证价值、闭环评估和论文写作。  
> 推荐冲刺周期：**2026-07-14 至 2026-07-28**。  
> 适用对象：多人研发团队、多个 Codex/Agent 并行协作，或单人通过多个独立工作线程推进。

---

## 0. 最终交付物定义

### 0.1 必须完成的最小论文闭环

在截止前，至少应完成以下闭环：

```text
THÖR-MAGNI 原始轨迹
    ↓
按冻结 policy 做 recording-level 切分与 provenance audit
    ↓
按 split/type 隔离的真实 MotionSnippet libraries
    ↓
事件中心半合成盲区场景生成
    ↓
候选局部轨迹与风险 GT
    ↓
占据预测 + 手工风险聚合 baseline
    ↓
轨迹条件风险模型 + 校准
    ↓
反事实验证价值 GT
    ↓
验证价值模型
    ↓
execute / verify / reject 离线闭环
    ↓
核心表格、消融、案例图
```

Schema 3 已冻结 future-endpoint 时间契约；其中继续使用 dynamic-object schema v2
引入的三类对象语义。主论文目标事件默认使用 `human` snippet；所有
`carried_object/unknown_dynamic` 仍保留在 observed/oracle world，并参与可见性、
占据和真实 footprint 风险计算。W2 及之后的任务必须拒绝 schema 2 的
`q0...q14 / 0.0...2.8 s` artifact，并在每一级
门禁后重建下游数据、checkpoint、calibration 和结果；非人 target 只作为显式扩展。
下游只能读取 contract/snippet 冻结的 `object_type` 与 footprint spec，禁止根据原始
body name、role 或文件名重新分类。W2 的 target-type policy 必须包含白名单和三类
对象的完整归一化权重映射；主论文默认 `human=1.0`、其余为 `0.0`。解析后 policy 的
稳定 digest 必须逐级写入并校验所有下游 manifest、checkpoint 和评测结果。

### 0.2 必须交付的代码与结果

1. 一个可重复运行的 THÖR-MAGNI 数据适配器；
2. 一个可批量生成遮挡/结构性盲区事件的 2D-BEV 生成器；
3. 一个候选局部轨迹 rollout 模块；
4. 二值碰撞、连续风险、near-miss、TTC 等 GT 生成器；
5. 至少一个 occupancy/trajectory forecasting baseline；
6. 轨迹条件风险模型；
7. 校准模块；
8. 反事实验证价值标签生成器；
9. 验证价值模型；
10. execute / verify / reject 决策器；
11. 离线闭环或轻量 2D 闭环评估；
12. 一键运行脚本、配置文件、日志和核心图表。

### 0.3 可延后或放入补充材料的内容

- Arena-Rosnav 完整 ROS2 闭环；
- JRDB 跨数据集泛化；
- 复杂 cross-attention 风险网络；
- 占据辅助 head；
- 多种底盘模型；
- 3D 点云或图像前端；
- 真实机器人部署。

这些内容不应阻塞论文最小闭环。

---

# 1. 并行化的核心原则

## 1.1 先冻结接口，不等待真实数据全部准备好

项目最容易串行阻塞的方式是：

```text
数据全部整理完
    ↓
再写样本生成器
    ↓
再训练风险模型
    ↓
再写验证价值模块
```

推荐方式是第一天冻结数据接口，并同时提供一个小型 `ToyWorldDataset`：

```text
接口冻结 + toy fixture
    ├── 数据 Agent：接真实 THÖR 数据
    ├── 场景 Agent：开发事件生成器
    ├── 风险 Agent：先用 toy 数据开发模型和训练代码
    ├── 验证 Agent：先用 toy world 开发 G* 生成器
    └── 评估 Agent：先用 mock 输出开发指标和图表
```

真实数据一旦就绪，只替换数据源，不重写模型和评估逻辑。

## 1.2 所有工作流通过文件契约和类型契约连接

禁止 Agent 之间靠口头约定传递数组含义。所有核心数据必须使用统一 schema。

## 1.3 每条工作流同时定义三档结果

- **理想结果**：论文有竞争力时希望达到的结果；
- **最低验收结果**：可以继续进入下一阶段的最低条件；
- **降级方案**：达不到最低线时，如何缩减范围而不是停工。

## 1.4 先做可证伪的最小实验

每个模块都应先在 100—1000 个 toy/半合成样本上证明：

- 输入输出正确；
- 标签单调性正确；
- 不存在 GT 泄漏；
- 模型能过拟合小样本；
- 指标计算没有方向错误。

确认这些之后才扩展到几十万样本。

---

# 2. 推荐团队/Agent 分工

## 2.1 理想七 Agent 配置

| Agent | 工作流 | 主要责任 |
|---|---|---|
| A0 | 集成与接口 | schema、配置、CI、合并、实验注册 |
| A1 | 数据适配 | THÖR 解析、切分、轨迹片段库 |
| A2 | 场景与 BEV | 事件中心生成、遮挡、可见性、BEV 历史 |
| A3 | 轨迹与风险 GT | rollout、扫掠体、碰撞/连续风险标签 |
| A4 | 风险模型 | occupancy baseline、risk-only、校准、消融 |
| A5 | 验证价值 | scenario bank、反事实观测、G*、价值模型 |
| A6 | 评估与论文 | 离线闭环、指标、图表、实验表、方法写作 |

## 2.2 四 Agent 紧凑配置

| Agent | 合并任务 |
|---|---|
| A | A0 + A6：集成、评估、论文 |
| B | A1 + A2：真实数据、事件生成、BEV |
| C | A3 + A4：轨迹、风险 GT、风险模型 |
| D | A5：验证价值全链路 |

## 2.3 单人多线程配置

单人应同时维护四条独立工作线程：

```text
Thread-DATA     数据与事件生成
Thread-RISK     风险模型与 baseline
Thread-VERIFY   验证价值标签与模型
Thread-EVAL     指标、图表、论文
```

每天只在固定时间合并，避免频繁上下文切换。

---

# 3. 第一天必须冻结的工程接口

## 3.1 核心数据类型

```python
@dataclass
class BaseState:
    state_id: str
    split: str
    recording_id: str
    dynamic_object_ids: tuple[str, ...]
    timestamp: float
    robot_history: np.ndarray      # [K, 3] -> x, y, yaw
    robot_state: np.ndarray        # [D]
    visible_dynamic_object_history: dict  # object_id -> [K, 3]
    visible_dynamic_object_specs: dict    # object_id -> type + footprint
    static_map_local: np.ndarray | None
    metadata: dict


@dataclass
class LocalTrajectory:
    trajectory_id: str
    poses: np.ndarray              # [T, 3]
    controls: np.ndarray           # [T, 2] -> v, omega
    swept_mask: np.ndarray         # [H, W]
    tta_map: np.ndarray            # [H, W]
    braking_map: np.ndarray        # [H, W]
    task_cost: float
    metadata: dict


@dataclass
class OracleWorld:
    world_id: str
    base_state_id: str
    static_occupancy: np.ndarray   # [H, W]
    dynamic_object_trajectories: dict  # object_id -> [T, 3]
    dynamic_object_specs: dict         # object_id -> type + footprint
    occluders: list
    blind_spot_config: dict
    random_seed: int
    metadata: dict


@dataclass
class RiskSample:
    sample_id: str
    split: str
    base_state_id: str
    pair_group_id: str
    bev_history: np.ndarray        # [K, C, H, W]
    state_channels: np.ndarray     # [Cs, H, W]
    trajectory_channels: np.ndarray# [Ct, H, W]
    robot_state: np.ndarray
    collision_label: int
    risk_severity: float
    min_clearance: float
    near_miss: int
    first_collision_time: float | None
    metadata: dict


@dataclass
class VerificationSample:
    sample_id: str
    split: str
    base_state_id: str
    nominal_trajectory_id: str
    verification_action_id: str
    bev_history: np.ndarray
    state_channels: np.ndarray
    trajectory_channels: np.ndarray
    verification_fov_mask: np.ndarray
    verification_action_vector: np.ndarray
    value_target: float
    useful_target: int
    br_before: float
    post_risk: float
    metadata: dict
```

全局产物契约为 `schema_version=3.0.0`。其中保留 dynamic-object schema v2
冻结的 `human`、`carried_object`、`unknown_dynamic` 三类对象语义；这不得被误解
为仍可发布全局 schema 2 产物。数据适配器保留所有有效非机器人 BODY，以
`recording_id::body_name` 作为对象 ID。THÖR split 固定评测未见 recording/已知
recording-day session：recording 和 source object 不跨 split，session 重叠逐项报告，
participant identity 标记 unavailable。`human` 使用配置圆 footprint，非人对象优先
使用 QTM marker P95 矩形估计并在无有效 marker 时回退配置。snippet 必须按
`snippets/<split>/<object_type>/` 隔离；旧序列化产物全部重建，不做静默兼容。

## 3.2 固定 tensor 约定

第一版统一使用：

```yaml
bev:
  width_m: 16.0
  height_m: 16.0
  resolution_m: 0.1
  height_cells: 160
  width_cells: 160

history:
  steps: 8
  dt: 0.2

future:
  horizon_s: 3.0
  dt: 0.2
  steps: 15
```

## 3.3 样本存储

推荐：

```text
metadata：Parquet / JSONL
大数组：Zarr
依赖受限时：NPZ shards + JSONL
```

`data/` 仅作为原始输入且只读。生成产物写入：

```text
outputs/
  splits/
  recording_indexes/<split>/
  base_state_indexes/<split>/
  snippets/<split>/<object_type>/
  event_centered_blind_spot/schema-v3/
    risk-data/<run-id>/
    risk-model/<run-id>/
    verification-data/<run-id>/
    verification-model/<run-id>/
    closed-loop/<run-id>/
    reports/<run-id>/
```

以上运行产物不得提交；每个 schema-v3 run 目录必须原子创建，已有 run ID 直接失败，
不得静默覆盖。

## 3.4 Toy fixture

A0 必须在第一天提供固定 toy fixture：

```text
4 个 base states
6 条候选轨迹/state
4 个隐藏世界/state
4 个验证动作
```

它应覆盖：

- 空盲区；
- 碰撞；
- near miss；
- 时序安全；
- 一个高价值验证动作；
- 一个负价值验证动作。

所有 Agent 都必须能在 toy fixture 上运行。

### 理想结果

- 所有 schema 有类型检查；
- toy fixture 的风险、可见性和 G* 有人工可验证答案；
- 所有训练/评估脚本无需真实 THÖR 数据即可启动。

### 最低验收

- 能序列化和读取 `RiskSample`、`VerificationSample`；
- 一个风险 batch 和一个验证 batch 能通过模型 forward。

### 降级方案

若 Zarr 配置耗时，立即使用 NPZ + JSONL，不让存储格式阻塞研究。

---

# 4. 并行工作流总图

```text
                    ┌─ W1: THÖR 数据适配/切分/snippet ─┐
W0: 接口/CI/toy ────┤                                   ├─ W2: 事件/BEV ─ W3: 风险 GT ─┐
                    └─ ToyWorldGenerator ───────────────┘                          │
                                                                                   ├─ W5: 风险模型/校准 ─┐
                    Toy/真实 world ─ W6: 验证 GT ─ W7: 验证价值模型 ───────────────┤
                                                                                   └─ W8: 闭环/图表/论文
                    W4: occupancy baseline 可在 W2 输出少量数据后独立推进 ──────────┘
```

关键并行策略：

1. W4/W5 不等 W1 全量完成，先在 toy 和 5k 样本上开发；
2. W6/W7 不等风险模型成熟，先用 oracle safety loss 生成 G*；
3. W8 不等真实模型，先用 oracle/mock policy 开发闭环和指标；
4. 论文方法图、数据生成图和公式从第 2 天开始写，不等待最终结果。

---

# 5. W0：项目基础设施、接口与集成

## 5.1 任务

- 初始化仓库和目录；
- 固定 schema 与配置；
- 实现 toy fixture；
- 实现随机种子管理；
- 实现实验注册、日志和 checkpoint 路径；
- 配置最小 CI；
- 维护 `STATUS.md` 和 `DECISIONS.md`；
- 负责每日合并。

## 5.2 交付文件

```text
project/
  pyproject.toml
  configs/base.yaml
  schemas.py
  tests/test_schema.py
  tests/test_toy_fixture.py
  tests/fixtures/toy_world.py
  STATUS.md
  DECISIONS.md
```

## 5.3 理想结果

- `pytest` 全部通过；
- 所有脚本都接受 `--config`、`--seed`、`--output-dir`；
- 每个实验自动保存 config、git commit、环境信息和 metrics；
- 不同 Agent 的产物可无修改互换。

## 5.4 最低验收

- toy fixture 可加载；
- schema 不再随意变化；
- 任一模型训练脚本可跑 10 step。

## 5.5 风险与降级

- 不做复杂 MLOps；
- 日志优先 CSV/JSON，W&B 可选；
- CI 只覆盖核心单元测试。

---

# 6. W1：THÖR 数据适配、分组切分与 typed 动态对象片段库

## 6.1 任务

1. 解析机器人轨迹；
2. 解析所有非机器人 BODY 的 pose/marker，分类为
   `human/carried_object/unknown_dynamic`；
3. 统一时间戳并重采样；
4. 按冻结的 recording assignment 切分；THÖR session overlap 允许但必须报告；
5. 提取 base states；
6. 按 split/type 提取真实 `MotionSnippet`；
7. 输出速度、加速度、曲率、footprint/orientation source 和 rejection 统计。

## 6.2 严格切分流程

```text
原始 recordings
    ↓
按 recording_id 冻结；补齐官方 recording-day session provenance
    ↓
train / calibration / val / test
    ↓
各 split 独立建立 snippet library
    ↓
各 split 内独立生成半合成事件
```

禁止在生成样本后随机切分。

## 6.3 Snippet 过滤

生产片段固定为 4.4 秒、23 个真实采样点；速度阈值按 object type 读取
schema 3 配置中保留的 dynamic-object v2 类型字段：

```yaml
motion_snippet_layout_version: history8_current7_future15_v1
duration_s: 4.4
sample_dt_s: 0.2
sample_count: 23
history_steps: 8
current_index: 7
future_steps: 15
human_min_speed_mps: 0.30
nonhuman_min_speed_mps: 0.05
max_speed_mps: 2.00
max_gap_s: 0.3
max_accel_mps2: 2.50
```

索引 `0:8` 为 `-1.4...0.0 s` 的真实历史，索引 7 为当前，索引 `8:23` 为
`0.2...3.0 s` 的真实未来。每个 snippet 保存 object type、相对 pose/yaw、速度、
footprint spec、source recording/session/object、完整布局和新 split digest。禁止补点、
外推、跨 gap 插值；旧 16 点/3.0 s library 由 loader 明确拒绝。

## 6.4 理想结果

- 有效 base states ≥ 10,000；
- 有效 human snippets 理想 ≥ 5,000；非人 snippets 按实际数量独立报告；
- train/test snippet 来源完全隔离；
- 重采样前后的速度、加速度和曲率分布均有量化报告；
- 坐标变换可逆误差 < 1e-4；
- 无 NaN、无严重时间断裂。

## 6.5 最低验收

- base states ≥ 2,000；
- snippets ≥ 1,000；
- 能画出 50 条机器人/typed dynamic-object 轨迹进行人工检查；
- recording/source-object/disallowed overlap 为 0；THÖR session overlap 枚举后允许。

## 6.6 降级方案

若 THÖR 原始传感器解析复杂：

- 只读取官方导出的机器人与所有非机器人刚体坐标/marker 轨迹；
- 静态地图和障碍物由程序化生成器提供；
- 不解析 RGB、点云和复杂传感器包。

---

# 7. W2：事件中心半合成场景与 BEV 渲染

## 7.1 任务

- 接收 `BaseState`、`LocalTrajectory`、typed `MotionSnippet`；
- `LocalTrajectory` 只接受 `sop04_audited_bank_v2` /
  `future_endpoints_dt_to_horizon_v1`：15 个 future pose 是 `q1…q15`、时间为
  `0.2…3.0 s`，冲突零基 index `k` 映射为 `(k+1)*0.2 s`；旧 `0.0…2.8 s`
  布局不得兼容
- SOP05 preflight 必须由 CLI 显式接收目录外可信 SOP04 handoff digest，并在
  `sop05_input_lock_v2` / run identity 中绑定 bank/layout/time/offset、bank semantic
  digest 和 external handoff digest
- 只接受 `history8_current7_future15_v1`；固定使用
  `history=poses[0:8]`、`current=poses[7]`、`future=poses[8:23]`，未来冲突锚点的
  source time 为 `1.4 + conflict_time_s`；
- SOP05 正式生产算法固定为 `blind_reachability_quota_first_v3`，当前只发布
  environment collision 母事件；历史 `joint_occluder_first_v4` 产物必须拒绝；
- producer/report 固定为 `sop05_generation_run_v6` /
  `sop05_pair_generation_report_v4`，并显式拒绝紧邻旧 v5/v3；pair process pool 的
  bounded window 同时计算 pending futures 与按 rank 待消费 reports，建议 Slurm
  `cpus-per-task == --workers`，不得用多个 `ntasks` 重复启动同一 producer；
- candidate 变换固定为 `reachability_candidate_se2_v2`；`ReachabilityIdentity` /
  candidate 显式绑定 `source_session_id`，并通过 `source_snippet_id` 在可信库中
  间接定位 recording。transform payload 与 generated event/world identity 必须显式同时
  绑定非空 `source_recording_id` 与 `source_session_id`；
  `blind_reachability_first_v1` / `reachability_candidate_se2_v1` 或缺少 session
  lineage 的产物必须 fail closed；
- 运行前对 schedule 内唯一 `trajectory_id` 稳定排序，每条只预热一次机器人
  canonical future 连续扫掠体；单进程直接复用，多进程通过 `fork` copy-on-write
  复用父进程已预热的 `robot_sweep_cache_v1`。base-specific history 与 seam
  仍按 pair 构建一次，不在同一 pair 的遮挡提议内重复构建；
- 按交互距离、
  bearing bin、遮挡物类型/尺寸/yaw 生成稳定 causal free-space 候选，先拒绝
  静态重叠或与机器人/上下文对象扫掠冲突的遮挡物；
- 用与正式 renderer 一致的栅格、ray casting 和遮挡几何生成 current blind-region
  mask，并按 footprint/yaw 生成目标中心可放置 mask；
- 在候选轨迹上枚举对齐的冲突时刻/点，从真实 23 点 snippet 当前位置到冲突点
  构造 reachability arc；先用 mask 与 chord fast path 筛选，无法证明安全的候选进入
  每 anchor 有上限的 exact fallback；
- 对目标 snippet 只做该 reachability candidate 冻结的 SE(2) 变换，正式路径固定
  `time_scale=1.0`，不外推、不逐帧扭曲；主分布默认 human target；
- exact 验证必须同时覆盖真实 circle/rectangle footprint、静态和帧间连续净空、
  当前不可见、未来连续出现以及选定 future endpoint 的 collision；
- 几何版本栈固定为 `causal_occluder_schedule_v1` /
  `causal_occluder_proposal_v2` / `blind_region_causal_delta_v2` /
  `footprint_center_mask_causal_delta_v2` /
  `occluder_collision_sweep_preparation_v2`。proposal identity 绑定正式
  interaction region，blind region 必须是 baseline/current renderer 的 causal delta
  与该 region 精确取交；连续碰撞 preparation v2 对 circle 忽略 yaw-only
  旋转加密，rectangle 仍保留 yaw 加密；
- summary 中保存 obstacle proposal、reachability transform/chord 与 exact validation 三层
  可对账计数和稳定 rejection reasons；
- 从完整 `target_type_policy` 解析白名单和三类归一化权重，并把稳定 digest 写入事件
  manifest；非人 target 只能由显式扩展配置启用；
- 直接使用 snippet 冻结的 type/spec；生成的 target ID 必须确定且不与上下文 ID
  冲突，原 `source_object_id` 独立保留用于 provenance；
- SOP06 固定 `independent_partial_pairs_v1`：collision 母事件可独立发布，五类
  negative 分别尝试与保留，一类失败不使其他变体失效；
- 所有带目标的 SOP06 variant（包括重建的 temporal-safe）必须保持母事件
  同一 `source_recording_id` + `source_session_id` lineage；禁止只比对 recording
  或从其他 session 替换 snippet；
- 每组输出固定六位 coverage mask 和逐位稳定 missing reason；训练不要求额外
  contrast，完整六位只是 conditional audit 资格；
- 正式 v5 必须拒绝历史 `joint_environment_pair_v2`，禁止以联合多 LOS 搜索作为
  训练发布门槛；
- 渲染历史可见 BEV、不可观测 mask、last-seen 和 age map；
- 输出完整 `OracleWorld`。

## 7.2 六类事件

六个名称定义固定的 coverage 位，不定义生产 quota。`collision` 是母事件；其他位
从同一 base/trajectory/occluder/target lineage 独立构造，失败时保留稳定缺失原因。

```text
collision
near_miss
temporal_safe
spatial_safe
irrelevant_hidden
empty_blind_spot
```

下游对已成功变体做平衡重采样时可参考：

```yaml
collision: 0.20
near_miss: 0.20
temporal_safe: 0.20
spatial_safe: 0.15
irrelevant_hidden: 0.15
empty_blind_spot: 0.10
```

该比例不得用作 producer 的缺类失败条件，平衡采样必须另存 manifest，并报告重采样前
的原始 coverage。校准和测试使用更自然、低碰撞的先验分布。

## 7.3 关键几何约束

- 遮挡物不能和机器人候选轨迹扫掠体碰撞；
- 当前机器人到隐藏目标动态对象的视线必须被遮挡或落在结构性盲区；
- 所有动态对象的真实 circle/rectangle footprint 轨迹不能穿墙；
- 对象速度/加速度必须满足所属 type 的配置范围；
- 不同成对样本只改变必要因素，如时间偏移或横向距离；
- empty 只移除 `target_dynamic_object_id`，其他动态对象保持不变；
- 主 paired 样本中非目标动态对象不得形成 collision/near miss；否则拒绝或标记为
  `multi_object_context`，仅进入自然/OOD 分析；
- 事件必须有固定 `pair_group_id`。

## 7.4 理想结果

- 在固定 proposal budget 下，environment collision 母事件生成率稳定，并报告各候选阶段
  的接受率与拒绝原因；
- 无效几何比例 < 1%；
- `collision` 样本 100% 真碰撞；
- `temporal_safe` 空间路径相交但时序不碰撞；
- `irrelevant_hidden` 中确有隐藏目标动态对象，但与轨迹无关；
- 同一 pair 的场景几何保持一致；
- 每类 negative 的 coverage 和 missing-reason 分布可追溯；条件完整组率只作审计，
  不作为训练数据发布率；
- 人工可视化 100 个样本无明显穿墙/瞬移。

## 7.5 最低验收

- 在明确的固定 proposal budget 下能稳定发布物理合法的 collision 母事件；
- 各 negative 独立尝试，所有成功变体都保留，所有缺位都有稳定原因；
- 5,000 个可用 base events；
- 主要几何单元测试通过。

上述 5,000 是目标规模，必须由实际发布 manifest 验证；本文的接口更新不代表
已完成该规模运行。

## 7.6 降级方案

若复杂遮挡摆放接受率低：

1. 在不变更 `blind_reachability_quota_first_v3` 物理契约的前提下，限制为单矩形货架/柱子；
2. 优先直线/小弧线候选轨迹和侧向 human target，但仍保留稳定随机化与失败报告；
3. 复杂静态地图作为单独 stress/OOD 层，不与基础自由空间产物混合计数；
4. structural/mixed 只能通过未来显式版本升级加入，不得在当前 formal
   environment producer 中静默切换。

---

# 8. W3：候选轨迹、扫掠体和风险 GT

## 8.1 任务

- 差速模型 rollout；
- 当前局部原点 `q0` 只作为积分种子；正式数组 `poses[0:15]=q1...q15`，零基 index
  `k` 对应 `(k+1)*dt`，覆盖 `0.2...3.0 s`；
- 候选库绑定 `future_endpoints_dt_to_horizon_v1`，拒绝旧 `q0...q14 / 0.0...2.8 s`；
- 过滤静态碰撞和动力学不合理轨迹；
- 生成 swept mask、TTA、braking map；
- 计算 collision、near miss、min clearance、TTC、连续风险严重度；
- 主标签只统计由调用方显式声明、且经 renderer 验证当前不可见的对象；
  world 中的可见或未声明对象不得改变 hidden-risk 标签；
- 从 `dynamic_object_specs` 构造 circle/rectangle footprint，并记录
  `critical_object_id/type`；
- 启动时把 BEV 物理宽高对角线解析为有限 `no_object_clearance_sentinel_m` 并写入
  manifest；无相关对象的行使用该值且从 clearance 分布聚合排除，但仍计入
  collision/severity 指标；
- 通过 history-only renderer 构建 `RiskSample` 输入，再在隔离的 label 分支读取
  `OracleWorld` future；metadata 禁止携带 future/oracle 数组或 hidden-object-ID 集合；
- 将 `RiskSample` 按 `sample_id` 稳定排序写入 `risk_shard_npz_jsonl_v2`
  immutable NPZ+JSONL shard；每个目录
  精确包含 `samples.npz`、`metadata.jsonl`、`summary.json`，固定 expected count 和
  单 split，完整 loader 重读与 digest/leakage 验证后才原子发布，且不覆盖已有目录；
  v1 shard 一律 fail closed。
- 正式 `scripts/04_generate_risk_dataset.py` 对一个 formal
  `independent_partial_pairs_v1` group 中所有已成功 variants 做原子组装/发布；
  缺位保留 coverage + stable reason，不要求每个训练 group 凑齐 six-pack。
  只有 conditional complete audit 路径要求六位齐全；CLI 的 exact sample count
  是 shard 总边界，不是每组六类 quota。
- `split_audit_records` 对 base/source recording 跨 split 泄漏执行 fail-closed
  校验，并完整报告允许的 THÖR session overlap；当前 CLI 只组装本 shard，
  collection 调用方仍必须传入全部 split 记录完成 global cross-shard audit。

当前已实现 schema 3 hidden-risk GT、`RiskSample` 组装、deterministic immutable
shard API 与正式数据集 CLI，但只有 unit/toy-fixture 和确定性验证。
10–100 真实样本 smoke、global cross-shard audit 以及目标规模运行仍待执行。

## 8.2 候选轨迹

第一版：

```yaml
linear_velocities: [0.0, 0.2, 0.4, 0.6, 0.8]
angular_velocities: [-0.8, -0.4, 0.0, 0.4, 0.8]
horizon_s: 3.0
dt: 0.2
```

正式候选库冻结为 `trajectory_bank_version=sop04_audited_bank_v2`、
`pose_time_layout_version=future_endpoints_dt_to_horizon_v1`，首末 pose 时间分别为
`0.2 s` 与 `3.0 s`。v2 audit 必须通过 future-endpoint kinematics、query-map、
shape/dtype/finite 和 serial/parallel exact-match，并以目录外 external handoff digest
交给 W2/SOP05；v1 或 `poses[0]=q0` 的库立即拒绝。

可加入少量后退 stress test，但不作为主训练分布。

## 8.3 GT

二值碰撞：

\[
Y_{coll}=\mathbf1[\exists j,\tau:B_r(q_\tau)\cap A_j(t+\tau)\neq\emptyset]
\]

连续严重度：

\[
Y_{risk}=\max_{j,\tau}
\exp\left(-\frac{\max(d_{j,\tau},0)}{\sigma_d}\right)
\exp\left(-\frac{\tau}{\sigma_t}\right)
\]

碰撞时直接设为 1。它是 oracle-defined severity，不声称为真实概率。

## 8.4 理想结果

以下数量是后续实验目标，不是当前完成声明：

- 所有 collision 标签与多边形求交完全一致；
- 连续严重度随 clearance 减小单调增加；
- 同 clearance 下，越早发生危险，严重度越高；
- near-miss 只包含未碰撞且 clearance 小于阈值的样本；
- 有效轨迹比例 ≥ 70%；
- 风险样本 ≥ 240,000；
- 训练集 collision/near-miss/hard negative 分布可控。

## 8.5 最低验收

以下 50,000 门槛须以 immutable shard summaries 实际汇总验证；接口/单元测试
通过不等于达到该规模。

- 风险样本 ≥ 50,000；
- 二值标签、clearance 和 TTC 单元测试通过；
- circle-circle、circle-rectangle、rectangle-rectangle 和多对象最小 clearance
  手算测试通过；
- 人工查看 100 个标签无明显错误。

## 8.6 降级方案

如果连续严重度引发不稳定：

- 主结果先用二值 collision + near-miss 三分类；
- 连续风险作为辅助回归；
- 分位数模型可先预测二值损失或分桶风险等级。

---

# 9. W4：占据/轨迹预测 baseline

## 9.1 目标

提供公平的两阶段 baseline：

```text
历史 BEV/可见动态对象历史
    ↓
occupancy 或动态对象未来轨迹预测
    ↓
栅格化成 p_occ(x, τ)
    ↓
手工概率聚合 over swept volume
    ↓
trajectory risk
```

## 9.2 实现优先级

### 必做 baseline

轻量 ConvGRU occupancy：

```text
past BEV occupancy → future occupancy probability
```

### 可选复用

- Social-STGCNN；
- Trajectron++；
- SOGMP-style stochastic occupancy predictor。

时间紧时，不要让第三方代码兼容问题阻塞主方法；轻量 ConvGRU 足以作为可控 baseline。

## 9.3 风险聚合

至少实现两种：

```text
sum/weighted sum
probabilistic union: 1 - product(1 - p_occ)
```

并对聚合后的 trajectory risk 做相同校准，保证公平。

## 9.4 理想结果

- occupancy baseline 能学到合理的 future occupancy；
- 轨迹风险 baseline 的 AUROC 明显高于 last-observation 和 age-decay；
- 即使对 baseline 做风险校准，主方法仍能降低 false-safe；
- 可生成直观的 occupancy 预测案例图。

## 9.5 最低验收

- 一个轻量 occupancy baseline 可训练；
- 有 last-observation、age-decay、occupancy+aggregation 三类风险 baseline；
- 所有 baseline 使用同一 train/calibration/test split。

## 9.6 降级方案

若 Social-STGCNN/Trajectron++ 复现耗时：

- 将其留在相关工作；
- 主实验只使用 ConvGRU occupancy + hand aggregation；
- 不影响主贡献成立。

---

# 10. W5：轨迹条件风险模型与校准

## 10.1 模型层级

按以下顺序推进，不要一开始直接做复杂模型：

```text
R0: risk-only 小 CNN
R1: temporal CNN / ConvGRU + trajectory channels
R2: trajectory-query cross-attention
R3: R2 + occupancy auxiliary head（可选）
```

Day 8 前根据验证集结果只保留一个主模型。

## 10.2 训练目标

```text
quantile pinball loss
+ collision BCE
+ optional occupancy auxiliary loss
```

## 10.3 校准

- calibration group 完全独立；
- 计算 Q90/Q95 的 one-sided conformal residual；
- 报告全局和分组 coverage；
- 分组维度包括 blind type、critical area、age 和人群密度。

## 10.4 理想结果（目标值，不是保证）

在 held-out 半合成测试上：

```text
Collision AUROC           ≥ 0.90
Brier Score               ≤ 0.10
校准后 ECE                ≤ 0.05
90% risk coverage         88%—92%
False-safe rate           比 occupancy baseline 相对降低 ≥ 30%
Risk upper-bound width    明显小于 always-conservative baseline
```

## 10.5 最低验收

```text
Collision AUROC           ≥ 0.80
90% coverage              85%—95%
False-safe rate           比 occupancy baseline 相对降低 ≥ 10%—15%
至少一个 hard-negative 子集上优于 occupancy aggregation
```

## 10.6 失败诊断

若模型只学到“盲区面积越大风险越高”：

- 增加同面积、不同动态事件的配对样本；
- 强化 temporal-safe 和 irrelevant-hidden hard negatives；
- 做去掉历史/age/trajectory 的消融；
- 检查候选轨迹 query 是否实际输入模型。

若 risk-only 不优于 occupancy baseline：

- 先检查 GT 和数据配对；
- 改为二值 collision 主任务；
- 提升 trajectory conditioning；
- 不急于引入更大 backbone。

---

# 11. W6：Scenario Bank 与反事实验证价值 GT

## 11.1 任务

- 从真实/半合成事件生成 M 个与当前可见观测一致的隐藏世界；
- 实现验证 primitive；
- 模拟验证后可见区域；
- 更新 observation signature；
- 验证后围绕原 nominal intent 重新采样候选集；
- 计算 scenario-bank empirical decision risk；
- 输出已经扣除验证成本的净价值 `G*`。

## 11.2 重要一致性

1. `G*` 中验证成本只扣一次；
2. verify 后必须重新规划，不只评估原始轨迹；
3. 论文称其为 simulator-defined target，不称严格 Bayes GT；
4. 测试输入不能包含验证后 oracle occupancy；
5. scenario bank 的 train/val/test seed 和 snippet 来源隔离。

## 11.3 两级实现

### Level 1：精确离散 posterior（优先用于 toy 和小 bank）

使用验证观测可见 occupancy 的离散签名，将产生相同观测的 world 分组。

### Level 2：soft signature posterior

在大规模生成时使用标准化 signature 和 softmax 距离权重。

## 11.4 理想结果

- toy world 的 G* 与人工枚举完全一致；
- 高价值验证动作确实暴露关键扫掠区域；
- 无关大视野动作可能信息增益高但 G* 低；
- 增大验证成本时 G* 单调下降；
- 完美揭示关键动态对象的动作，其 post-risk 显著下降；
- scenario bank 大小从 8→16→32 时，动作排序趋于稳定；
- verification samples ≥ 60,000，推荐 ≥ 200,000。

## 11.5 最低验收

- 6 个验证 primitive；
- M=8 或 16 的 scenario bank；
- 10,000 个可用 verification samples；
- toy 单元测试和成本单调性测试通过；
- 至少有 20% 正价值、20% 负价值样本。

## 11.6 降级方案

若 soft posterior 对 `tau_o` 很敏感：

1. 主实验使用 exact/discrete grouping；
2. 或使用 realized decision gain 作为简化目标；
3. 把 soft posterior 作为扩展/消融；
4. 必须报告 M、tau 和 bank composition 敏感性。

---

# 12. W7：验证价值网络

## 12.1 输入输出

输入：

```text
z
nominal trajectory ξ
verification geometry/FOV mask
verification action token
verification duration/distance/yaw
```

输出：

```text
G_pred：净验证价值
P_useful：G*>0 的概率
```

## 12.2 模型顺序

```text
V0: CNN concat baseline
V1: shared/frozen risk BEV encoder + action encoder
V2: trajectory/action query attention（可选）
```

先冻结风险 encoder 可降低联合训练不稳定性。

## 12.3 损失

```text
Huber value regression
+ useful BCE
+ pairwise ranking
```

## 12.4 理想结果（目标值）

```text
Useful-action F1             ≥ 0.80
Pairwise ranking accuracy    ≥ 0.80
Spearman correlation         ≥ 0.75
Top-1 regret                 ≤ oracle value range 的 10%
选中最优或次优动作比例       ≥ 85%
```

## 12.5 最低验收

```text
Useful-action F1             ≥ 0.65
Pairwise ranking accuracy    ≥ 0.65
Top-1 regret                 优于 visible-area / swept-coverage baseline
闭环 false-safe 优于 never-verify
```

## 12.6 失败诊断

若模型只偏好成本最低的动作：

- 对不同动作成本做配对样本；
- 把 gross information gain 和 action cost 分别作为辅助输入/监督；
- 检查 `G*` 是否重复扣成本。

若模型只偏好可见面积最大的动作：

- 增加“看很多无关区域”的负样本；
- 加入 `new visible ∩ swept volume` 相近但决策价值不同的配对；
- 强化 ranking loss。

---

# 13. W8：离线闭环、基线、消融和论文图表

## 13.1 离线闭环优先

不等待 Arena 完整接入，先实现轻量 2D closed loop：

```text
当前 world / state
    ↓
生成候选轨迹
    ↓
risk model + calibration
    ↓
value model
    ↓
execute / verify / reject
    ↓
执行短步 / 更新可见性 / 重新规划
```

## 13.2 必做策略

```text
Never Verify
Always Verify
Visible Area Gain
Swept-Coverage Gain
Occupancy Entropy Gain
Learned Decision Value
Oracle Verification Value（上界）
```

## 13.3 必做指标

```text
Collision Rate
False-safe Execution Rate
Near-miss Rate
Verification Count
Unnecessary Verification Rate
Reject Rate
Task Success Rate
Completion Time
Extra Path Length
```

## 13.4 理想闭环结果（目标值）

与 Never Verify 相比：

```text
collision / false-safe 相对降低 ≥ 30%
```

与 Always Verify 相比：

```text
验证次数 ≤ 40%—60%
任务成功率接近，完成时间更短
```

与 Visible Area / Entropy Gain 相比：

```text
相同验证预算下 false-safe 更低
或相同安全水平下验证次数更少
```

## 13.5 最低验收

- learned value 至少在一个固定验证预算下优于 visible-area；
- risk calibration 明显降低 false-safe；
- 有一张有效的 safety-efficiency Pareto 图；
- 有 5—10 个可视化成功/失败案例。

## 13.6 失败降级

若完整闭环耗时：

- 使用 episode-level offline replay；
- 每个 state 只做一次 execute/verify/reject 决策；
- 报告 oracle next-state 更新；
- Arena 作为 supplement 或 future work。

---

# 14. 两周日程与并行波次

以下按 2026-07-14 启动、2026-07-28 截止设计。

## Wave 0：接口冻结与解除阻塞（7/14）

### A0

- 建仓库、schema、config、toy fixture；
- 建 `STATUS.md`、`DECISIONS.md`；
- 预建论文目录和实验结果表模板。

### A1/A2/A3/A4/A5/A6

- 所有 Agent 阅读 schema；
- 用 toy fixture 跑通各自最小脚本；
- 不等待真实数据。

### 当日理想结果

```text
pytest 通过
RiskModel forward 通过
ValueModel forward 通过
toy G* 与人工结果一致
所有 Agent 可独立开发
```

### 当日最低结果

- schema 和目录冻结；
- toy risk/value sample 可读取。

---

## Wave 1：数据与模型原型并跑（7/15—7/17）

### A1：数据

- 解析 THÖR 机器人与所有非机器人 dynamic-object 轨迹/spec；
- 完成 group split；
- 提取第一批 snippets 和 base states。

### A2：场景

- 在 toy/base state 上实现结构性 blind sector；
- 实现单矩形遮挡；
- 实现 BEV visibility、age map 和 paired events。

### A3：轨迹/GT

- 实现差速 rollout、扫掠体、风险 GT；
- 完成所有单元测试。

### A4：风险模型

- 用 toy/随机数据开发 risk-only CNN；
- 开发 pinball+BCE、数据加载、评估；
- 实现 last observation、age decay、occupancy ConvGRU 骨架。

### A5：验证价值

- 用 toy worlds 实现 verify primitives、反事实 FOV、exact posterior、重新规划和 G*；
- 开发 value model 骨架。

### A6：评估/论文

- 开发所有指标和 plotting API；
- 写 Problem Formulation、Method Overview、Data Generation 草稿；
- 先画占位系统图。

### 7/17 理想结果

```text
base states ≥ 2,000
snippets ≥ 1,000
半合成事件 ≥ 5,000
risk samples ≥ 20,000
risk model 能过拟合 1,000 样本
G* toy tests 全部通过
```

### Go/No-Go 1

若 7/17 尚未稳定解析 THÖR：

- 只用官方轨迹 CSV；
- 所有遮挡和静态地图程序化；
- 禁止继续投入 RGB/点云解析。

---

## Wave 2：第一版真实实验闭环（7/18—7/20）

### 数据线

- 扩展到 5k base states；
- 生成 50k—100k risk samples；
- 验证 paired event 分布和无效率。

### 风险线

- 训练 occupancy baseline；
- 训练 risk-only CNN；
- 完成第一版 calibration；
- 产出 AUROC、Brier、ECE、coverage 和 false-safe。

### 验证线

- 用真实半合成 world 构造 M=8/16 scenario bank；
- 生成 10k—30k verification samples；
- 训练第一版 value model。

### 评估线

- 用 oracle/model stub 跑离线 execute/verify/reject；
- 完成 Never/Always/Visible/Swept baselines。

### 7/20 理想结果

```text
Risk AUROC ≥ 0.80
90% coverage 在 85%—95%
Value useful F1 ≥ 0.60
Learned verify 至少优于随机选择
第一张 calibration curve
第一张 safety-efficiency 曲线
```

### Go/No-Go 2

如果风险模型仍不能超过 occupancy baseline：

1. 冻结复杂模型开发；
2. 检查标签、hard negatives 和 split；
3. 使用 binary collision 主任务；
4. 只保留 risk-only CNN + 清晰数据贡献。

如果 value model 不学习：

1. 检查 G* 成本是否重复；
2. 用 exact posterior；
3. 临时使用 realized decision gain；
4. 将 value learning 缩为动作排序任务。

---

## Wave 3：主模型选择与全量生成（7/21—7/23）

### A1/A2/A3

- 生成论文版 risk dataset：推荐 240k+；
- 生成 verification dataset：推荐 60k—200k；
- 固定最终 train/calibration/val/test manifest；
- 之后不再改数据定义。

### A4

并行训练：

```text
occupancy + hand aggregation
risk-only
risk + temporal encoder
risk + trajectory query
optional risk + occupancy auxiliary
```

7/23 前只保留最佳主模型和必要消融。

### A5

- 训练 value regression/classification/ranking；
- 做 M、tau、action cost 小规模敏感性；
- 完成 visible-area、swept-coverage、entropy baselines。

### A6

- 写 Related Work 和 Experiments 设置；
- 自动生成表格；
- 建立 final run checklist。

### 7/23 理想结果

```text
主风险模型相对 occupancy baseline 的 false-safe 降低 ≥ 20%
校准 coverage 接近目标
Value F1 ≥ 0.70
Pairwise ranking ≥ 0.70
Learned verification 在离线闭环优于 Visible/Swept baseline
```

### 架构冻结

7/23 之后：

- 不再新增 backbone；
- 不再改 GT 公式；
- 只修 bug、跑消融和补结果。

---

## Wave 4：最终实验、消融与图表（7/24—7/26）

### 必跑实验矩阵

#### 风险

```text
Last observation
Age decay
Occupancy + hand aggregation
Risk-only
Risk + calibration
Risk + optional occupancy aux
```

#### 验证

```text
Never
Always
Visible area
Swept coverage
Entropy
Learned value
Oracle value
```

#### 消融

```text
without age map
without history
without trajectory query
without calibration
without ranking loss
same-area paired hard negatives
scenario bank M sensitivity
verification cost sensitivity
```

### 7/26 理想结果

- 所有主表平均至少 3 个随机种子；
- 主方法在核心指标上有一致改进；
- 产生完整主图；
- 有失败案例和 limitation；
- 代码可从空目录生成小规模结果。

### 最低结果

- 主结果 1—3 个种子；
- 至少一套完整 ablation；
- safety-efficiency Pareto 成立；
- 所有数字可追溯到日志。

---

## Wave 5：写作、复核和提交（7/27—7/28）

### 论文冻结顺序

1. 问题定义与贡献；
2. 数据生成和 GT；
3. 风险模型；
4. 验证价值；
5. 实验和结果；
6. limitations；
7. supplementary 和代码说明。

### 提交前硬检查

```text
G* 成本只扣一次
verify 后重新规划
scenario bank 不称严格 Bayes GT
continuous risk 不称真实概率
train/cal/test group split 无泄漏
模型测试输入不包含 oracle 信息
所有 baseline 使用相同数据和校准集
```

---

# 15. 关键路径与非关键路径

## 15.1 真正关键路径

```text
数据轨迹解析
→ 事件生成
→ 风险 GT
→ 风险模型
→ 风险校准
→ 离线闭环主结果
```

验证价值学习是第二条关键路径：

```text
事件生成
→ scenario bank
→ G*
→ value model
→ verify baseline 对比
```

## 15.2 不得阻塞关键路径的工作

- Social-STGCNN/Trajectron++ 第三方复现；
- Arena ROS2 集成；
- JRDB 泛化；
- 占据辅助 head；
- 大模型或复杂 attention；
- 真实机器人部署。

若这些工作在 7/23 前未跑通，立即移入 stretch goal。

---

# 16. Agent 之间的交付契约

## 16.1 每个 Agent 提交时必须包含

```text
代码
最小配置
一条可执行命令
单元测试
10—100 个 fixture 输出
README 中的输入/输出说明
预期运行时间和内存
```

## 16.2 PR 不能修改的内容

除 A0 外，其他 Agent 不得自行修改：

```text
核心 dataclass 字段
split 定义
坐标系定义
BEV 尺寸
风险 GT 公式
G* 是否包含验证成本
```

需要改动时必须写入 `DECISIONS.md`。

## 16.3 每日同步模板

```markdown
## YYYY-MM-DD / Agent-X
- Completed:
- Produced artifacts:
- Metrics:
- Blockers:
- Interface changes requested:
- Next 24h:
```

---

# 17. 推荐 CLI 与产物

本节的命令是目标编排界面，不是真实数据规模完成声明。截至本次契约对齐，
schema 3 risk GT、`RiskSample`、单个 immutable shard API 和正式
`scripts/04_generate_risk_dataset.py` CLI 已实现并有 unit/toy-fixture 验证；
10–100 真实样本 smoke、global cross-shard audit、shard collection manifest 及
目标规模运行仍待实现或验证。

## 17.1 数据

```bash
python -m scripts.prepare_splits \
  --config configs/thor.yaml

python -m scripts.extract_snippets \
  --split train \
  --config configs/thor.yaml

# 正式单 shard CLI；下列 digest 必须来自可信目录外 handoff
python scripts/04_generate_risk_dataset.py \
  --sop03-root <schema3-sop03-root> \
  --sop04-root <schema3-sop04-root> \
  --sop04-handoff-digest <64-hex> \
  --sop05-root <schema3-sop05-root> \
  --sop05-publication-digest <64-hex> \
  --split train \
  --config configs/base.yaml \
  --paired-config configs/paired_variants.yaml \
  --seed 42 \
  --output-dir outputs/event_centered_blind_spot/schema-v3/risk-data/main-seed42-v1/train/shard-00000 \
  --shard-index <nonnegative-int> \
  --expected-event-count <positive-int> \
  --expected-sample-count <positive-int> \
  --checksum-workers 8
```

预期产物：

```text
outputs/splits/split_manifest.parquet
outputs/snippets/<split>/<object_type>/...
outputs/event_centered_blind_spot/schema-v3/risk-data/<run-id>/<split>/shard-*/samples.npz
outputs/event_centered_blind_spot/schema-v3/risk-data/<run-id>/<split>/shard-*/metadata.jsonl
outputs/event_centered_blind_spot/schema-v3/risk-data/<run-id>/<split>/shard-*/summary.json
```

## 17.2 风险

```bash
python -m training.train_occupancy_baseline \
  --config configs/occupancy_baseline.yaml

python -m training.train_risk \
  --config configs/risk_model.yaml

python -m training.calibrate_risk \
  --checkpoint outputs/event_centered_blind_spot/schema-v3/risk-model/main-seed42-v1/best.pt \
  --split calibration
```

预期产物：

```text
outputs/event_centered_blind_spot/schema-v3/risk-model/<run-id>/best.pt
outputs/event_centered_blind_spot/schema-v3/risk-model/<run-id>/metrics-val.json
outputs/event_centered_blind_spot/schema-v3/risk-model/<run-id>/calibration/calibration.json
outputs/event_centered_blind_spot/schema-v3/reports/<run-id>/reliability-curve.png
outputs/event_centered_blind_spot/schema-v3/reports/<run-id>/risk-examples/
```

## 17.3 验证价值

```bash
# planned collection-level interface; exact collection manifest is not frozen yet
python -m scripts.build_verification_dataset \
  --risk-shard-root outputs/event_centered_blind_spot/schema-v3/risk-data/main-seed42-v1/train/ \
  --scenario-bank-size 16 \
  --config configs/verification_gt.yaml

python -m training.train_verify \
  --config configs/value_model.yaml
```

预期产物：

```text
outputs/event_centered_blind_spot/schema-v3/verification-data/<run-id>/<split>/shard_*.npz
outputs/event_centered_blind_spot/schema-v3/verification-model/<run-id>/best.pt
outputs/event_centered_blind_spot/schema-v3/verification-model/<run-id>/metrics.json
outputs/event_centered_blind_spot/schema-v3/reports/<run-id>/action-ranking-examples/
```

## 17.4 闭环

```bash
python -m evaluation.eval_closed_loop \
  --risk-checkpoint outputs/event_centered_blind_spot/schema-v3/risk-model/main-seed42-v1/best.pt \
  --value-checkpoint outputs/event_centered_blind_spot/schema-v3/verification-model/main-seed42-v1/best.pt \
  --config configs/closed_loop.yaml

python -m evaluation.plot_pareto \
  --input outputs/event_centered_blind_spot/schema-v3/closed-loop/main-seed42-v1/all_methods.csv
```

---

# 18. 每阶段的结果看板

建议维护 `RESULTS_BOARD.md`：

| 模块 | 指标 | 当前 | 最低线 | 理想线 | 状态 |
|---|---|---:|---:|---:|---|
| 数据 | 有效 base states |  | 2k | 10k |  |
| 数据 | 有效 snippets |  | 1k | 5k |  |
| 生成 | event acceptance |  | 50% | 80% |  |
| 风险 | AUROC |  | 0.80 | 0.90 |  |
| 风险 | 90% coverage |  | 85—95% | 88—92% |  |
| 风险 | false-safe reduction |  | 10% | 30% |  |
| 价值 | Useful F1 |  | 0.65 | 0.80 |  |
| 价值 | Ranking accuracy |  | 0.65 | 0.80 |  |
| 闭环 | collision reduction |  | 15% | 30% |  |
| 闭环 | verify vs always |  | <70% | <50% |  |

这些数值是项目目标，不是承诺。若理想线未达到，应诚实报告，并用消融解释原因。

---

# 19. 风险清单与快速转向规则

## 19.1 THÖR 数据适配拖延

**触发条件**：7/17 无法稳定输出 2k base states。  
**转向**：只用轨迹 CSV；程序化静态地图；取消原始传感器解析。

## 19.2 半合成样本被质疑不真实

**预防**：

- 主分布 human 运动片段来自真实数据，其他动态对象也保留真实轨迹和 footprint；
- 配对反事实样本；
- held-out recording（已知 recording-day session）；
- 参数 OOD 测试；
- 少量自然真实片段 sanity check；
- 报告生成器限制。

## 19.3 风险模型没有超过 occupancy baseline

**触发条件**：7/20 false-safe 无改进。  
**转向**：

- 强化 trajectory query；
- 加 hard-negative 对；
- 先做二值风险；
- 把论文主 claim 缩为 calibration 和 verification value；
- 不继续堆大模型。

## 19.4 G* 不稳定

**触发条件**：动作排序随 M/tau 大幅改变。  
**转向**：

- exact posterior；
- realized decision gain；
- 把 soft posterior 降为 ablation；
- 主结果报告敏感性和限制。

## 19.5 Arena 接入失败

**转向**：

- 使用自建 2D-BEV closed loop；
- Arena 放补充/未来工作；
- 不让 ROS 阻塞论文。

## 19.6 算力不足

**转向**：

```text
160×160 → 96×96 或 128×128
ConvGRU → temporal channel stacking
cross-attention → concatenate + CNN
40 epochs → early stopping
全量 verification → 只对边界风险轨迹生成
```

---

# 20. 最终优先级

## P0：任何情况下必须完成

- 无泄漏 split；
- 半合成事件生成；
- 风险 GT；
- occupancy baseline；
- risk model + calibration；
- verify GT；
- value model；
- 一套离线闭环；
- 主图主表。

## P1：尽量完成

- 多个风险架构消融；
- scenario bank 敏感性；
- Arena 小规模闭环；
- 结构性盲区 stress test；
- 3 seeds。

## P2：有余力再做

- JRDB 跨数据集；
- Social-STGCNN/Trajectron++；
- occupancy auxiliary head；
- 完整 ROS2 插件；
- 真实机器人。

---

# 21. 最终理想实验故事

理想情况下，论文应形成如下证据链：

1. **数据层**：事件中心生成器能从真实人类轨迹构造大量物理可行、成对可控的盲区事件；
2. **风险层**：直接学习 trajectory-conditioned risk，比“occupancy prediction + 手工聚合”更少产生 false-safe；
3. **校准层**：动作级风险上界达到目标 coverage，并显著降低风险低估；
4. **验证层**：学习到的决策价值优于可见面积、扫掠覆盖和熵增益；
5. **闭环层**：在接近 Always Verify 的安全水平下，使用明显更少的验证动作；
6. **泛化层**：在未见地图、遮挡物尺寸、参与者和事件先验下仍保持趋势；
7. **诚实边界**：明确连续风险是 oracle severity、G* 是 simulator-defined target，不主张严格真实概率或无条件安全。

---

# 22. 一句话执行原则

> **所有人并行，但所有模块只通过冻结的数据契约连接；先用 toy fixture 解除依赖，再用真实半合成数据替换；每两到三天设置一次 Go/No-Go，任何第三方复现、复杂模型和 ROS 集成都不得阻塞“风险模型 + 验证价值 + 离线闭环”这一最小论文闭环。**
