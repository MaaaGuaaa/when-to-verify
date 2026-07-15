# 事件中心盲区样本生成、轨迹风险学习与主动验证：Codex 工程实现规格

> 项目暂定名：**Learning When to Verify: Decision-Calibrated Hidden Risk for Robot Local Planning**  
> 中文暂定名：**学习何时验证：面向机器人局部规划的决策校准隐藏风险**  
> 文档目的：把当前研究思路整理成可由 Codex/工程 Agent 直接拆解和实现的技术规格。  
> 推荐第一版范围：**2D BEV、差速轮式机器人、动态行人、程序化遮挡、短时局部规划**。

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

THÖR-MAGNI 等真实数据中，天然“轨迹进入盲区且与隐藏行人冲突”的事件很少。因此不依赖自然事件，而采用：

> **事件中心的半合成样本生成（Event-Centered Semi-Synthetic Generation）**

具体做法：

1. 从真实数据提取机器人状态和真实行人轨迹片段；
2. 从当前机器人状态程序化采样候选局部轨迹；
3. 在候选轨迹上选择潜在冲突点和冲突时刻；
4. 在冲突点附近放置程序化遮挡物，或施加结构性盲区；
5. 对真实行人轨迹片段做刚体变换和小幅时间缩放，使其从盲区出现并在指定时刻与机器人轨迹交互；
6. 同一场景生成碰撞、近碰、时序安全、空间安全、无关隐藏人、空盲区等成对样本；
7. 用完整 oracle 世界自动计算风险 GT 和验证价值 GT。

---

## 1. 明确范围与非目标

### 1.1 第一版必须实现

- 2D/2.5D BEV 表示；
- 差速轮式机器人运动模型；
- 真实行人轨迹片段库；
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
- 提取真实行人轨迹、速度、加速度分布；
- 建立行人 trajectory snippet library；
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

### 3.4 轨迹片段库也必须分 split

训练集中的行人 snippet 不得出现在 test 生成器中。每个 split 单独建立：

```text
snippets/train/
snippets/calibration/
snippets/val/
snippets/test/
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
history_duration = 1.6 s
history_dt       = 0.2 s
history_steps K  = 8
future_horizon   = 3.0 s
future_dt        = 0.2 s
future_steps T   = 15
```

原始轨迹统一用线性插值/角度 unwrap 后插值到上述时间网格。

### 4.3 BEV 网格

推荐：

```text
range_x = [-8m, 8m]
range_y = [-8m, 8m]
resolution = 0.1m
H = W = 160
```

### 4.4 机器人和行人几何

第一版简化：

```text
robot footprint：矩形或圆形
    width  = 0.55m
    length = 0.70m
    safety inflation = 0.15m

pedestrian footprint：圆形
    radius = 0.30m
```

数据集有精确尺寸时可以覆盖默认值。

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
    "pedestrians": {
        pedestrian_id: {
            "timestamps": np.ndarray[M],
            "positions": np.ndarray[M, 2],
            "velocities": np.ndarray[M, 2],
        }
    },
    "static_map": Optional[np.ndarray],
}
```

### 5.2 base state 提取

从每个 recording 每隔 0.5~1.0 秒取一个 base state：

```python
BaseState = {
    "base_id": str,
    "recording_id": str,
    "t0": float,
    "robot_history": np.ndarray[K, 3],
    "robot_state": np.ndarray[D],
    "visible_ped_history": dict,
    "oracle_ped_history": dict,
    "oracle_ped_future": dict,
    "static_map_local": np.ndarray[H, W],
}
```

筛选条件：

- 历史和未来窗口完整；
- 机器人当前状态有效；
- 机器人不在静态障碍内部；
- 局部范围内允许插入程序化遮挡物；
- 初期允许 base state 中没有自然动态体，因为后续会插入移植行人。

---

## 6. 真实行人轨迹片段库

### 6.1 目标

避免用纯随机直线行人。真实行人轨迹片段保留速度变化、轻微弯曲和加速度统计；程序只做 SE(2) 变换和小幅时间缩放。

### 6.2 snippet 提取

从每个行人轨迹中滑窗提取：

```text
snippet_duration = 3.0~5.0 s
snippet_dt = 0.2 s
min_mean_speed = 0.35 m/s
max_mean_speed = 1.8 m/s
max_acceleration = 2.5 m/s²
```

归一化为局部坐标：

```text
snippet 第一个位置为 (0, 0)
初始运动方向对齐 +x
```

结构：

```python
PedSnippet = {
    "snippet_id": str,
    "source_recording_id": str,
    "participant_id": str,
    "positions": np.ndarray[Tp, 2],
    "velocities": np.ndarray[Tp, 2],
    "duration": float,
    "mean_speed": float,
    "max_acc": float,
}
```

### 6.3 允许变换

\[
\tilde h_\tau=R(\phi)h_{a\tau}+t
\]

范围：

```text
rotation φ：任意
translation t：由冲突点决定
time scale a：0.8~1.2
额外位置噪声：最多 0.05m，默认不用
```

### 6.4 变换后有效性检查

- 不穿过静态障碍或新插入遮挡物；
- 速度 0.3~2.0 m/s；
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

随机放墙、随机放人、随机采样轨迹会导致有效冲突事件极低。正确做法是：

> 先确定轨迹上的潜在冲突时刻和位置，再反向构造遮挡物和隐藏行人。

### 8.2 总流程

```text
BaseState z
    ↓
采样候选轨迹 ξ
    ↓
选择冲突时间 τ* 和位置 p*
    ↓
选择事件类型：环境遮挡 / 结构性盲区 / 混合
    ↓
放置遮挡物或施加 FOV 盲区
    ↓
抽取真实 PedSnippet
    ↓
变换 snippet，使行人在 τ* 附近通过 p*
    ↓
生成 paired variants
    ↓
渲染历史观测与 oracle 世界
    ↓
计算风险 GT / 验证价值 GT
```

### 8.3 冲突点选择

从轨迹中选择：

```text
τ* ∈ [1.0s, 2.2s]
```

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

遮挡物中心：

\[
c_o=p^*+d_o\mathbf n^*
\]

参数：

```yaml
occluder_types:
  wall:
    length_range: [1.0, 3.0]
    width_range: [0.2, 0.5]
  shelf:
    length_range: [1.0, 2.5]
    width_range: [0.4, 0.8]
  pillar:
    length_range: [0.4, 0.8]
    width_range: [0.4, 0.8]
normal_offset_range: [0.5, 1.5]
```

硬约束：

1. 遮挡物不得与机器人候选轨迹扫掠体相交；
2. 遮挡物不得与基础静态障碍大面积重叠；
3. 当前机器人到行人历史位置的视线需被遮挡物截断；
4. 行人未来从遮挡边界出现时应在物理上连续；
5. 不允许行人轨迹穿墙。

### 8.5 结构性盲区生成

不放实体障碍，直接定义传感器可见范围：

```yaml
structural_fov:
  forward_fov_deg: [160, 180, 220]
  range_m: [6.0, 8.0, 10.0]
  optional_blind_sectors:
    - center_deg: 150
      width_deg: 60
```

行人必须在当前及部分历史时刻落入 FOV 外，未来与候选轨迹发生交互。

### 8.6 混合盲区

同时存在 FOV 限制和实体遮挡。第一版比例建议：

```text
environment occlusion 60%
structural blind spot 30%
mixed 10%
```

---

## 9. 将真实行人轨迹移植到冲突事件

### 9.1 对齐目标

选择 snippet 中某个索引 \(k^*\)，要求变换后：

\[
\tilde h_{k^*}\approx p^*
\]

并让该时刻对应：

\[
t_{k^*}\approx \tau^*
\]

### 9.2 横穿方向

初始目标方向通常接近轨迹法向：

\[
\angle(\dot h_{k^*},\mathbf n^*)<\theta_{max}
\]

默认 \(\theta_{max}=35^\circ\)。

也可以生成斜穿/同向交汇事件，但第一版以侧向横穿为主。

### 9.3 求变换

1. 对 snippet 做时间缩放；
2. 将 \(h_{k^*}\) 平移到原点；
3. 将 \(\dot h_{k^*}\) 旋转到目标横穿方向；
4. 平移到 \(p^*\)；
5. 检查完整历史/未来轨迹。

### 9.4 失败重采样

以下任一失败时重新选 snippet/遮挡物参数：

- 当前时刻行人可见；
- 过去历史中行人已被看到但不符合目标样本类型；
- 行人穿墙；
- 行人与机器人当前重叠；
- 未来轨迹超出范围过多；
- 速度/加速度超限；
- 事件无法形成指定 collision/near-miss/safe 版本。

---

## 10. 配对反事实样本生成

对同一 BaseState、候选轨迹、遮挡物和行人 snippet，生成多个只改变关键变量的版本。

### 10.1 样本类型

#### A. Collision

通过时间偏移/平移，使 footprint 在某一未来时刻相交。

#### B. Near miss

保持空间路径近似相交，但最小安全距离：

```text
0.05m ~ 0.35m
```

#### C. Temporal-safe hard negative

空间路径相交，但时间错开：

```text
±0.8s ~ ±1.5s
```

#### D. Spatial-safe hard negative

时间接近，但行人横向偏移，使最小距离：

```text
0.5m ~ 1.0m
```

#### E. Irrelevant hidden pedestrian

盲区里有人，但完全不接近当前候选轨迹。

#### F. Empty blind spot

同一场景移除行人。

### 10.2 训练比例建议

```yaml
collision: 0.20
near_miss: 0.20
temporal_safe: 0.20
spatial_safe: 0.15
irrelevant_hidden: 0.15
empty: 0.10
```

训练集可以平衡；calibration/test 应额外生成更自然的低碰撞先验，例如：

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
ped_snippet_id
```

便于做 paired evaluation 和防止生成器捷径。

---

## 11. 历史观测与 BEV 信念渲染

### 11.1 模型可见输入与 oracle 世界必须分离

模型输入只能由模拟传感器可见内容生成；完整隐藏行人轨迹只用于标签。

### 11.2 每个历史时刻渲染

对于 \(t-K+1,\ldots,t\)：

1. 读取真实机器人历史位姿；
2. 应用同一个程序化遮挡物/FOV 模型；
3. 放置变换后的行人历史位置；
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

### 12.5 风险标签伪代码

```python
def compute_risk_gt(robot_traj, hidden_ped_trajs, cfg):
    collision = False
    min_clearance = float("inf")
    first_collision_time = None
    max_severity = 0.0

    for k, robot_pose in enumerate(robot_traj.poses):
        tau = k * cfg.future_dt
        robot_shape = inflated_robot_shape(robot_pose, cfg.robot_inflation)

        for ped_traj in hidden_ped_trajs:
            ped_shape = pedestrian_shape(ped_traj.position_at(k), cfg.ped_radius)
            clearance = shape_distance(robot_shape, ped_shape)
            min_clearance = min(min_clearance, clearance)

            if shapes_intersect(robot_shape, ped_shape):
                collision = True
                if first_collision_time is None:
                    first_collision_time = tau
                severity = 1.0
            else:
                severity = math.exp(-max(clearance, 0.0) / cfg.sigma_distance)
                severity *= math.exp(-tau / cfg.sigma_time)

            max_severity = max(max_severity, severity)

    return {
        "collision": int(collision),
        "risk_severity": float(max_severity),
        "min_clearance": float(min_clearance),
        "first_collision_time": first_collision_time,
        "near_miss": int((not collision) and min_clearance < cfg.near_miss_distance),
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
source_participant_id
ped_snippet_id
trajectory primitive (v, omega)
occluder type / size / pose
blind spot type
random seed
```

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
- pedestrian density。

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
2 个无关隐藏行人变体
```

每个世界必须和当前模型输入一致：

- 当前 visible cells 的 occupancy 一致；
- 差异仅位于不可观测区域或未来状态；
- 不违反静态几何；
- 不产生当前时刻直接可见的新增行人。

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

### 23.1 最小可行版

```text
base states: 5,000
候选轨迹/base: 12
paired event variants/trajectory: 4
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

相同几何只改变时间偏移/横向偏移/是否存在行人。

### 24.2 随机化

- 遮挡物尺寸、朝向、类型；
- FOV 角度和 range；
- 行人速度和时间偏移；
- 轨迹 primitive；
- event time；
- background static map。

### 24.3 split 隔离

- recording 隔离；
- participant 隔离；
- snippet 隔离；
- generator seed 隔离；
- test 使用部分未见参数范围。

### 24.4 关键 controlled tests

1. 同样盲区面积，不同隐藏人位置；
2. 同样路径空间交叉，不同到达时间；
3. 同样行人轨迹，不同候选机器人轨迹；
4. 同样验证可见面积，但对关键扫掠区域覆盖不同；
5. 空盲区与有无关行人的对照。

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
│   │   ├── pedestrian_transplant.py
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
    ├── test_pedestrian_transplant.py
    ├── test_pair_variants.py
    ├── test_risk_gt.py
    └── test_verification_gt.py
```

---

## 29. 推荐配置文件示例

```yaml
seed: 42

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

pedestrian:
  radius_m: 0.30
  min_speed: 0.3
  max_speed: 2.0
  max_acceleration: 2.5

trajectories:
  linear_velocities: [0.2, 0.4, 0.6, 0.8]
  angular_velocities: [-0.8, -0.4, 0.0, 0.4, 0.8]
  reverse_velocities: [-0.2, -0.4]
  reverse_probability: 0.2
  horizon_s: 3.0
  dt_s: 0.2

blind_spots:
  environment_ratio: 0.6
  structural_ratio: 0.3
  mixed_ratio: 0.1
  fov_options_deg: [160, 180, 220]
  sensor_range_options_m: [6.0, 8.0, 10.0]

occluders:
  normal_offset_range_m: [0.5, 1.5]
  max_sampling_trials: 100

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

```bash
# 1. 按 recording / participant 切分
python scripts/00_make_splits.py --config configs/data_thor.yaml

# 2. 解析原始记录
python scripts/01_index_recordings.py --split train
python scripts/01_index_recordings.py --split calibration
python scripts/01_index_recordings.py --split val
python scripts/01_index_recordings.py --split test

# 3. 每个 split 单独建立行人轨迹片段库
python scripts/02_build_snippet_library.py --split train
python scripts/02_build_snippet_library.py --split calibration
python scripts/02_build_snippet_library.py --split val
python scripts/02_build_snippet_library.py --split test

# 4. 提取 base states
python scripts/03_extract_base_states.py --all-splits

# 5. 生成事件中心风险数据
python scripts/04_generate_risk_dataset.py \
  --config configs/generator_train.yaml \
  --split train

# 6. 训练 occupancy baseline（可并行/可后做）
python scripts/05_train_occupancy_baseline.py --config configs/risk_model.yaml

# 7. 训练主风险模型
python scripts/06_train_risk_model.py --config configs/risk_model.yaml

# 8. 独立 calibration groups 做校准
python scripts/07_calibrate_risk.py --checkpoint outputs/risk/best.ckpt

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
- 行人当前不可见，未来可从边缘出现。

### 31.3 配对样本测试

同一 pair group：

- collision variant 的 min clearance ≤ 0；
- temporal-safe 空间路径相交但时间不相交；
- empty variant 没有行人；
- 除指定变量外几何相同。

### 31.4 风险 GT 测试

- 碰撞时 `collision=1` 且 `risk_severity=1`；
- 距离越近风险不降低；
- 同距离下越早出现风险越高；
- visible actor 不计入 hidden-risk 主标签。

### 31.5 验证价值测试

构造人工小场景：

- 一个动作正好看到冲突行人，价值应高；
- 一个动作只看到无关区域，价值应低/负；
- 增大验证成本后价值应下降；
- 空盲区场景验证通常不应有正价值；
- `PostRisk` 已含 `c(v)`，在线不可重复加成本；
- verify 后候选集必须从新位姿重生成。

### 31.6 最小工程验收

```text
✓ 能从一个 recording 生成 base states
✓ 能建立独立 train/test snippet libraries
✓ 能从一条直线候选轨迹生成墙后横穿行人事件
✓ 能生成六类 paired variants
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

1. 解析 THÖR robot/ped trajectories；
2. 坐标转换和 resampling；
3. 候选轨迹 rollout；
4. 行人 snippet library；
5. 在直线轨迹旁放矩形墙体；
6. 移植横穿 snippet；
7. 生成 collision + temporal-safe pair；
8. 渲染 BEV；
9. 风险 GT。

### Phase B：风险模型

1. risk-only CNN；
2. BCE + continuous severity；
3. quantile heads；
4. occupancy baseline；
5. calibration。

### Phase C：完整事件多样性

1. 六类 paired variants；
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

1. 提出事件中心的反事实盲区场景生成器，利用真实行人轨迹先验构造大规模 collision/near-miss/hard-negative 样本；
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

> 本项目从真实机器人和行人轨迹中提取运动先验，但不依赖自然数据中稀少的遮挡冲突事件；它围绕候选局部轨迹上的潜在冲突点，程序化构造遮挡物和隐藏行人，生成成对的碰撞、近碰和 hard-negative 场景，用完整 oracle 世界训练轨迹相关风险模型，并通过反事实验证动作与重规划计算模拟器定义的净决策价值，最终在 execute、verify 和 reject 之间做局部规划决策。

