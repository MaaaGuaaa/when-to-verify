# SOP05R M5 半平面余量采样放置设计

_状态：v8 已实现；M5 产率目标尚未达到；2026-07-26。_

## 目标

在固定同步碰撞点的前提下，根据机器人 index 3 射线、障碍物代表点射线和可旋转
的行人 index 3 射线之间的半平面关系，直接构造少量高概率满足
`seen -> occlude` 的普通角度。生成阶段与验收阶段必须调用同一个历史可见性判定，
不再通过独立的 `initial_visible` / `initially_hidden` 请求分组二次淘汰候选。

以下约束保持不变：

- 真实 Long40 行人轨迹为 8 帧历史、index 7 当前帧和 32 帧未来；
- 机器人和行人在同一个 future index、同一个世界坐标碰撞；
- 行人只允许刚体旋转和平移，空间缩放固定为 `1.0`；
- 决策时间固定为 index 7 的 `t=0`，碰撞下限仍为决策后 `1.2 s`；
- 最终仍执行范围、速度、加速度、静态障碍、上下文和连续碰撞检查。

## 当前问题

旧版 M5 生成 360 个 1 度旋转并批量检查，且生成谓词和验收谓词不一致。第一版
构造式修改又把目标误解成“寻找行人中心轨迹刚好由与障碍物相交变为不相交的
临界角”，为此增加 10 度粗扫、1 度细化和反向扫掠。该搜索既费时，也把行人放在
障碍物边界附近，未给 footprint 留出净空。

v5 的真实 Long40 20 条工程 smoke 基线为：

```text
M4 = 24 / 74
M5 = 21 / 94 = 22.3%
M6 = 20 / 21
tested rotation candidates = 12,109
initial_visibility_missing = 6,031
target_occluder_collision = 5,505
```

该 smoke 只用于同输入、同种子的工程对比，不代表正式规模产率。临界角版本的
30 次真实诊断共构造 3,684 个边界角，其中 126 个满足历史语义，但 126 个全部被
`target_occluder_collision` 拒绝；额外增加普通 60 度余量后，10 次尝试已有 2 个
候选通过。这说明根因是临界搜索刻意贴近障碍物，而不是可行角度范围过窄。

## 统一历史可见性定义

令 `blocked[i]` 表示机器人和行人在历史 index `i` 的中心连线与任一表示障碍物
相交。`False` 表示可见，`True` 表示遮挡。历史只使用 indices `0..7`。

首选样本定义为：

```text
blocked[0:4] 全为 False
并且 blocked[4:8] 至少一个为 True
```

也就是前四帧全部可见，之后至少出现一次遮挡。这一类记为
`prefix4_visible_then_occluded`，是新的首选 `initial_visible` 语义。

回退样本不要求前四帧全部可见，但必须同时满足：

```text
历史可见帧数 >= 4
历史遮挡帧数 >= 1
存在 i < j，使 blocked[i] == False 且 blocked[j] == True
```

回退样本记为 `fallback_seen_then_occluded`。不满足上述顺序关系的“先遮挡后可见”
或纯可见/纯遮挡窗口不得发布。

M5 不再预先抽取 requested stratum，也不维持 `0.8 / 0.2` 的人为初始分组配额。
候选的 observed class 由上述统一函数事后确定。首选候选先检查，首选全部失败后
继续检查回退候选。

## 固定碰撞锚点

对一个机器人 route anchor：

```text
k = 7 + round(route_time_s / 0.2)
robot_xy[k] == transformed_human_xy[k] == collision_xy
```

行人完整轨迹绕 source index `k` 旋转并平移到 `collision_xy`。任何旋转都保持
index `k` 的位置和时间不变，因此在点模型下，未来碰撞时间不随旋转变化。有限
体积可能使首次接触略微提前，但这只由最终连续碰撞 authority 检查，不参与候选
角度构造。

路线锚点先按累计路径长度落在 `[0.20, 0.95]`、处于 Long40 future horizon 内且
严格早于路线终点筛选。保留其中路径进度最大的至多 8 个采样点，并按从后到前的
确定性顺序 first-fit；不再随机打乱锚点。这样优先为行人片段保留较长的前置轨迹，
同时保持每个 template 的锚点预算不变。

## 半平面余量采样

### 1. 定义三条射线

令 guide index `g=3`：

```text
a = robot_xy[g] - collision_xy
b = representative_occluder_xy - collision_xy
c0 = source_human_xy[g] - source_human_xy[k]
theta_align = atan2(a) - atan2(c0)
```

`a` 是碰撞点到机器人 index 3 的射线，`b` 是碰撞点到障碍物代表点的射线，
`c0` 是旋转前碰撞点到行人 index 3 的射线。`theta_align` 先令 `c0` 与 `a`
重合。若任一射线长度小于数值容差，则该组合以
`guide_ray_degenerate` 拒绝，不通过随机角度掩盖退化输入。

### 2. 选择障碍物代表点

对 L 形障碍物仍按其矩形 primitive 处理。选取到线段
`collision_xy -> robot_xy[3]` 欧氏距离最小的障碍 primitive 中心作为代表点；
距离并列时按 `occluder_id` 决定。中心点只是对“障碍物所处方向”的稳定近似，
不用于代替后续精确线段遮挡或 footprint 碰撞检查。

### 3. 在同一半平面增加普通角度余量

计算障碍物相对机器人射线的有符号夹角：

```text
signed_ab = wrap(atan2(b) - atan2(a))
side = sign(signed_ab)              # 近似共线时固定为 +1
signed_ac = signed_ab + side * margin
rotation = wrap(theta_align + signed_ac)
```

以 `a` 为零点建立每 5 度的固定角度网格。从 `b` 开始，只沿使 `|ab|` 单调
增大的方向检查网格中的下一格、再下一格。仅保留
`0 < side * signed_ac < 180 度` 的候选。于是每个保留候选天然满足：

```text
abs(angle(a, c)) > abs(angle(a, b))
b 与 c 位于直线 a 划分的同一个开半平面
```

这里不检查完整行人轨迹是否与障碍物相交，不寻找相切或“刚好脱离相交”的边界，
也没有反向扫掠或 1 度细化。主阶段每个锚点和时间缩放最多产生 35 个确定性候选，
少于全圆 5 度采样的 72 个候选。只有主阶段所有通过历史语义的候选均未通过物理
门禁时，才进入回退阶段：扫描主阶段之外的固定网格，并按与 `b` 的圆周距离由近到远
检查。

## 候选排序与统一验收

每个余量角度只计算一次精确的 8 帧同步中心线遮挡向量，并调用同一个纯函数分类：

1. `prefix4_visible_then_occluded`；
2. `fallback_seen_then_occluded`；
3. 不合格。

每个搜索阶段内，所有首选角度先进入物理检查，随后才是回退角度。M5 first-fit
接受第一个通过全部检查的候选；只有主阶段无物理可行候选才展开其余方向。M6 根据持久化的 `blocked_indices`
调用同一个分类函数复核，不能再额外检查 index 0、决策帧或 requested stratum。

半平面关系只负责提出高概率候选，不直接给出可见性标签；最终发布的历史可见性
仍由 8 条精确同步中心线检查决定，不允许近似误判。

遮挡 witness 取历史中最后一个遮挡 index。由于两类合格样本都要求存在更早的
可见帧，该 witness 必然能表达 `seen -> occlude` 顺序。

## 最终物理检查

半平面余量构造不能替代以下 authority：

- 行人 footprint 全 40 帧位于栅格内；
- 行人 swept footprint 不与表示障碍、源静态图和上下文动态对象相撞；
- 行人速度和加速度满足真实对象策略；
- 锚点位置、Long40 index 和 route time 保持同步；
- 机器人与行人的首次连续接触位于决策后 `[1.2, 6.4] s`。

其中 target-occluder footprint 碰撞只会淘汰净空不足的角度，不改变碰撞锚点
或可见性类别。

## 配置与版本边界

建议冻结以下 placement 参数：

```yaml
placement:
  occluder_angular_margin_step_deg: 5.0
```

`occlusion.initial_visible_weight` 和 `occlusion.initially_hidden_weight` 删除；保留
`minimum_visible_history_frames=4` 和 `minimum_occluded_history_frames=1` 作为回退
样本门禁。首选前缀长度固定为 4，并进入 occlusion 语义版本。

本次行为改变必须升级语义身份，建议值为：

```text
generator       = obstacle_first_lightweight_teb_v8
placement       = anchored_human_half_plane_step_long40_v8
occlusion       = seen_then_occlude_prefix4_v4
collection      = sop05r_nominal_trajectory_collection_v8
run             = sop05r_lightweight_teb_generation_run_v7
manifest        = sop05r_lightweight_teb_manifest_v7
summary         = sop05r_lightweight_teb_summary_v8
completion      = sop05r_lightweight_teb_producer_complete_v7
candidate_search= synchronized_half_plane_step_seen_then_occlude_v5
```

Schema 仍为 `4.0.0`，Long40 layout、planner 和 template 版本不变。v8 loader 必须
拒绝旧版本或混合语义产物。

## 证据与计数器

M5 summary 至少记录：

- 三条射线退化数；
- 构造的余量角度数、实际物理候选数；
- 首选和回退候选数、各自接受数；
- 每个既有物理拒绝原因。

placement provenance 保存 `theta_align`、`signed_ab`、半平面方向、选中余量、最终
旋转角度、observed class、candidate-search 版本、配置 digest 和 seed。

## 测试与验收

实现必须按 TDD 完成以下回归：

1. indices `0..3` 全可见且 `4..7` 存在遮挡时分类为首选；
2. 前四帧未全可见但满足至少四帧可见和严格 `seen -> occlude` 顺序时分类为回退；
3. 只有 `occlude -> seen`、纯可见或纯遮挡的历史被拒绝；
4. index 3 对齐角度正确，旋转前后碰撞 index 的位置和时间完全不变；
5. 障碍物在 `a` 左、右两侧时，所有候选 `c` 都分别与其同侧且夹角更大；
6. 采样在 180 度前停止，且候选器不调用完整轨迹与障碍物相交扫描；
7. 候选排序始终首选优先、回退随后，生成和 M6 复核使用同一分类函数；
8. 全部既有范围、footprint、上下文、运动学和连续碰撞门禁保持有效。

聚焦测试通过后，在 Slurm 使用与 v5 基线相同的真实 Schema 4 Long40 输入、种子、
137 个 base-state 上限和新的 no-overwrite 路径运行 20 条 smoke，并独立严格 reload。
验收要求：

- 发布和严格 reload 均为 20/20；
- M6 条件成功率不低于 v5 基线；
- M5 成功率高于 v5 的 `21/94=22.3%`；
- `initial_visibility_missing` 不再存在；
- 报告主阶段每个锚点和时间缩放至多 35 个角、回退触发次数，以及相对全圆采样的实际降幅；
- 报告首选/回退接受比例、完整拒绝分布和运行时间；
- 首选样本是否占多数只作为实测结论报告，不作为硬编码发布配额。

## 范围外事项

本设计不改变 M4 障碍模板、route-anchor 路径比例、行人 snippet 选择、Long40 时间
契约、TEB planner、连续碰撞 authority 或后续 revealability 定义；仅将同一合格
路径比例内的锚点优先级改为后段优先。时间缩放需要
源轨迹外推时仍应拒绝，不通过裁剪、补零或复制终点绕过。

## 工程验证结果

- 同一 30 次真实 M5 诊断中，半平面 10 度采样和全圆 10 度采样均接受 7 次；
  前者构造 3,633 个角、执行 47 次物理检查，后者分别为 15,192 和 185，因而选择
  半平面方案；
- 最终 20 条 smoke 为 M4 `30/87`、M5 `20/175=11.4%`、M6 `20/20`，并通过
  completion 严格 reload；
- M5 未超过旧版 `21/94=22.3%`。候选层 22,233 个角中 391 个满足历史语义，
  其中 327 个被 `target_occluder_collision` 拒绝。继续加密角度没有证据支持；
  剩余工作应转向行人片段与碰撞锚点的物理可行性排序或构造。
