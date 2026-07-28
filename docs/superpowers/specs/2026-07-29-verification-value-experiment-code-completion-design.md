# 验证价值实验代码补全设计

## 目标与完成边界

本轮交付“可正式运行的实验代码”，不在本轮生成全量数据、训练正式模型或执行大规模
回放。完成后，SOP11–16 应能由冻结配置驱动，从 SOP5 正式样本发布一直运行到验证价值
模型、闭环评测和结构化实验汇总；小型 fixture 能验证整条证据链，正式规模只需替换
输入路径和 Slurm 资源参数。

本轮不重写已经存在且语义正确的动作、反事实观测、后验、重规划、标签、模型和闭环
模块。实现以补齐生产入口、配置迁移、校准、实验编排和证据校验为主。

## 冻结科学语义

- 所有主动路径统一使用 Schema 4 Long40：8 帧历史、32 个未来端点、40 点轨迹片段。
- 模型输入只包含部署时可获得的历史和动作信息；world variant、碰撞真值、未来轨迹、
  posterior oracle 只用于标签或离线审计。
- 验证价值使用相对任务后悔
  `max(0, J_policy / J_nominal - 1)`；动作代价只计一次。
- `value_target = BR_before - post_risk`，`useful_target = int(value_target > 0)`，不按标签
  正负补采样、筛样或平移标签。
- `reject_cost` 只由训练 split 校准，冻结后才允许生成 held-out 标签；held-out、test 和
  replay 不得反向影响阈值。
- 数据分组、随机种子和任务身份必须跨进程稳定，不依赖 Python `hash()`。

## 端到端数据流

```text
SOP5 finalized handoff
  -> 可续跑 verification release shards
  -> train-only reject_cost calibration
  -> frozen GT config + calibration artifact
  -> train/validation/test/replay collections
  -> V0 与消融/基线的多种子训练
  -> held-out 指标、切片和校准结果
  -> authenticated closed-loop replay
  -> experiment matrix + result registry + paper-ready tables
```

每一层消费上一层的不可变 manifest，而不是仅接收一个可替换的路径。manifest 至少绑定
schema、生成语义版本、配置摘要、数据划分、上游摘要、种子、样本数和 payload checksum。

## 实现范围

### 1. Schema 4 收口

将 `configs/verify_model.yaml`、验证来源 fixture 和所有主动验证实验入口迁移至 Schema 4。
加载器继续 fail closed，不为兼容旧配置降低版本检查。测试中保留的旧 schema 只能作为
“应拒绝”的显式负例，不得被正式入口消费。

### 2. 正式数据发布入口

以正在形成的 `verification_release` 实现为唯一正式发布路径：固定任务清单、完整六动作
组、有限内存 shard、严格重载、原子发布、请求身份一致时续跑。toy/smoke 的 10–100
样本限制不得出现在正式模式。预期不可用场景记录结构化拒绝；其他异常立即失败，不能
静默跳过或按标签补样。

并发写入中的以下文件在其状态稳定后只做集成审查和必要补丁，不覆盖其现有实现：
`scripts/08_generate_verification_dataset.py`、
`src/generation/verification_release.py` 及对应测试。

### 3. 训练集阈值校准

新增可复用校准入口，读取训练 release 中保存的 unclipped policy losses，在候选
`reject_cost` 上重估而不重新仿真。输出机器可读 JSON，包含输入/配置摘要、候选值、
种子、总量与有效量、正负比例、分位数、逐动作统计、rejection-selection rate 和最终
选择理由。

选择规则实现为配置化、确定性的纯逻辑。没有候选值满足冻结标准时命令失败并保留完整
诊断，不自动调低动作成本、不重心化标签。校准产物通过摘要绑定到后续 GT 生成配置。

### 4. 训练、基线与 held-out 评测编排

训练入口支持显式数据 manifest、输出目录、seed、resume 和 run identity；恢复时验证
配置及数据摘要，不兼容时拒绝复用 checkpoint。正式实验至少覆盖：

- V0 主模型，多种子独立训练并汇总均值与离散程度；
- no-ranking 消融；
- 规则/启发式验证价值基线和已有学习基线；
- 风险方法与受控场景组；
- 已登记的结构、动作库、先验、验证成本、margin、risk weight、`M` 与 `tau` 敏感性。

held-out 评测一次遍历同时输出价值回归、useful 分类、决策质量、动作级统计、场景切片、
校准曲线所需原始分箱统计和可追溯逐样本记录。聚合脚本只能读取结构化结果，不能手抄
数字或根据文件名推断实验语义。

### 5. 闭环与实验矩阵证据绑定

闭环回放消费已认证的 collection、checkpoint 和冻结策略配置，并在结果中记录三者
摘要。矩阵 runner 不仅检查路径、seed、status 和 policy，还应校验声明的风险方法、
价值方法、消融开关与敏感性参数确实出现在相应配置/manifest/checkpoint 证据中。

同一 run identity 的成功结果保持不可变；完全相同的请求可幂等复用，身份冲突则失败。
失败、部分写入或缺少必要切片的结果不能进入最终汇总。

### 6. 正式产物

既有目录约定下生成以下结构化产物：

- release manifest、shard summaries 和结构化 rejection records；
- reject-cost calibration JSON；
- 每个训练 run 的 manifest、checkpoint 索引和训练历史；
- 每个 held-out/closed-loop run 的逐样本结果、聚合指标和 manifest；
- matrix status、完整性报告及由结果自动生成的 CSV/JSON 表。

所有重要文件采用 staging 后原子发布，默认拒绝覆盖不同身份的已有成功产物。

## 错误处理与防泄漏

- schema、版本、摘要、split、样本分组或配置身份不一致时立即失败。
- 校准器明确拒绝非训练 split；训练统计量只在训练集拟合。
- collection 构建检查 schema、shape、dtype、NaN/Inf、范围、掩码、样本数、完整六动作
  组及跨 split group overlap。
- 导入模块不触发数据生成、训练、外部调用或写盘。
- 可恢复错误与科学不合格场景使用有类型的记录；未知异常不得宽泛捕获后继续。

## 验证策略

采用测试驱动方式，先用回归测试固定当前失败，再实现最小改动。所有测试、lint、
typecheck 和小型端到端运行均通过 Slurm：

1. Schema 4 配置、模型、训练、评测和来源测试；
2. release 单/多 worker 确定性、续跑、原子发布和身份冲突测试；
3. train-only 校准、手算样例、无可用候选和摘要绑定测试；
4. checkpoint resume 身份、multi-seed 汇总、基线/切片输出测试；
5. matrix 语义声明与真实证据不匹配时的拒绝测试；
6. 小型 fixture 的 release → calibration → train → held-out → closed-loop → report 链路。

小型链路只证明接口和证据链完整，不代表正式规模的数值结论或运行性能。

## 验收标准

- 主动验证路径、配置和正向测试中不存在 Schema 3。
- 正式 CLI 无 smoke 样本上限，并清楚区分 toy/smoke/final 模式。
- 校准阈值只能从 train 选择，冻结产物可被后续严格验证。
- V0、no-ranking、基线、多种子、held-out 切片、闭环和矩阵均有可运行入口及配置绑定。
- 相关最小测试与小型端到端链路通过，结果可由结构化产物重建。
- 不要求本轮启动全量数据生成、正式训练或生产级回放；这些数值结论仍需后续 Slurm
  正式任务提供。
