# SOP05 到 SOP06 双通道交接

_给负责 SOP06 批量渲染的实现方；自然分布数据现在即可开始，A-supplement 后续单独接入。_

---

## 当前结论

1. 先把现有自然分布数据渲染到 `outputs/sop06_history_bev_natural_v1/`。
2. 后续 A-supplement 渲染到 `outputs/sop06_history_bev_a_supplement_v1/`。
3. 两批产物保持独立、不可变，不向第一批目录追加文件。
4. SOP06 只渲染 SOP05 已确定的场景，禁止重新旋转、重新抽 presence 或重新选择未来轨迹。

## 生产入口

建议新增：

```text
scripts/06_generate_single_scene_bev.py
  --source-mode complete_mother|partial_m6_reconstruction
  --source-root <mother-or-partial-root>
  --final-scenario-root <six-file-release>
  --split <split>
  --output-dir <new-output-root>
  [partial M6 reconstruction arguments]
  --workers <n>
```

仅给 `final-scenario-root` 不够。SOP06 还需要母事件中的 `BaseState`、地图、机器人轨迹、
障碍物和上下文行人。

## 立即可跑的自然数据

| entry | source mode | accepted | SOP06 output |
|---|---|---:|---|
| `train-first10k` | `partial_m6_reconstruction` | 6790 | `natural_v1/train-first10k` |
| `train-after10k` | `partial_m6_reconstruction` | 75889 | `natural_v1/train-after10k` |
| `calibration` | `complete_mother` | 11622 | `natural_v1/calibration` |
| `val` | `complete_mother` | 14402 | `natural_v1/val` |
| `test` | `complete_mother` | 14264 | `natural_v1/test` |

### train-first10k

```text
source_root=outputs/sop05r_teb_long40_train_10k_m6_fault_isolation_v1
final_scenario_root=outputs/sop05_final_blindspot_train_m6_first10k_p30_initial_visibility_v3
sop03_root=outputs/sop03_thor_full_schema4_v1
long40_human_artifact=outputs/sop03_thor_motion_snippet_long40_human_schema4_v1/train/human
base_state_start=0
max_base_states=4000
source_digest=98ae57564d18cefe958f8780f21e5732263a355553353c1f1a6bf6d9f5e221b6
```

### train-after10k

```text
source_root=outputs/.sop05r_teb_long40_train_after10k_full_m6_v1.sop05r-teb-stage-5fvk6eb5/sop05r_teb_long40_train_after10k_full_m6_v1
final_scenario_root=outputs/sop05_final_blindspot_train_m6_after10k_115004_p30_initial_visibility_v3
sop03_root=outputs/sop03_thor_full_schema4_v1
long40_human_artifact=outputs/sop03_thor_motion_snippet_long40_human_schema4_v1/train/human
base_state_start=1026
max_base_states=11755
source_digest=23c1c53db976eb7464c5f1dadfe1e1c724347cdbbb6b359d29109eb42aca0a99
```

这两个 train 条目不能调用普通 `load_sop05r_teb_output(..., require_complete=True)`。
必须复用 `src/generation/sop05_partial_m6_final.py` 的 partial-M6 重建语义，使用相同
`SOP03 / Long40 / BaseState` 范围重算 source identity，并与上面的 digest 比较。

### held-out splits

```text
calibration:
  source_root=outputs/sop05r_teb_long40_calibration_m6_v1
  final_scenario_root=outputs/sop05_final_blindspot_calibration_p30_h0_only_v1
  source_digest=9bab0d997912a14fc108e3e20fa73d42be247ac8bd8ed6acdc7bf9d0d0e7b8ed

val:
  source_root=outputs/sop05r_teb_long40_val_m6_v1
  final_scenario_root=outputs/sop05_final_blindspot_val_p30_h0_only_v1
  source_digest=7c490904d5b4a61c088ddd1e6418ac6da45c180a6e7932a05b493e4c011803bb

test:
  source_root=outputs/sop05r_teb_long40_test_m6_v1
  final_scenario_root=outputs/sop05_final_blindspot_test_p30_h0_only_v1
  source_digest=a97f39830dbebd66a41f1075fdbb6f2d2367df1a1284ea8e5fa88ec4d98b142c
```

held-out 条目必须同时严格加载 mother root 和 final root，并检查：

```text
final.source_publication_semantic_digest == mother.publication_semantic_digest
```

## 从 finalized record 构造 SOP06

1. 扩展 final-scenario loader，使其返回已经校验过的 `history_poses`、`future_poses`、
   `target_present` 和 `source_record_indices`；不要绕过现有 checksum、shape 和行对齐校验。
2. 只处理 `records.jsonl` 中 `status=accepted` 的记录。
3. 使用 `scenario_id` 作为 SOP06 `sample_id`，原样保留 `mother_id` 和 `split`。
4. B 类把已认证的 8 帧历史及 observed mask 交给
   `adapt_seen_prior_result`/等价持久化适配器；32 帧未来只保留在 oracle 侧。
5. A-present 不向 renderer 暴露目标历史，只把 32 帧未来放入 oracle world；A-empty
   不加入目标历史或目标未来。
6. 调用 `render_sop06_single_publication`。不得再次执行 `generate_unseen_prior_mother`
   或 seen-prior 未来选择。

建议新增一个直接面向持久化 final record 的公共适配器，避免为了调用 SOP06 而伪造
sampling provenance 或重新运行 SOP05。

## 输出与断点恢复

- 每个 entry 单独发布一个 shard，不要先合并五个 entry。
- 不要调用 `coordinate_sop06_single_release` 聚合 natural 与 supplement；其现有 hard cap
  是 `125000`。
- 输出必须使用 staging、严格重载和原子改名；存在同名输出时只允许完全一致的 replay。
- manifest 至少记录 `source_mode`、SOP05 source digest、final release checksum identity、
  split、accepted count 和 source family。

## A-supplement 后续接入

A-supplement 会提供四个新的 complete-mother source/final 配对，接口与 held-out 完全相同：

```text
source_family=a_supplement
source_mode=complete_mother
output_root=outputs/sop06_history_bev_a_supplement_v1/<split>
```

自然分布 SOP06 不需要等待这些产物。补充集完成后启动第二组 SOP06 作业即可。

## 最小验收

- source digest 与 final root 一致
- 每个 accepted `scenario_id` 恰好产生一条 SOP06 observation
- `(split, mother_id)` 和 `sample_id` 在所有输出中唯一
- A 的 renderer input 不含目标真值历史；B 只包含 observed mask 允许的历史帧
- renderer input 与 metadata 不含 future、oracle、angle、attempt、collision、clearance 或 risk
- natural 与 A-supplement 输出目录互不覆盖

详细设计见
[`2026-07-27-sop05-a-supplement-sop06-dual-lane-design.md`](superpowers/specs/2026-07-27-sop05-a-supplement-sop06-dual-lane-design.md)。

