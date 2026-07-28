# SOP11–13 可续跑并行数据发布设计

## 目标与语义边界

生产数据的唯一工作单元是一个 SOP5 母事件已经采样出的一个子事件。下游针对该固定
子事件执行一次 SOP11–13 任务，并在任务内部共同评估六个验证动作。生产入口不得再次
生成 world variant、scenario bank、posterior 或下游扰动；不可用任务只记录明确原因，
不得按价值标签补采样。

目标是在不改变标签和数据格式科学语义的前提下，限制内存与共享文件系统压力，提高
不同母事件之间的并行度，并使中断后的损失最多局限于一个未完成 shard。

## 执行架构

采用“固定任务清单 + 分区任务 + 有界 fork worker pool”：

1. 启动时从可信上游 handoff 固定母事件顺序和任务身份。
2. 一个 Slurm 分区任务只加载一次公共只读上下文，并只处理自己的连续任务边界。
3. worker 只接收母事件索引，每次完成一个母事件的六动作组；同一母事件不拆分并行。
4. 默认四个 worker，在途结果不超过 worker 数，BLAS/OpenMP 线程固定为一，避免嵌套
   并行和结果积压。
5. 正式并发由小批基准的吞吐、峰值 RSS 和写盘速率决定；最大并发需保留至少 30%
   Slurm 内存余量，且不得超过分配 CPU。

不采用“一母事件一 Slurm 任务”，因为它会反复加载公共输入并造成元数据 I/O 风暴；
也不采用全量单进程缓存，因为失败重算范围和内存峰值过大。

## 分片、I/O 与续跑

- 默认每 16 个母事件形成一个完整边界，即最多 96 个训练样本；边界始终保留完整
  六动作组。
- parent 仅缓存当前 shard，使用 staging 目录写压缩 NPZ、JSONL 和 summary，严格加载
  一次后原子发布；发布后立即释放样本。
- payload 校验和按固定大小块流式读取，禁止 `read_bytes()` 再持有整份压缩文件。
- 根目录保存不可变 request identity；已完成 shard 只有在身份、边界和 digest 一致时
  才复用。失败后保留此前完整 shard，只重算当前及之后的未完成边界。
- 多个 Slurm 分区写各自隔离目录；最终汇总只读取 shard summary/digest，不再次全量
  解压所有 NPZ。

## 错误处理与验收

预期的 `VerificationSourceIneligibleError` 形成结构化拒绝记录；其他异常立即失败并保留
已完成 shard。最终 manifest 统计 processed、accepted、rejected、六动作样本数、每动作
正负数和 value quantile，并验证：

- 每个已接受母事件恰有一个 `sampled_child_world_id` 和完整六动作组；
- 任务 ID、sample ID、split 和 shard 边界无重复或遗漏；
- 数组 schema、shape、dtype、NaN/Inf、checksum 和 semantic digest 正确；
- 不出现 bank/posterior/M 控制或 oracle 字段进入模型输入。

验证分两层：通过 Slurm 运行相关单元测试；再用少量真实母事件比较 workers=1 与
workers=4 的结果 digest 完全一致，并记录 wall time、峰值 RSS 和最大写盘速率。正式
运行只保留必要的逐 shard 验收和一次最终 manifest 汇总，不做重复全量审查。
