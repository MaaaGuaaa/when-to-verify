# 跨服务器环境复现

本文给出当前仓库可验证的环境边界和另一台 Linux 服务器上的安装步骤。它区分两件事：

1. **SOP-01～07 数据管线环境**：当前已经由单元测试、toy fixture、真实数据 smoke 和批量生成验证。
2. **SOP-08～14 训练与评测环境**：单进程 CPU 和单 GPU smoke 已验证；目标规模训练、分布式参数和最终论文运行配置尚未冻结。

因此，本文可以复现当前数据管线、训练/校准/评测入口及其 smoke test；它不声称目标规模论文实验已经运行完成。

## 0. 给另一台服务器 Codex 的当前任务

当前任务是**只准备环境，不启动训练**。可以把本节和本文路径直接交给另一台服务器上的 Codex。

Codex 应执行：

1. checkout 用户指定的项目 commit，确认 worktree clean；
2. 按第 2 节创建 `blind-risk` 环境并安装精确版本；
3. 执行 `python -m pip check`；
4. 在 Slurm GPU allocation 中执行第 3 节 CUDA sanity；
5. 执行第 4 节仓库测试；
6. 返回 Python 可执行文件、版本、GPU 名称、测试命令和结果。

Codex 当前不得：

- 自行替换或绕过仓库已有的训练、校准和评测入口；
- 启动训练、调参或生成 checkpoint；
- 修改 `pyproject.toml`、科学配置或 schema 来迁就服务器；
- 把未完成的 `outputs/` 目录当作训练集；
- 因驱动或 CUDA 不匹配而静默改用另一版 PyTorch。

完成标志是“环境、GPU 与仓库 smoke 检查通过，等待完整数据 handoff 和正式运行指令”，不是“目标规模训练已复现”。

## 1. 已验证版本

2026-07-19 使用下列环境完成当前代码测试和 SOP-05～07 数据生成：

| 组件 | 已验证版本 | 说明 |
| --- | --- | --- |
| Python | 3.10.14 | 当前正式解释器 |
| NumPy | 1.24.4 | 当前数组与 NPZ 语义基线 |
| PyYAML | 6.0.1 | 配置加载 |
| pytest | 8.3.5 | 测试 |
| PyTorch | 2.0.1+cu118 | 后续训练的暂定基线 |
| CUDA runtime in PyTorch | 11.8 | `torch.version.cuda` |
| cuDNN | 8.7.0 | `torch.backends.cudnn.version() == 8700` |
| pip | 24.2 | 已验证安装器 |
| setuptools | 82.0.1 | 已验证构建工具 |

当前 `pyproject.toml` 已精确固定核心、测试和 `train` extra 的版本，并在
`tool.learning-when-to-verify.verified-environment` 中记录已验证 CUDA 基线。它仍不是包含所有系统库与驱动的 lockfile；标准
Python dependency 字段也不能指定站点 NVIDIA 驱动。

## 2. 推荐安装：GPU 训练基础环境

以下命令使用 Conda 只创建 Python 隔离环境，其余包由 pip 精确安装。环境名可以修改。

```bash
conda create -n blind-risk python=3.10.14 -y
conda activate blind-risk

python -m pip install --upgrade pip==24.2 setuptools==82.0.1
python -m pip install torch==2.0.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e ".[test,train]"
python -m pip check
```

PyTorch 官方历史版本页给出的 2.0.1/CUDA 11.8 完整组合还包含
`torchvision==0.15.2` 和 `torchaudio==2.0.2`。当前项目没有使用这两个包，因此默认不安装，避免扩大依赖面。需要时再按
[PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) 的同一版本组合安装。

如果新服务器暂时只跑 SOP-01～07 CPU 数据管线，可以跳过 PyTorch，并使用
`python -m pip install -e ".[test]"`。后续开始训练前再安装 CUDA 11.8 wheel，并重新执行本文全部检查。

## 3. GPU 与 Slurm 检查

先确认 NVIDIA 驱动能够运行 CUDA 11.8 构建。登录节点没有 GPU 并不代表安装失败；必须在分配到 GPU 的计算节点上检查。

```bash
nvidia-smi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

在 Slurm 集群上，使用站点实际的 GPU partition 名称：

```bash
ENV_PYTHON="$(command -v python)"
GPU_PARTITION="gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

srun --partition="$GPU_PARTITION" \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=00:10:00 \
  "$ENV_PYTHON" -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

所有启用 `torch.use_deterministic_algorithms(True)` 的 GPU 训练与测试作业都必须在
Python 进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；否则 CUDA 10.2 及以上的
CuBLAS 线性层会被 PyTorch 的确定性门禁拒绝。该变量属于运行环境，不改变模型或数据契约。

若站点不使用 `--gres=gpu:1`，只替换资源申请参数，不修改项目代码或科学配置。

## 4. 仓库验证

固定要复现的 commit 后再安装 editable package。不要从一个 dirty worktree 生成正式产物。

```bash
git status --short
git rev-parse HEAD

python -m pip check
python -m pytest -q
python scripts/00_validate_contracts.py --config configs/base.yaml
```

最低限度的快速检查是：

```bash
python -m pytest tests/test_contracts.py tests/test_toy_fixture.py -q
python scripts/00_validate_contracts.py --config configs/base.yaml
```

正式复现记录至少保存：

```bash
python --version
python -m pip freeze --all
python - <<'PY'
import numpy
import torch
import yaml

print("numpy", numpy.__version__)
print("PyYAML", yaml.__version__)
print("torch", torch.__version__)
print("torch CUDA build", torch.version.cuda)
PY
printf 'CUBLAS_WORKSPACE_CONFIG=%s\n' "${CUBLAS_WORKSPACE_CONFIG:-unset}"
git rev-parse HEAD
```

## 5. Git、数据和产物边界

- `data/` 原始数据、`outputs/` 生成产物和本地 Conda 环境不随 Git 仓库分发。
- 在另一台服务器训练前，需要单独复制已完成的 SOP-03、SOP-04、SOP-05 和 SOP-07 产物，保留原目录层级和 handoff/manifest 文件。
- 不要复制仍在生成的目录，也不要把 partial shard 当成 complete artifact。
- 使用 handoff 中记录的 digest 和每个 shard 的正式 loader 验证副本；不能只检查文件数量。
- 任何需要 Git 身份的生成命令都应传入绝对、非符号链接的 Git 可执行文件：

```bash
GIT_EXECUTABLE="$(readlink -f "$(command -v git)")"
test -x "$GIT_EXECUTABLE"
```

机器专属的路径、Slurm partition、CPU/GPU 数量和输出目录应作为运行参数记录，不应写回冻结配置。

## 6. 当前限制与更新门槛

当前仓库中的正式 CLI 已覆盖数据准备、风险/占据训练、校准、验证数据与模型以及离线评测。因此：

- PyTorch 2.0.1+cu118 已通过训练 smoke、单 GPU batch、checkpoint round-trip 和固定 seed 检查，但当前声明仍不是包含驱动与系统库的完整 lockfile。
- 已合入的是单进程正式入口和 toy/smoke 证据；尚不能承诺目标规模 optimizer、AMP、分布式训练或最终论文 checkpoint 跨机器完全一致。
- 开始目标规模训练前，仍须固定数据族 digest、运行配置、Slurm 资源、代码 commit 和断点恢复策略，并完成一批小规模正式数据 smoke。
- 如果训练实现引入新依赖，应同时更新依赖声明和本文件，不得让实际环境继续领先于文档。

在上述门槛完成前，对外表述应为“数据管线与单 GPU smoke 环境可复现”，不能表述为“目标规模论文训练已经复现”。
