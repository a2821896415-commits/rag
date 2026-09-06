---
name: 修复 venv 缺失的 torch（GPU 版 2.5.1+cu121）
overview: 核对确认 pyproject.toml 与 uv.lock 期望状态完全一致（均要求 torch 2.5.1+cu121，来源 download.pytorch.org/whl/cu121；requires-python 已为 >=3.11），不存在依赖声明层不一致。当前查询/评估报 torch NoneType 崩溃的根因是 .venv 实际未安装 torch（pytorch.org 下载未完成）。修复 = 完成 uv sync 把 torch 装进 venv，并验证 CUDA 可用后重跑任务。
todos:
  - id: check-lock
    content: 运行 uv lock --check 确认 pyproject.toml 与 uv.lock 已一致
    status: completed
  - id: sync-torch
    content: 执行 uv sync 把 torch 2.5.1+cu121 装回 venv（允许官方源下载耗时）
    status: completed
    dependencies:
      - check-lock
  - id: verify-env
    content: 验证 torch/transformers 可导入且 torch.cuda.is_available() 为 True
    status: completed
    dependencies:
      - sync-torch
  - id: fallback-cpu
    content: 若下载持续失败，改回清华 PyPI CPU torch 并回退 .env 设备为 cpu
    status: cancelled
    dependencies:
      - sync-torch
---


## 用户诉求
查询代码与评估脚本当前全部跑不了，均报 `transformers` 导入期的 `TypeError: expected string or bytes-like object, got 'NoneType'`（`_torch_version` 为 None）。用户怀疑是 `uv.lock` / `pyproject.toml` 期望状态不一致导致。

## 核对结论（已读原文）
- `pyproject.toml` 与 `uv.lock` **完全一致，不存在声明层冲突**：
  - `pyproject.toml:26,67-70` → `torch>=2.5.1`，`[tool.uv.sources]` 将 torch/torchvision/torchaudio 指向 `pytorch-cu121`。
  - `pyproject.toml:62-65` → `[[tool.uv.index]] pytorch-cu121 = https://download.pytorch.org/whl/cu121`（`explicit=true`）。
  - `uv.lock:139` → `{ name = "torch", specifier = ">=2.5.1", index = "https://download.pytorch.org/whl/cu121" }`。
  - `uv.lock:5885-5887` → `torch 2.5.1+cu121` 来自 `https://download.pytorch.org/whl/cu121`。
- `requires-python = ">=3.11"`，3.14 解析冲突已消除。

## 真实根因
不是 lock/pyproject 不一致，而是 **`.venv` 里 torch 没装成功**：此前 `uv sync/uv run` 从 `download.pytorch.org` 拉 `torch 2.5.1+cu121`（Windows 上约 2GB）未能完成/被中断，导致 `import torch` 直接 `ImportError` → `transformers` 内 `_torch_version=None` → NoneType 报错。国内 aliyun/tsinghua 的 `pytorch-wheels/cu121` 均不含 Windows CUDA 轮子，故 Windows GPU torch 只能从官方源拉。

## 核心目标
在不改动已生效的 GPU 默认配置（`.env` 的 `BGE_DEVICE/BGE_RERANKER_DEVICE=cuda:0` + `embedding_utils.py` 兜底 `cuda:0`）前提下，把可用的 torch 装回 venv，让查询与评估脚本恢复运行。



## 技术栈
- 包管理：uv（已锁定 `uv.lock`，torch 解析为 `2.5.1+cu121`）
- 运行环境：Python 3.11+，`download.pytorch.org/whl/cu121` 作为 torch 唯一 GPU 源
- 受影响运行链路：`transformers`/`flagembedding`/`pymilvus-model`/`BGE-M3`/`bge-reranker-large` 均依赖 `torch`

## 实施方案
1. **一致性校验（只读，不改动）**：执行 `uv lock --check` 确认 `pyproject.toml` 与 `uv.lock` 无需重新解析。预期输出"lock 已是最新"，证明无需 `uv lock`，避免误触发重新解析引入 `requires-python`/cu 源震荡。
2. **补齐 venv 中的 torch（主路径，保 GPU）**：执行 `uv sync`。它会按现有 lock 把 `torch 2.5.1+cu121` 从官方源下载并装回 venv。注意：
   - 此前中断的下载在 uv 缓存中可能部分保留，`uv sync` 可断点续传，二次执行通常比首次更快。
   - 这是 Windows 上获得 GPU torch 的唯一可靠途径（国内镜像无 Windows CUDA 轮子）。
   - 耗时较长（大体积 wheel），需允许其完成，勿中途取消。
3. **环境验证**：`uv run python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available())"`。期望 `2.5.1+cu121` 且 `cuda.is_available()==True`，证明 torch 已就位、GPU 默认配置有效。
4. **回退方案（仅当主路径持续下载失败）**：将 `pyproject.toml` 的 torch 源从 `pytorch-cu121` 改回默认清华 PyPI（CPU 版，快速可靠），重锁并 `uv sync`；同时把 `.env` 的 `BGE_DEVICE`/`BGE_RERANKER_DEVICE` 与 `embedding_utils.py` 兜底改回 `cpu`。此举以牺牲 GPU 加速换取"必能跑通"，用于网络无法访问官方源时的兜底。

## 实施注意
- 不要重复跑 `uv lock`：当前 lock 与 pyproject 已一致，重新解析反而可能再次触发 cu 源/Python 版本解析问题。
- 保留 `requires-python = ">=3.11"`，勿恢复含 3.14 的范围。
- GPU `.env` 设置仅在存在 CUDA torch 时生效；主路径保持现状即可。
- 验证时优先用 `uv run`（自动按 lock 同步），避免直接用系统 python 指向错误解释器。

## 架构与目录
本任务为依赖/环境修复，不涉及业务代码架构变更：
- `pyproject.toml`：保持现状（torch 源 = pytorch-cu121），仅在回退方案中改为清华 PyPI。
- `uv.lock`：保持现状（torch 2.5.1+cu121），主路径不重锁。
- `.venv/`：修复目标，需补齐 torch。
- `.env` / `app/shared/model/embedding_utils.py`：主路径保持不变；回退时才改设备为 cpu。

