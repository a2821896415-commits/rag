---
name: backfill_stale_chunk_uid
overview: 重跑 backfill_chunk_uid.py 刷新测试集中 20 个失效的 golden chunk_uid（集中在撒哈拉的故事/活着/蛤蟆先生去看心理医生三本书）。先只读核对本地 chunks.json 是否反映重导入后的当前内容，再执行回填（自动备份 dataset.json），最后用 lint 验证 0 警告。
todos:
  - id: precheck-chunks
    content: 用 [subagent:code-explorer] 只读预检 3 本书本地 chunks.json 与 evidence 命中情况
    status: completed
  - id: run-backfill
    content: 执行 backfill_chunk_uid.py 写回 dataset.json（自动备份）并查看 matched/unmatched
    status: completed
    dependencies:
      - precheck-chunks
  - id: verify-lint
    content: 重跑 lint_dataset.py 确认 20 个 uid warning 降为 0
    status: completed
    dependencies:
      - run-backfill
  - id: verify-eval
    content: 可选 run_eval --no-gen --limit 3 确认 case 回到 exact 且 recall=1.0
    status: completed
    dependencies:
      - verify-lint
---

## 用户需求
用户要求重跑 `evaluation/backfill_chunk_uid.py`，刷新 `dataset.json` 中 20 个失效的 golden `chunk_uid`。

## 背景与定位
经 `lint_dataset.py` 全量校验，100 条 case 中有 20 个 golden `chunk_uid` 在 Milvus 中查不到（0 error / 20 warning）。这些失效 uid 全部属于 3 本书：撒哈拉的故事（6 个）、活着（7 个）、蛤蟆先生去看心理医生（7 个）。根因是这 3 本书在数据集生成之后用新 schema 重新入库，Milvus 生成了新 `chunk_uid`，旧 uid 随之失效。用户点名核对的 `f0c9664a7898003a`（三体 b01-001）经验证有效，不受影响。

## 目标
通过重跑回填脚本，让这 20 条 case 恢复 exact 精确匹配（而非软匹配），消除 lint 的 20 个 warning，使检索评测的 recall@k / mrr / ndcg 指标更精确、可复现。


## 技术方案

### 复用现有脚本，不新增代码
直接执行已有的 `evaluation/backfill_chunk_uid.py`，不改动其逻辑。该脚本：
- 扫描本地 `output/**`、`source/**` 下的 `chunks.json`（split 阶段落盘切片），**不读 Milvus**；
- 复用 `app/rag/import_/split_service.py` 的 `compute_chunk_uid(c)` 为每个本地块计算稳定 uid（公式仅依赖 `file_title/parent_title/part/content` 等切分字段，与入库时一致）；
- 对 `dataset.json` 每个 `golden_chunks` 条目，按 `(item_name, content_type, evidence⊂content)` 子串匹配到本地块并写入其 `chunk_uid`；匹配池优先级：同书同类型 > 任意书同类型 > 同书全部类型 > 跨类型兜底；needle 长度依次取 `len(ev)/30/15`；
- 写回前自动备份为 `dataset.json.pre_chunkuid.bak`，并打印 `matched/unmatched`。

### 关键风险与对策（必须先做只读预检）
backfill 只信任本地 `chunks.json`，**完全不查 Milvus**。它能否真正修好这 20 个 uid，取决于本地切片是否已是这 3 本书"重新入库后"的当前版本：
- 本地切片为最新 → 算出的新 uid 与当前 Milvus 一致 → 回填成功；
- 本地切片仍是旧的，或重导入后内容/分段变化导致 evidence 子串匹配不上 → 复现旧 uid 或标 `unmatched`，等于白跑，甚至把已有效的 uid 改坏。

因此**写回前必须先做只读预检**：确认 `output/`、`source/` 下这 3 本书的 `chunks.json` 存在且反映重导入后的当前版本，并抽样用与 `find_chunk` 一致的逻辑验证某条 golden `evidence` 能在本地块中命中，预判回填成功率。预检与回填后验证均为只读，可在确认前/后安全执行；唯有写回 `dataset.json` 一步为状态变更，需本计划确认后方可执行。

### 回填后验证
1. 重跑 `lint_dataset.py`：Tier2 直连 Milvus 比对每个 golden `chunk_uid`，warnings 应从 20 降为 0；
2. 可选跑 `run_eval.py --no-gen --limit 3`：确认这 3 本书的 case 回到 `exact` 模式且 recall=1.0；
3. 若仍有 `unmatched`，说明对应书本地切片缺失/过期，需先重新入库或更新本地切片再重跑。

### 安全与回滚
脚本已内置备份 `dataset.json.pre_chunkuid.bak`；若回填结果异常，可直接用备份恢复，不影响知识库（回填仅改测试集 JSON，不触碰 Milvus / MongoDB）。

## 涉及文件
- `evaluation/backfill_chunk_uid.py`（待执行，写 dataset.json）
- `evaluation/dataset.json`（被修改，有自动备份）
- `evaluation/lint_dataset.py`（Tier2 直连 Milvus 校验 uid）
- `app/rag/import_/split_service.py`（`compute_chunk_uid`）
- 本地切片：`output/**/chunks.json`、`source/**/chunks.json`


## Agent Extensions
### SubAgent
- **code-explorer**
  - 用途：在 `output/` 与 `source/` 下跨多文件、多目录定位并抽查这 3 本书（撒哈拉的故事 / 活着 / 蛤蟆先生去看心理医生）的 `chunks.json`，确认本地切片为当前重导入版本，且 golden `evidence` 子串可在本地块中命中（与 `backfill_chunk_uid.py` 的 `find_chunk` 逻辑一致）。
  - 期望产出：3 本书 `chunks.json` 路径清单 + 抽样命中结论，作为写回前只读预检依据，判断回填能否成功。
