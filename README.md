# DeepWiki Pipeline

基于 DeepWiki MCP 服务，对 GitHub 仓库的 wiki 文档进行源码回填、LLM 叙述改写（narrative）并批量产出结构化数据集。

---

## 目录

- [项目结构](#项目结构)
- [数据流概览](#数据流概览)
- [依赖](#依赖)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
  - [1. 批量生成 Narrative](#1-批量生成-narrative)
  - [2. 提取与处理原始 DeepWiki 数据](#2-提取与处理原始-deepwiki-数据)
  - [3. Token 统计](#3-token-统计)
  - [4. 数据交付与导出](#4-数据交付与导出)
- [工具脚本速查](#工具脚本速查)
  - [主流程](#主流程)
  - [Token 统计类](#token-统计类)
  - [数据导出类](#数据导出类)
  - [辅助工具](#辅助工具)
- [输出数据 Schema](#输出数据-schema)

---

## 项目结构

```
DeepwikiPipline/
├── deepwiki_mcp_client.py            # 主入口 CLI
├── deepwiki_narratives.py            # Narrative 生成逻辑封装
├── vllm_client.py                    # vLLM 请求客户端
├── token_count_local.py              # 单文件 token 统计（调试用）
├── export_narratives.py              # 单 repo narrative 导出
├── update_hf.py                      # HuggingFace Hub 上传
│
├── deepwiki_pipeline/                # 核心 pipeline 包
│   ├── pipeline.py                   #   Pipeline 编排
│   ├── narrative.py                  #   LLM 叙述生成
│   ├── hydration.py                  #   源码片段回填
│   ├── parsing.py                    #   Markdown / Outline 解析
│   ├── models.py                     #   数据模型 (PipelineOutput, SubsectionResult …)
│   ├── mcp.py                        #   MCP 会话与 tool 调用
│   └── data_clean/                   #   URL / HTML 清洗
│       └── clean_hydrated.py
│
├── mcp_tool/                         # MCP 工具脚本（可独立运行）
│   ├── ask.py                        #   ask_question
│   ├── contents.py                   #   read_wiki_contents
│   ├── structure.py                  #   read_wiki_structure
│   ├── structure_clean.py            #   清洗后的 outline 下载
│   ├── fetch_wiki_chunk.py           #   获取指定 page/section
│   └── hydrate_sections.py           #   离线水化 → 文本导出
│
├── utils/                            # 工具脚本集
│   ├── multipro_raw_token.py         #   多进程统计 narrative / deepwiki token
│   ├── multipro_parquet_token.py     #   多进程统计 parquet content 列 token
│   ├── multipro_repo_zip_token.py    #   多进程统计 repo zip token
│   ├── token_count_parquet_by_repo_map.py  # 按 repo_map 过滤统计 parquet token
│   ├── token_count_indexed_code.py   #   统计 narrative 中引用代码的 token
│   ├── token_count_from_repo_indices.py    # 从 repo_indices 统计代码 token
│   ├── export_narratives_to_parquet.py     # narratives.json → 单 parquet（含 token）
│   ├── add_narrative_tokens_to_parquet.py  # 给 parquet 追加 narrative_tokens 列
│   ├── export_repo_tokens_jsonl.py   #   parquet → repo:tokens JSONL
│   ├── export_token_counts.py        #   多进程 token 统计 + 输出
│   ├── extract_repo_hdfs_map.py      #   构建 repo→hdfs_path 映射
│   ├── extract_repo_index.py         #   提取 Sources 引用索引
│   ├── merge_narrative_code.py       #   narrative 引用替换为源码
│   ├── concat_narratives.py          #   多 narrative JSON 拼接
│   ├── inspect_narratives.py         #   检查 narrative 质量 (pass率/违规/critic)
│   ├── find_index.py                 #   检查 Sources 引用
│   ├── read_deepwiki_parquet.py      #   读取 deepwiki parquet → JSONL
│   ├── vllm_load_test.py             #   vLLM 压测工具
│   └── ipv6_to_urls.py               #   IPv6 → vLLM URL 批量转换
│
├── prompts/                          # Prompt 模板
│   ├── system_prompt.txt             #   SWE-bench annotator schema
│   ├── judge_strict.txt              #   Judge JSON schema
│   ├── repo_analysis_template.txt
│   └── repo_analysis_prompt.txt
│
├── repos.txt                         # 仓库列表示例
└── result_data/                      # 默认输出目录
```

---

## 数据流概览

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│  Parquet (content+hdfs_path)  或  --generate-dataset owner/repo │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  deepwiki_mcp_client.py                           │
│  1. 克隆仓库 / 从 HDFS 解压 repo zip                             │
│  2. MCP read_wiki_structure → Outline                            │
│  3. MCP read_wiki_contents → Markdown 页面                       │
│  4. 源码回填 (hydration): 引用 → fenced code block               │
│  5. Design LLM 改写 → narrative (WHY→HOW→CONTRACT)              │
│  6. Critic LLM 校验 → 多轮 refinement                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                        OUTPUT (per repo)                         │
│  {repo}_deepwiki.txt / .json   ← 完整 wiki + 源码               │
│  {repo}_narratives.json        ← section 级 narrative 列表       │
│  上传至 --hdfs-output-dir (可选)                                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      后处理 (utils/)                              │
│  Token 统计  →  数据导出 Parquet  →  追加 token 列  →  JSONL     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 依赖

> 项目暂无 `requirements.txt`，以下为从代码推断的核心依赖：

| 包 | 用途 |
|---|------|
| `pyarrow` | Parquet 读写 |
| `tiktoken` | Token 计数（默认 `cl100k_base`） |
| `requests` | HTTP 请求 |
| `tqdm` | 进度条 |
| `transformers` | 可选：HuggingFace tokenizer |
| `litellm` | 可选：多 vLLM server 负载均衡 |
| `huggingface_hub` | 可选：上传 HF Hub |

---

## 核心特性

1. **仓库克隆与回收自动化**
   - 根据 `--repo-commit` 自动 `git clone --filter=blob:none`，支持短 SHA / 标签 / 分支。
   - 运行完成后自动删除临时目录；网络或授权失败抛出带 stderr 的 `MCPError`。

2. **索引解析与源码回填**
   - 解析 `[path#Lx-Ly]()`、Markdown 链接、`Sources:` 列表等引用。
   - 将引用替换为真实源码片段（fenced code block），同时保留 `reference → code` 映射。
   - 自动过滤异常字符、URL、HTML 标签及中文路径。

3. **叙述生成策略**
   - Design LLM 遵循 **WHY → HOW → CONTRACT** 输出教科书式段落；禁止列表、表格、ASCII 图等格式噪音。
   - Prompt "扩写而非压缩"，补充设计动机、约束及集成细节。
   - Critic LLM 校验结构，支持多轮 refinement。

4. **并行与重试控制**
   - 页面级 `ThreadPoolExecutor` 并行（`--max-workers`）。
   - 仓库级 `--repo-workers` / `--repo-mp-workers` 并发。
   - `--repo-batch-size` 限制每批仓库数，避免资源峰值。
   - vLLM 请求支持超时、重试、退避。

5. **LiteLLM 多 server 分发（可选）**
   - `--design-vllm-server-urls` / `--judge-vllm-server-urls` 传入逗号分隔地址列表，自动 LiteLLM Router 负载均衡。

6. **数据清洗与增量输出**
   - `deepwiki_pipeline/data_clean` 提供 URL/HTML 清洗。
   - `--skip-existing` + `--continue-on-error` 支持断点续跑。
   - 中间结果实时写入，可增量接续。

---

## 快速开始

### 1. 批量生成 Narrative

从 Parquet 批量读取仓库，生成 narrative 并上传 HDFS：

```bash
export DEEPWIKI_VLLM_OUTAGE_THRESHOLD=30

python deepwiki_mcp_client.py \
  --parquet-all \
  --parquet-input-dir <parquet_dir> \
  --parquet-scan-batch-size 128 \
  --repo-mp-workers 96 \
  --repo-workers 1 \
  --max-workers 8 \
  --section-workers 8 \
  --repo-cache-dir /tmp/deepwiki_repo_cache \
  --repo-cache-cleanup on-success \
  --narrative-output-dir result_data/batch_narratives \
  --narrative-format json \
  --narrative-modes code critic \
  --design-use-vllm \
  --design-vllm-server-urls "<url1>,<url2>,..." \
  --design-vllm-model gpt-oss-120b \
  --design-vllm-temperature 0.7 \
  --continue-on-error \
  --hdfs-output-dir "hdfs://harunawl/home/byte_data_seed_wl/user/<username>/deepwiki_data" \
  --skip-existing \
  --log-level INFO \
  --shard-count 5 \
  --shard-index 0 \
  --shard-progress-total
```

> **提示**：可用 `utils/ipv6_to_urls.py` 批量将 IPv6 地址转为 vLLM URL：
>
> ```bash
> python utils/ipv6_to_urls.py <<'EOF'
> 2605:340:cd51:a01:5f43:589a:ed6c:4350
> 2605:340:cd51:a01:62c0:d74d:4ef0:6395
> EOF
> ```

### 2. 提取与处理原始 DeepWiki 数据

水化（hydrate）原始 DeepWiki 输出，将引用替换为真实源码：

```bash
python mcp_tool/hydrate_sections.py <owner/repo> \
  --repo-commit <commit_sha> \
  --output result_data/<repo>_hydrated.txt
```

### 3. Token 统计

#### 3.1 统计 narrative 产物的 token 量

扫描 `deepwiki_data/` 目录下每个 repo 的 `*_narratives.json` 和 `*_deepwiki.txt`：

```bash
python utils/multipro_raw_token.py \
  --mode deepwiki_data \
  --base <deepwiki_data_dir> \
  --workers 16 --mp-start spawn
```

使用 HuggingFace tokenizer（如 Qwen3-8B）：

```bash
python utils/multipro_raw_token.py \
  --mode deepwiki_data \
  --base <deepwiki_data_dir> \
  --workers 16 --mp-start spawn \
  --tokenizer-backend hf \
  --hf-model <model_path>
```

#### 3.2 统计原始 Parquet 中的 wiki token 量

统计原始 DeepWiki parquet 中 `content` 列的 token：

```bash
python utils/multipro_parquet_token.py \
  --base <parquet_dir> \
  --column content \
  --workers 128 --mp-start spawn
```

#### 3.3 按 repo 列表过滤统计 Parquet token

只统计指定仓库列表内的 token：

```bash
python utils/token_count_parquet_by_repo_map.py \
  --parquet-base <parquet_dir> \
  --repo-map result_data/repo_hdfs_map.json \
  --workers 16 --mp-start spawn
```

#### 3.4 统计 narrative 中引用代码的 token

```bash
python utils/token_count_indexed_code.py \
  --narratives-dir result_data/batch_narratives \
  --parquet-dir <parquet_dir> \
  --cache-dir /tmp/deepwiki_repo_cache \
  --hdfs-bin hdfs \
  --encoding cl100k_base \
  --repo-workers 64
```

#### 3.5 从 repo_indices 统计代码 token

```bash
python utils/token_count_from_repo_indices.py \
  --repo-indices result_data/repo_indices.json \
  --repo-hdfs-map result_data/repo_hdfs_map.json \
  --cache-dir /tmp/deepwiki_repo_cache \
  --hdfs-bin hdfs \
  --encoding cl100k_base \
  --repo-workers 16 \
  --progress
```

### 4. 数据交付与导出

#### 4.1 导出 narrative 到 Parquet（repo 级拼接）

将每个 repo 的 `*_narratives.json` 中的 section 用 `\n\n` 拼接成一条完整 narrative，输出为 Parquet：

```bash
python utils/export_narratives_to_parquet.py \
  --base <deepwiki_data_dir> \
  --output <output_path>/repo_level_narratives.parquet \
  --workers 16 --mp-start spawn
```

#### 4.1.1 导出 narrative 到 Parquet（极速版：不算 token）

对于超长 narrative（例如单条 ~1M token），token 计数会成为主要耗时。建议先落盘 narrative，再做离线 token 统计与分桶：

```bash
python utils/export_narratives_fast.py \
  --base <deepwiki_data_dir> \
  --output <output_path>/repo_level_narratives.parquet \
  --workers 16 \
  --mp-start forkserver \
  --rows-per-shard 5000 \
  --batch-size 8 \
  --task-timeout 0 \
  --max-file-mb 0 \
  --worker-max-vm-mb 0 \
  --resume \
  --queue-maxsize 0
```

输出为多文件：`repo_level_narratives.worker*.part*.parquet`（每个 worker 自己写 shard）。

##### 4.1.1.1 GB 级 *_narratives.json 的处理建议

如果存在单个 `*_narratives.json` 达到 GB 级（例如 2GB），`export_narratives_fast.py` 由于需要整文件 JSON 反序列化 + 拼接，
很容易触发 worker 被系统 OOM-kill（exitcode `-9`）。推荐做法：

- 上游先把大 JSON array 拆成多个小文件（仍以 `_narratives.json` 结尾，export 扫描能识别）：
  - `python utils/split_narratives_json.py --input /path/to/foo_narratives.json --rows-per-part 50000`
  - 或加大小上限：`--max-part-mb 256`
- 导出时先保守并发（例如 `--workers 1` 或 `--workers 2`），并保持 `--resume` 方便断点续跑。

#### 4.2 追加 token 列

在已有 Parquet 上追加 `narrative_tokens` 列（无需重新拼接 narrative）：

```bash
python utils/add_narrative_tokens_to_parquet.py \
  --input <input>.parquet \
  --output <output>.with_tokens.parquet \
  --tokenizer-backend tiktoken --encoding cl100k_base
```

#### 4.3 Tokenize + 分桶（支持断点续跑 + 多进程分片）

推荐用流式脚本直接对 `export_narratives_fast.py` 的多文件输出做 token 计数并分桶（避免一次性加载全量 Parquet）：

```bash
python utils/tokenize_and_bucket_parquet.py \
  --input-glob "<output_path>/repo_level_narratives.worker*.part*.parquet" \
  --output-dir "<output_path>/buckets_0213" \
  --tokenizer-backend tiktoken --encoding cl100k_base \
  --batch-size 1 --write-batch 8 \
  --resume --claim \
  --num-shards 32 --shard-index 0
```

说明：
- 多进程并行：启动 `--shard-index=0..31` 共 32 个进程即可吃满多核。
- 断点续跑：`--resume` 会跳过已完成输入；`--claim` 避免多个进程重复处理同一输入文件。

分桶完成后可一键汇总（按桶统计 rows / tokens_sum）：

```bash
python utils/summarize_bucket_run.py \
  --output-dir "<output_path>/buckets_0213" \
  --write-json "<output_path>/buckets_0213/summary.json"
```

#### 4.2.1 统计 token 并分桶（流式，适合超大数据）

把落盘后的 Parquet shards 流式读入，计算 `narrative_tokens` 并按 bucket 输出到不同目录（不会一次性把全部数据读进内存）：

```bash
python utils/tokenize_and_bucket_parquet.py \
  --input-glob "<output_path>/repo_level_narratives.worker*.part*.parquet" \
  --output-dir "<output_path>/buckets_repo_level_narratives" \
  --tokenizer-backend tiktoken \
  --encoding cl100k_base \
  --batch-size 1 \
  --write-batch 8 \
  --rows-per-shard 200
```

默认 buckets：
- `<128k`
- `128k–512k`
- `512k–2M`

如需保留 `>=2M`，加 `--keep-overflow`（输出到 `over_2M/`）；否则会计入 `skipped_overflow`。

#### 4.3 导出 repo → token 数 JSONL

```bash
python utils/export_repo_tokens_jsonl.py \
  --input <input>.with_tokens.parquet \
  --output repo_narrative_tokens.jsonl
```

---

## 工具脚本速查

### 主流程

| 脚本 | 功能 |
|------|------|
| `deepwiki_mcp_client.py` | 主入口：批量生成 narrative 数据集 |
| `mcp_tool/hydrate_sections.py` | 水化原始 wiki，将引用替换为源码 |
| `utils/merge_narrative_code.py` | 将 narrative 中的引用替换为源码片段 |

### Token 统计类

| 脚本 | 输入 | 统计目标 |
|------|------|---------|
| `utils/multipro_raw_token.py` | `deepwiki_data/` 目录 | `*_deepwiki.txt` + `*_narratives.json` 的 token 总量 |
| `utils/multipro_parquet_token.py` | Parquet 目录 | parquet `content` 列 token |
| `utils/multipro_repo_zip_token.py` | Parquet → HDFS zip | repo 源码 zip 的 token |
| `utils/token_count_parquet_by_repo_map.py` | Parquet + repo_map | 按 repo 列表过滤统计 parquet token |
| `utils/token_count_indexed_code.py` | narratives + parquet | narrative 中引用代码的 token |
| `utils/token_count_from_repo_indices.py` | repo_indices + repo_hdfs_map | 指定 repo 引用代码的 token |
| `utils/export_token_counts.py` | `deepwiki_data/` 目录 | 多进程统计 + 输出文件 |

### 数据导出类

| 脚本 | 功能 |
|------|------|
| `utils/export_narratives_to_parquet.py` | `*_narratives.json` → repo 级 Parquet（含 token 计数） |
| `utils/add_narrative_tokens_to_parquet.py` | 给已有 Parquet 追加 `narrative_tokens` 列 |
| `utils/export_repo_tokens_jsonl.py` | Parquet → repo:tokens JSONL |
| `utils/convert_midtrain_parquet.py` | 训练 parquet → midtrain parquet（重算 `token_count`） |
| `utils/extract_repo_hdfs_map.py` | 构建 `repo_name → hdfs_path` 映射 JSON |
| `utils/extract_repo_index.py` | 提取 Sources 引用索引 → `repo_indices.json` |
| `utils/concat_narratives.py` | 多 narrative JSON 拼接为单文本 |

### 辅助工具

| 脚本 | 功能 |
|------|------|
| `utils/vllm_load_test.py` | vLLM 服务压测（并发、延迟、吞吐） |
| `utils/ipv6_to_urls.py` | IPv6 地址批量转为 vLLM URL |
| `utils/inspect_narratives.py` | 检查 narrative 质量（pass 率 / 违规 / critic 结果） |
| `utils/find_index.py` | 检查 Sources 引用 |
| `utils/read_deepwiki_parquet.py` | 读取 DeepWiki parquet → JSONL 打印 |

---

## 输出数据 Schema

## Midtrain 数据转换

将现有训练 Parquet 转换为 midtrain 所需 Schema，并用指定 tokenizer 重新计算 `token_count`。

输出 Schema：

- `meta`：struct，包含 `docid` / `chunk_id` / `source`
- `content_split`：文本内容
- `token_count`：使用 tokenizer 对 `content_split` 计算得到的 token 数量

示例（按原目录结构输出）：

```bash
python utils/convert_midtrain_parquet.py \
  --input-root /mnt/hdfs/byte_data_seed_wl_write/user/xingtianshun/deepwiki_handover/0213_training_parquet \
  --output-root /mnt/hdfs/byte_data_seed_wl_write/user/xingtianshun/deepwiki_handover/0224_mt_train_parquet \
  --tokenizer /mnt/hdfs/byte_data_seed_wl/user/fangjunjie.99/tokenizers/bbpe155k-v6.4.3-ml.pret_v5.7_20251015 \
  --workers 16 --batch-size 64
```

常用可选参数：

- `--source xxx`：覆盖输出 `meta.source`
- `--subdirs under_128k,128k_512k,512k_2M`：只处理指定子目录
- `--max-files N`：小规模验证用


### `*_narratives.json`（section 级，per repo）

```json
[
  {
    "repo": "owner/repo",
    "page": "Overview",
    "section": "Architecture",
    "original_context": "原始 wiki markdown + Sources 引用",
    "narrative": "LLM 改写后的叙述文本",
    "critic": "Critic LLM 的评审结果",
    "verdict": "aligned / misaligned",
    "code_blocks": [{"reference": "path#L1-L10", "code": "..."}]
  }
]
```

### 导出 Parquet（repo 级，由 `export_narratives_to_parquet.py` 产出）

| 列名 | 类型 | 说明 |
|------|------|------|
| `folder` | string | `deepwiki_data/` 下的子目录名 |
| `filepath` | string | 原始 `*_narratives.json` 的完整路径 |
| `repo` | string | 仓库名（如 `01-ai/Yi`） |
| `narrative` | large_string | repo 级完整 narrative（section 用 `\n\n` 拼接） |
| `narrative_tokens` | int64 | narrative 的 token 数 |
| `rows` | int32 | 当前 repo 的 section 个数 |
| `missing_rows` | int32 | 拼接时被跳过的缺失行数 |
