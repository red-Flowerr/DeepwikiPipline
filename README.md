# Pipline v1

## 特性概览

pipeline v1 经过多轮迭代，目前具备以下能力：

1. **仓库克隆与回收自动化**  
   - 根据 `--repo-commit` 自动克隆 GitHub 仓库，支持短 SHA、标签或分支；若未指定则拉取默认分支。  
   - 运行完成后删除临时目录，无需手动准备 `repo_root`。克隆过程中若网络或授权失败，会抛出带 stderr 的 `MCPError` 便于排查。  
   - 使用 `git clone --filter=blob:none` 减少带宽，并直接 `git checkout <commit>`，避免 “couldn’t find remote ref” 一类问题。

2. **索引解析与源码回填**  
   - 解析 `[path#Lx-Ly]()`、普通 Markdown 链接（含 README、RST 等）以及 `Sources:`/列表条目中的引用。  
   - 默认将引用替换成真实源码片段（带 fenced code block），同时保留 `reference` → `code` 的映射。  
   - 自动过滤异常字符、URL、HTML 标签及中文路径，防止出现 “File name too long” 等错误。

3. **叙述生成策略**  
   - 逻辑 LLM 改写遵循 WHY → HOW → CONTRACT，输出教科书式段落；禁止列表、表格、ASCII 图、fenced code block 等格式噪音。  
   - Prompt 明确“扩写而非压缩”，在保持忠实的前提下补充设计动机、约束及集成细节。  
   - Critic LLM 校验结构是否严密，支持多轮 refinement；即便无 critic 模型也返回占位信息，保证流程完整。

4. **代码索引管理与拼接支持**  
   - `SubsectionResult.code_blocks` 存储 `{reference, code}`，方便后续按索引回填 narrative 或统计覆盖率。  
   - 附带 `utils/merge_narrative_code.py` 可将 narrative 中的引用替换为真实代码，并输出匹配统计（默认忽略 README 类引用）。

5. **并行与重试控制**  
   - 页面级处理通过 `ThreadPoolExecutor` 并行化，可用 `--max-workers` 控制线程数；默认 `min(32, 页面数)`。  
   - vLLM 请求支持超时、重试、退避和目标服务配置，便于在不同环境下稳定运行。

6. **数据清洗与增量输出**  
   - `deepwiki_pipeline/data_clean` 提供 URL/HTML 清洗脚本，防止脏数据进入改写流程。  
   - CLI 可在执行中将中间结果写入文本或 JSON，断点续跑时可直接接续生成。  
   - 其他工具如 `token_count_local.py`、`hydrate_sections.py` 方便统计 token、离线水化或调试。

7. **多仓批量并发生成**  
   - `--generate-dataset` 支持重复传参、逗号分隔或 `@repo_list.txt` 形式批量指定多个仓库。  
   - 使用 `--output-dir`、`--narrative-output-dir` 为每个仓库生成独立结果文件，文件名自动追加仓库名与提交号片段。  
   - 可通过 `--repo-workers` 控制仓库级并发度，默认取 `min(仓库数, CPU 核心数)`，每个仓库独立维护 MCP 会话与克隆目录。  
   - `--repo-batch-size` 可限制每个批次的仓库数量，例如大规模列表可以按 64 个一组顺序推进，避免资源峰值冲击。  

# Case
Tencent/ncnn  99ecca
volcengine/verl 809ae5

# Pipline
python deepwiki_mcp_client.py \
  --generate-dataset volcengine/verl \
  --repo-commit 809ae5 \
  --output result_data/verl_deepwiki.txt \
  --output-format text \
  --narrative-output result_data/verl_narratives.json \
  --narrative-format json \
  --narrative-modes code critic \
  --design-use-vllm \
  --design-vllm-server-url http://127.0.0.1:8801/v1/chat/completions \
  --design-vllm-model gpt-oss-120b \
  --design-vllm-temperature 0.7 \
  --judge-use-llm \
  --judge-vllm-server-url http://127.0.0.1:8801/v1/chat/completions \
  --judge-vllm-model gpt-oss-120b \
  --judge-vllm-temperature 0.2 \
  --judge-max-rounds 1 \
  --log-level INFO \
  --max-workers 4

## 批量运行示例

```bash
python deepwiki_mcp_client.py \
    --generate-dataset @repos.txt \
    --output-dir result_data/batch_outputs \
    --output-format text \
    --narrative-output-dir result_data/batch_narratives \
    --narrative-format json \
    --narrative-modes code critic \
    --design-use-vllm \
    --design-vllm-server-url http://127.0.0.1:8801/v1/chat/completions \
    --design-vllm-model gpt-oss-120b \
    --design-vllm-temperature 0.7 \
    --judge-use-llm \
    --judge-vllm-server-url http://127.0.0.1:8801/v1/chat/completions \
    --judge-vllm-model gpt-oss-120b \
    --judge-vllm-temperature 0.2 \
    --judge-max-rounds 1 \
    --repo-workers 4 \
    --repo-batch-size 64 \
    --max-workers 4 \
    --log-level INFO
```
--repo-workers 4：控制每个批次中最多同时处理几个仓库（也就是跨仓库的线程池规模）
--repo-batch-size 64：把总体仓库列表切成每批最多 64 个，逐批顺序执行，避免一次性启动太多仓库
--max-workers 4：只影响单个仓库内部的页面级并发度（DeepWikiPipeline的线程池）

# 提取数据

# 处理原始deepwiki脚本
python mcp_tool/hydrate_sections.py volcengine/verl \
  --repo-commit 809ae5 \
  --output result_data/verl_hydrated.txt

# 统计token量
python token_count_local.py \
  --text_path result_data/verl_hydrated_clean.txt \
  --tokenizer-path /mnt/hdfs/tiktok_aiic_new/user/codeai/hf_models/Qwen2.5-32B-Instruct \
  --add-special-tokens
