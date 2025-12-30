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
7. **Post-train 质量验证导出**  
   - CLI 支持 `--post-train-output` 将每个章节打包为 conversation（system/user/assistant），可直接送入后训练/评测流程。  
   - conversation 自动包含 narrative、critic 以及代码引用，便于构建参考答案和快速判分。
8. **SFT 指令对生成**  
   - 开启 `--qa-use-vllm` 可在 Narrative 基础上自动生成多类型 QA（概念/用法/约束），并通过 `--sft-output` 导出标准的 instruction-response 样本。  
   - 若不启用 QA LLM，也会回落为“解释类”指令对，`narrative` 即为高质量答案。
   - 默认系统 Prompt 位于 `prompts/sft_qa_system.txt`，用户模板位于 `prompts/sft_qa_user.txt`，可分别通过 `--qa-system-prompt @<path>`、`--qa-user-prompt @<path>` 自定义。
9. **码元 SFT 构造**  
   - `--code-explain-output` 自动生成【代码 → 设计解读】指令对，默认指令模板 `prompts/code_explain_instruction.txt`。  
   - `--code-gen-output` 自动生成【设计描述 → 代码】指令对，默认指令模板 `prompts/code_generate_instruction.txt`，无需额外 LLM。

# Case
Tencent/ncnn  99ecca
volcengine/verl 809ae5

# Pipline
python deepwiki_mcp_client.py \
  --generate-dataset volcengine/verl \
  --repo-commit 809ae5 \
  --output result_data/verl_deepwiki.txt \
  --output-format text \
  --post-train-output result_data/verl_post_train.jsonl \
  --sft-output result_data/verl_sft.jsonl \
  --code-explain-output result_data/verl_code_explain.jsonl \
  --code-gen-output result_data/verl_code_generate.jsonl \
  --qa-use-vllm \
  --qa-vllm-server-url http://[fdbd:dccd:cdd2:2101::1c4]:8000/v1/chat/completions \
  --qa-vllm-model gpt-oss-20b \
  --qa-system-prompt @prompts/sft_qa_system.txt \
  --qa-user-prompt @prompts/sft_qa_user.txt \
  --narrative-output result_data/verl_narratives.json \
  --narrative-format json \
  --narrative-modes code critic \
  --design-use-vllm \
  --design-vllm-server-url http://[fdbd:dccd:cdd2:2101::1c4]:8000/v1/chat/completions \
  --design-vllm-model gpt-oss-20b \
  --design-vllm-temperature 0.2 \
  --judge-use-llm \
  --judge-vllm-server-url http://[fdbd:dccd:cdd2:2101::1c4]:8000/v1/chat/completions \
  --judge-vllm-model gpt-oss-20b \
  --judge-vllm-temperature 0.0 \
  --judge-max-rounds 1 \
  --log-level INFO \
  --max-workers 16

# 合并数据
python utils/merge_narrative_code.py \
  --input result_data/verl_narratives.json \
  --output result_data/verl_narratives_merged.txt
# 导出 SFT instruction-response（自动生成 QA）
python deepwiki_mcp_client.py \
  --generate-dataset volcengine/verl \
  --repo-commit 809ae5 \
  --qa-use-vllm \
  --qa-vllm-model gpt-oss-20b \
  --qa-system-prompt @prompts/sft_qa_system.txt \
  --qa-user-prompt @prompts/sft_qa_user.txt \
  --sft-output result_data/verl_sft.jsonl
# 导出 Code Explanation 指令对
python deepwiki_mcp_client.py \
  --generate-dataset volcengine/verl \
  --repo-commit 809ae5 \
  --code-explain-output result_data/verl_code_explain.jsonl
# 导出 Code Generation 指令对
python deepwiki_mcp_client.py \
  --generate-dataset volcengine/verl \
  --repo-commit 809ae5 \
  --code-gen-output result_data/verl_code_generate.jsonl
# 导出 Post-train conversation（可直接投喂 SFT/RL 评测）
python deepwiki_mcp_client.py \
  --generate-dataset volcengine/verl \
  --repo-commit 809ae5 \
  --post-train-output result_data/verl_post_train.jsonl
# 处理原始deepwiki脚本
python mcp_tool/hydrate_sections.py volcengine/verl \
  --repo-commit 809ae5 \
  --output result_data/verl_hydrated.txt

# 统计token量
python token_count_local.py \
  --text_path result_data/verl_hydrated_o.txt \
  --tokenizer-path /mnt/hdfs/tiktok_aiic_new/user/codeai/hf_models/Qwen2.5-32B-Instruct \
  --add-special-tokens
