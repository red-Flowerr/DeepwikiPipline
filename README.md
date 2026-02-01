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
8. **LiteLLM 多 server 分发（可选）**  
   - 通过 `--design-vllm-server-urls` / `--judge-vllm-server-urls` 传入逗号分隔或重复参数形式的地址列表，自动使用 LiteLLM Router 做负载均衡。  
   - 需要 `pip install litellm`，未指定列表时会沿用单个 `--*-server-url` 作为兜底节点。  

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
  --design-vllm-server-urls http://[2605:340:cd51:7700:d0e8:b2ba:f474:56d6]:8802/v1/chat/completions,http://[2605:340:cd51:7700:c1d:467e:ab2f:7edb]:8802/v1/chat/completions,http://[2605:340:cd51:7700:578f:acc4:bceb:432f]:8802/v1/chat/completions,http://[2605:340:cd51:7700:312e:6895:c534:7eb3]:8802/v1/chat/completions \
  --design-vllm-model gpt-oss-120b \
  --design-vllm-temperature 0.7 \
  --judge-use-llm \
  --judge-vllm-server-urls http://[2605:340:cd51:7700:d0e8:b2ba:f474:56d6]:8802/v1/chat/completions,http://[2605:340:cd51:7700:c1d:467e:ab2f:7edb]:8802/v1/chat/completions,http://[2605:340:cd51:7700:578f:acc4:bceb:432f]:8802/v1/chat/completions,http://[2605:340:cd51:7700:312e:6895:c534:7eb3]:8802/v1/chat/completions \
  --judge-vllm-model gpt-oss-120b \
  --judge-vllm-temperature 0.2 \
  --judge-max-rounds 1 \
  --log-level INFO \
  --max-workers 4

## 批量运行示例


"http://[2605:340:cd51:7700:5ae2:c8f3:778:a9d0]:8000/v1/chat/completions,http://[2605:340:cd51:7700:286a:3a59:f4f2:b989]:8000/v1/chat/completions,http://[2605:340:cd51:7700:3930:80b8:8d05:f633]:8000/v1/chat/completions,http://[2605:340:cd51:7700:77b0:be0f:674c:6fe0]:8000/v1/chat/completions,http://[2605:340:cd51:7700:13a8:94f8:597d:e509]:8000/v1/chat/completions,http://[2605:340:cd51:7700:3e72:4aa3:98c9:48e5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:f233:7336:e8dd:6fbf]:8000/v1/chat/completions,http://[2605:340:cd51:7700:97c6:18fe:be8d:f5d]:8000/v1/chat/completions,http://[2605:340:cd51:7700:631d:5d4a:832f:5d27]:8000/v1/chat/completions,http://[2605:340:cd51:7700:98af:4a95:177c:10c4]:8000/v1/chat/completions,http://[2605:340:cd51:7700:256:e584:4eb6:f6ea]:8000/v1/chat/completions,http://[2605:340:cd51:7700:58d7:64bd:e047:d7a]:8000/v1/chat/completions"

```bash

python deepwiki_mcp_client.py \
   --parquet-all \
   --parquet-input-dir /mnt/hdfs/userx/shanyong/code/code_wiki/deepwiki \
   --parquet-scan-batch-size 128 \
   --repo-mp-workers 1 \
   --repo-workers 1 \
   --max-workers 1 \
   --section-workers 1 \
   --repo-cache-dir /tmp/deepwiki_repo_cache \
   --repo-cache-cleanup on-success \
   --skip-existing \
   --narrative-output-dir result_data/batch_narratives \
   --narrative-format json \
   --narrative-modes code critic \
   --design-use-vllm \
   --design-vllm-server-urls  \
   --design-vllm-model gpt-oss-120b \
   --design-vllm-temperature 0.7 \
   --judge-use-llm \
   --judge-vllm-server-urls "http://[2605:340:cd51:7700:48a3:70f4:7d3c:a57]:8000/v1/chat/completions,http://[2605:340:cd51:7700:3dbc:b7c3:dd35:8009]:8000/v1/chat/completions,http://[2605:340:cd51:7700:48a3:70f4:7d3c:a57]:8000/v1/chat/completions,http://[2605:340:cd51:7700:ed84:8d30:c89c:ca0a]:8000/v1/chat/completions,http://[2605:340:cd51:7700:ed84:8d30:c89c:ca0a]:8000/v1/chat/completions,http://[2605:340:cd51:7700:8978:4517:4283:273a]:8000/v1/chat/completions,http://[2605:340:cd51:7700:eb97:8e80:414c:c3f5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:eb97:8e80:414c:c3f5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:3dbc:b7c3:dd35:8009]:8000/v1/chat/completions,http://[2605:340:cd51:7700:cc59:6632:c1ae:8f1e]:8000/v1/chat/completions,http://[2605:340:cd51:7700:56f1:b739:c035:c85]:8000/v1/chat/completions,http://[2605:340:cd51:7700:b19d:b7c:f434:1f1c]:8000/v1/chat/completions" \
   --judge-vllm-model gpt-oss-120b \
   --judge-vllm-temperature 0.2 \
   --judge-max-rounds 1 \
   --continue-on-error \
   --hdfs-output-dir "hdfs://harunawl/home/byte_data_seed_wl/user/xingtianshun/deepwiki_data" \
   --log-level INFO


python deepwiki_mcp_client.py \
   --parquet-all \
   --parquet-input-dir /mnt/hdfs/userx/shanyong/code/code_wiki/deepwiki \
   --parquet-scan-batch-size 128 \
   --repo-mp-workers 96 \
   --repo-workers 1 \
   --max-workers 4 \
   --section-workers 4 \
   --repo-cache-dir /tmp/deepwiki_repo_cache \
   --repo-cache-cleanup on-success \
   --narrative-output-dir result_data/batch_narratives \
   --narrative-format json \
   --narrative-modes code critic \
   --design-use-vllm \
   --design-vllm-server-urls "http://[2605:340:cd51:7700:2398:bba5:7251:9a79]:8000/v1/chat/completions,http://[2605:340:cd51:7700:2dbb:7c35:26c9:bf7c]:8000/v1/chat/completions,http://[2605:340:cd51:7700:b7a:fd01:3787:7c89]:8000/v1/chat/completions,http://[2605:340:cd51:7700:eae3:c02a:fa98:af77]:8000/v1/chat/completions,http://[2605:340:cd51:7700:5fdc:9a08:e20a:7bfe]:8000/v1/chat/completions,http://[2605:340:cd51:7700:c532:8755:d45b:1067]:8000/v1/chat/completions,http://[2605:340:cd51:7700:fc3a:b90:519c:44a3]:8000/v1/chat/completions,http://[2605:340:cd51:7700:1796:e44f:d18c:7194]:8000/v1/chat/completions,http://[2605:340:cd51:7700:105:9f99:39dd:363a]:8000/v1/chat/completions,http://[2605:340:cd51:7700:65ee:2dfa:ca7a:a0c1]:8000/v1/chat/completions,http://[2605:340:cd51:7700:1286:adb6:1ec6:3dc9]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6db7:7ec3:d7c3:c3d3]:8000/v1/chat/completions" \
   --design-vllm-model gpt-oss-120b \
   --design-vllm-temperature 0.7 \
   --continue-on-error \
   --hdfs-output-dir "hdfs://harunawl/home/byte_data_seed_wl/user/xingtianshun/deepwiki_data" \
   --skip-existing \
   --log-level INFO
```
--skip-existing \
说明：当同时设置 `--skip-existing` 和 `--hdfs-output-dir` 时，会优先检查 HDFS 上是否已存在该 repo 对应的输出子目录；若目录已存在则直接跳过该 repo 的生成与上传（实现上会先对 `--hdfs-output-dir` 做一次非递归索引，避免每个 repo 都频繁调用 HDFS 命令）。

说明：当使用 server pooling（`--*-vllm-server-urls`）时，会对 endpoints 做去重；请求侧使用 round-robin（每次请求轮换起始 endpoint）来让所有节点参与分摊，并对连接/超时等瞬态失败的节点做短暂冷却剔除，避免持续打到坏节点；`--*-vllm-retries/--*-vllm-retry-backoff/--*-vllm-timeout` 也会应用到 pooling 调用路径。

多 worker 并发：如果你启动多个互相独立的 worker 实例（没有共享队列/状态），可以用 `--shard-count/--shard-index` 对 repo 做确定性分片，避免重复处理同一批数据。例如启动 8 个 worker：
- worker0: `--shard-count 8 --shard-index 0`
- worker1: `--shard-count 8 --shard-index 1`
- ...
--repo-workers 4：控制每个批次中最多同时处理几个仓库（也就是跨仓库的线程池规模）
--repo-batch-size 64：把总体仓库列表切成每批最多 64 个，逐批顺序执行，避免一次性启动太多仓库
--max-workers 4：只影响单个仓库内部的页面级并发度（DeepWikiPipeline的线程池）
--section-workers 8：在单个页面内部把 section 并行化（可选；默认每页串行），适合“页少但 section 很多”的仓库
--disable-hydration：关闭源码引用回填（faster，且避免大规模并发时被磁盘/网络盘 I/O 卡住）
--hydration-timeout 0.5：每个 section 的回填最多耗时 0.5s，超时则跳过回填继续跑
--hydration-workers 4：每个 repo 内回填任务的最大并发（配合 section-workers 防止 I/O 放大）

# 提取数据

# 处理原始deepwiki脚本
python mcp_tool/hydrate_sections.py volcengine/verl \
  --repo-commit 809ae5 \
  --output result_data/verl_hydrated.txt




# 测试vllm连通性
python utils/vllm_load_test.py \
   --urls "http://[ipv6]:8000/v1/chat/completions" \
   --model gpt-oss-120b \
   --prompt "Summarize: hello world." \
   --max-tokens 131072 \
   --temperature 0.2 \
   --warmup 5 \
   --requests 50 \
   --concurrency 16


# 统计原始有多少wiki token量
python utils/multipro_parquet_token.py \
   --base /mnt/hdfs/userx/shanyong/code/code_wiki/deepwiki \
   --column content \
   --workers 128 \
   --mp-start spawn

# 30min 200repo
# 1h 400repo


# 统计现在跑了多少token量
python utils/multipro_raw_token.py \
   --mode deepwiki_data \
   --base /mnt/hdfs/user_wl/xingtianshun/deepwiki_data \
   --workers 16 --mp-start spawn


python utils/multipro_raw_token.py \
   --mode deepwiki_data \
   --base /mnt/hdfs/user_wl/xingtianshun/deepwiki_data \
   --workers 16 --mp-start spawn \
   --tokenizer-backend hf \
   --hf-model /opt/tiger/oss_server_only/Qwen3-8B


python utils/ipv6_to_urls.py <<'EOF'
2605:340:cd51:7700:2398:bba5:7251:9a79
2605:340:cd51:7700:2dbb:7c35:26c9:bf7c
2605:340:cd51:7700:b7a:fd01:3787:7c89
2605:340:cd51:7700:eae3:c02a:fa98:af77
2605:340:cd51:7700:5fdc:9a08:e20a:7bfe
2605:340:cd51:7700:c532:8755:d45b:1067
2605:340:cd51:7700:fc3a:b90:519c:44a3
2605:340:cd51:7700:1796:e44f:d18c:7194
2605:340:cd51:7700:105:9f99:39dd:363a
2605:340:cd51:7700:65ee:2dfa:ca7a:a0c1
2605:340:cd51:7700:1286:adb6:1ec6:3dc9
2605:340:cd51:7700:6db7:7ec3:d7c3:c3d3
EOF


python utils/token_count_indexed_code.py \
   --narratives-dir result_data/batch_narratives \
   --parquet-dir /mnt/hdfs/userx/shanyong/code/code_wiki/deepwiki \
   --cache-dir /tmp/deepwiki_repo_cache \
   --hdfs-bin hdfs \
   --encoding cl100k_base \
   --repo-workers 64 \
   > token_count_indexed_code.log 2>&1 &


python utils/extract_repo_hdfs_map.py \
--narratives-dir result_data/batch_narratives \
--parquet-dir /mnt/hdfs/userx/shanyong/code/code_wiki/deepwiki \
--output result_data/repo_hdfs_map.json


python utils/token_count_from_repo_indices.py \
   --repo-indices /opt/tiger/oss_server_only/DeepwikiPipline/result_data/repo_indices.json \
   --repo-hdfs-map /opt/tiger/oss_server_only/DeepwikiPipline/result_data/repo_hdfs_map.json \
   --cache-dir /tmp/deepwiki_repo_cache \
   --hdfs-bin hdfs \
   --encoding cl100k_base \
   --repo-workers 16 \
   --progress

# 交付数据

python utils/export_narratives_to_parquet.py \
   --base /mnt/hdfs/user_wl/xingtianshun/deepwiki_data \
   --output /mnt/hdfs/user_wl/xingtianshun/deepwiki_handover/4w_wiki_repo_level_narratives_12B.parquet \
   --workers 16 --mp-start spawn