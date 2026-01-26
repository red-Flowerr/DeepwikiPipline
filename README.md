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

```bash
   








http://[2605:340:cd51:7700:d2a0:3ebe:6ad2:ccb2]:8000/v1/chat/completions,http://[2605:340:cd51:7700:78c7:e555:ac89:5633]:8000/v1/chat/completions,http://[2605:340:cd51:7700:a3ee:7aef:e6a9:10de]:8000/v1/chat/completions,http://[2605:340:cd51:7700:1134:7cdb:9a9e:b6e0]:8000/v1/chat/completions,http://[2605:340:cd51:7700:192e:909d:3448:7e0a]:8000/v1/chat/completions,http://[2605:340:cd51:7700:a442:17d0:4bf:5108]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6873:76a7:1507:fba]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6873:76a7:1507:fba]:8000/v1/chat/completions,http://[2605:340:cd51:7700:ea6f:755d:6615:4dc2]:8000/v1/chat/completions,http://[2605:340:cd51:7700:b205:844b:c560:cc3e]:8000/v1/chat/completions,http://[2605:340:cd51:7700:ac40:6b9:4bac:e956]:8000/v1/chat/completions,http://[2605:340:cd51:7700:a442:17d0:4bf:5108]:8000/v1/chat/completions


http://[2605:340:cd51:7700:e96f:4b63:978a:11e4]:8000/v1/chat/completions,http://[2605:340:cd51:7700:e9ff:7550:86b0:bd8c]:8000/v1/chat/completions,http://[2605:340:cd51:7700:f6e2:dcdb:d569:6fa9]:8000/v1/chat/completions,http://[2605:340:cd51:7700:3758:5cb8:16ec:aea7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:246f:5e8a:a4b:b9a5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:8c0e:31a0:2f3c:93b2]:8000/v1/chat/completions,http://[2605:340:cd51:7700:816e:914b:c9ce:3fc7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:48cc:d0fa:7ee:97d0]:8000/v1/chat/completions,http://[2605:340:cd51:7700:3758:5cb8:16ec:aea7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:227f:3813:5457:7b77]:8000/v1/chat/completions,http://[2605:340:cd51:7700:b7c8:169c:1130:8410]:8000/v1/chat/completions,http://[2605:340:cd51:7700:246f:5e8a:a4b:b9a5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:97:75c4:63eb:7507]:8000/v1/chat/completions,http://[2605:340:cd51:7700:2787:ad8e:6f21:2ae5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:1e37:a428:9cee:677a]:8000/v1/chat/completions,http://[2605:340:cd51:7700:83a:3e2e:9ccd:84fc]:8000/v1/chat/completions,http://[2605:340:cd51:7700:408b:5149:1030:1b93]:8000/v1/chat/completions,http://[2605:340:cd51:7700:808c:c5ef:17f6:ec10]:8000/v1/chat/completions,http://[2605:340:cd51:7700:44fd:a43c:4aaa:be7f]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6542:ae4e:9544:90f7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:48cc:d0fa:7ee:97d0]:8000/v1/chat/completions,http://[2605:340:cd51:7700:808c:c5ef:17f6:ec10]:8000/v1/chat/completions,http://[2605:340:cd51:7700:8dac:3019:cbae:7d61]:8000/v1/chat/completions,http://[2605:340:cd51:7700:83a:3e2e:9ccd:84fc]:8000/v1/chat/completions,http://[2605:340:cd51:7700:e9ff:7550:86b0:bd8c]:8000/v1/chat/completions,http://[2605:340:cd51:7700:daae:23b5:435f:5e38]:8000/v1/chat/completions,http://[2605:340:cd51:7700:816e:914b:c9ce:3fc7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:44fd:a43c:4aaa:be7f]:8000/v1/chat/completions,http://[2605:340:cd51:7700:408b:5149:1030:1b93]:8000/v1/chat/completions,http://[2605:340:cd51:7700:b7c8:169c:1130:8410]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6b11:3d28:f8b2:d047]:8000/v1/chat/completions,http://[2605:340:cd51:7700:1177:b349:28c3:3b84]:8000/v1/chat/completions

python deepwiki_mcp_client.py \
   --parquet-all \
   --narrative-output-dir result_data/batch_narratives \
   --narrative-format json \
   --narrative-modes code critic \
   --parquet-input-dir /mnt/hdfs/userx/shanyong/code/code_wiki/deepwiki \
   --repo-cache-dir /tmp/deepwiki_repo_cache \
   --design-use-vllm \
   --design-vllm-retry-backoff 3 \
   --design-vllm-server-urls "http://[2605:340:cd51:7700:d2a0:3ebe:6ad2:ccb2]:8000/v1/chat/completions,http://[2605:340:cd51:7700:78c7:e555:ac89:5633]:8000/v1/chat/completions,http://[2605:340:cd51:7700:a3ee:7aef:e6a9:10de]:8000/v1/chat/completions,http://[2605:340:cd51:7700:1134:7cdb:9a9e:b6e0]:8000/v1/chat/completions,http://[2605:340:cd51:7700:192e:909d:3448:7e0a]:8000/v1/chat/completions,http://[2605:340:cd51:7700:a442:17d0:4bf:5108]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6873:76a7:1507:fba]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6873:76a7:1507:fba]:8000/v1/chat/completions,http://[2605:340:cd51:7700:ea6f:755d:6615:4dc2]:8000/v1/chat/completions,http://[2605:340:cd51:7700:b205:844b:c560:cc3e]:8000/v1/chat/completions,http://[2605:340:cd51:7700:ac40:6b9:4bac:e956]:8000/v1/chat/completions,http://[2605:340:cd51:7700:a442:17d0:4bf:5108]:8000/v1/chat/completions,http://[2605:340:cd51:7700:e96f:4b63:978a:11e4]:8000/v1/chat/completions,http://[2605:340:cd51:7700:e9ff:7550:86b0:bd8c]:8000/v1/chat/completions,http://[2605:340:cd51:7700:f6e2:dcdb:d569:6fa9]:8000/v1/chat/completions,http://[2605:340:cd51:7700:3758:5cb8:16ec:aea7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:246f:5e8a:a4b:b9a5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:8c0e:31a0:2f3c:93b2]:8000/v1/chat/completions,http://[2605:340:cd51:7700:816e:914b:c9ce:3fc7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:48cc:d0fa:7ee:97d0]:8000/v1/chat/completions,http://[2605:340:cd51:7700:3758:5cb8:16ec:aea7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:227f:3813:5457:7b77]:8000/v1/chat/completions,http://[2605:340:cd51:7700:b7c8:169c:1130:8410]:8000/v1/chat/completions,http://[2605:340:cd51:7700:246f:5e8a:a4b:b9a5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:97:75c4:63eb:7507]:8000/v1/chat/completions,http://[2605:340:cd51:7700:2787:ad8e:6f21:2ae5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:1e37:a428:9cee:677a]:8000/v1/chat/completions,http://[2605:340:cd51:7700:83a:3e2e:9ccd:84fc]:8000/v1/chat/completions,http://[2605:340:cd51:7700:408b:5149:1030:1b93]:8000/v1/chat/completions,http://[2605:340:cd51:7700:808c:c5ef:17f6:ec10]:8000/v1/chat/completions,http://[2605:340:cd51:7700:44fd:a43c:4aaa:be7f]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6542:ae4e:9544:90f7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:48cc:d0fa:7ee:97d0]:8000/v1/chat/completions,http://[2605:340:cd51:7700:808c:c5ef:17f6:ec10]:8000/v1/chat/completions,http://[2605:340:cd51:7700:8dac:3019:cbae:7d61]:8000/v1/chat/completions,http://[2605:340:cd51:7700:83a:3e2e:9ccd:84fc]:8000/v1/chat/completions,http://[2605:340:cd51:7700:e9ff:7550:86b0:bd8c]:8000/v1/chat/completions,http://[2605:340:cd51:7700:daae:23b5:435f:5e38]:8000/v1/chat/completions,http://[2605:340:cd51:7700:816e:914b:c9ce:3fc7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:44fd:a43c:4aaa:be7f]:8000/v1/chat/completions,http://[2605:340:cd51:7700:408b:5149:1030:1b93]:8000/v1/chat/completions,http://[2605:340:cd51:7700:b7c8:169c:1130:8410]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6b11:3d28:f8b2:d047]:8000/v1/chat/completions,http://[2605:340:cd51:7700:1177:b349:28c3:3b84]:8000/v1/chat/completions" \
   --design-vllm-model gpt-oss-120b \
   --design-vllm-temperature 0.7 \
   --judge-use-llm \
   --judge-vllm-server-urls "http://[2605:340:cd51:7700:d2a0:3ebe:6ad2:ccb2]:8000/v1/chat/completions,http://[2605:340:cd51:7700:78c7:e555:ac89:5633]:8000/v1/chat/completions,http://[2605:340:cd51:7700:a3ee:7aef:e6a9:10de]:8000/v1/chat/completions,http://[2605:340:cd51:7700:1134:7cdb:9a9e:b6e0]:8000/v1/chat/completions,http://[2605:340:cd51:7700:192e:909d:3448:7e0a]:8000/v1/chat/completions,http://[2605:340:cd51:7700:a442:17d0:4bf:5108]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6873:76a7:1507:fba]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6873:76a7:1507:fba]:8000/v1/chat/completions,http://[2605:340:cd51:7700:ea6f:755d:6615:4dc2]:8000/v1/chat/completions,http://[2605:340:cd51:7700:b205:844b:c560:cc3e]:8000/v1/chat/completions,http://[2605:340:cd51:7700:ac40:6b9:4bac:e956]:8000/v1/chat/completions,http://[2605:340:cd51:7700:a442:17d0:4bf:5108]:8000/v1/chat/completions,http://[2605:340:cd51:7700:e96f:4b63:978a:11e4]:8000/v1/chat/completions,http://[2605:340:cd51:7700:e9ff:7550:86b0:bd8c]:8000/v1/chat/completions,http://[2605:340:cd51:7700:f6e2:dcdb:d569:6fa9]:8000/v1/chat/completions,http://[2605:340:cd51:7700:3758:5cb8:16ec:aea7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:246f:5e8a:a4b:b9a5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:8c0e:31a0:2f3c:93b2]:8000/v1/chat/completions,http://[2605:340:cd51:7700:816e:914b:c9ce:3fc7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:48cc:d0fa:7ee:97d0]:8000/v1/chat/completions,http://[2605:340:cd51:7700:3758:5cb8:16ec:aea7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:227f:3813:5457:7b77]:8000/v1/chat/completions,http://[2605:340:cd51:7700:b7c8:169c:1130:8410]:8000/v1/chat/completions,http://[2605:340:cd51:7700:246f:5e8a:a4b:b9a5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:97:75c4:63eb:7507]:8000/v1/chat/completions,http://[2605:340:cd51:7700:2787:ad8e:6f21:2ae5]:8000/v1/chat/completions,http://[2605:340:cd51:7700:1e37:a428:9cee:677a]:8000/v1/chat/completions,http://[2605:340:cd51:7700:83a:3e2e:9ccd:84fc]:8000/v1/chat/completions,http://[2605:340:cd51:7700:408b:5149:1030:1b93]:8000/v1/chat/completions,http://[2605:340:cd51:7700:808c:c5ef:17f6:ec10]:8000/v1/chat/completions,http://[2605:340:cd51:7700:44fd:a43c:4aaa:be7f]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6542:ae4e:9544:90f7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:48cc:d0fa:7ee:97d0]:8000/v1/chat/completions,http://[2605:340:cd51:7700:808c:c5ef:17f6:ec10]:8000/v1/chat/completions,http://[2605:340:cd51:7700:8dac:3019:cbae:7d61]:8000/v1/chat/completions,http://[2605:340:cd51:7700:83a:3e2e:9ccd:84fc]:8000/v1/chat/completions,http://[2605:340:cd51:7700:e9ff:7550:86b0:bd8c]:8000/v1/chat/completions,http://[2605:340:cd51:7700:daae:23b5:435f:5e38]:8000/v1/chat/completions,http://[2605:340:cd51:7700:816e:914b:c9ce:3fc7]:8000/v1/chat/completions,http://[2605:340:cd51:7700:44fd:a43c:4aaa:be7f]:8000/v1/chat/completions,http://[2605:340:cd51:7700:408b:5149:1030:1b93]:8000/v1/chat/completions,http://[2605:340:cd51:7700:b7c8:169c:1130:8410]:8000/v1/chat/completions,http://[2605:340:cd51:7700:6b11:3d28:f8b2:d047]:8000/v1/chat/completions,http://[2605:340:cd51:7700:1177:b349:28c3:3b84]:8000/v1/chat/completions" \
   --judge-vllm-model gpt-oss-120b \
   --judge-vllm-max-tokens 512 \
   --judge-vllm-temperature 0.2 \
   --judge-max-rounds 1 \
   --parquet-scan-batch-size 256 \
   --repo-workers 256 \
   --max-workers 64 \
   --section-workers 32 \
   --repo-cache-cleanup on-success \
   --hdfs-output-dir "hdfs://harunawl/home/byte_data_seed_wl/user/xingtianshun/deepwiki_data" \
   --skip-existing \
   --log-level DEBUG
```
说明：当同时设置 `--skip-existing` 和 `--hdfs-output-dir` 时，会优先检查 HDFS 上是否已存在该 repo 对应的输出子目录；若目录已存在则直接跳过该 repo 的生成与上传（实现上会先对 `--hdfs-output-dir` 做一次非递归索引，避免每个 repo 都频繁调用 HDFS 命令）。

说明：当使用 server pooling（`--*-vllm-server-urls`）时，会对 endpoints 做去重；请求侧使用 round-robin（每次请求轮换起始 endpoint）来让所有节点参与分摊，并对连接/超时等瞬态失败的节点做短暂冷却剔除，避免持续打到坏节点；`--*-vllm-retries/--*-vllm-retry-backoff/--*-vllm-timeout` 也会应用到 pooling 调用路径。
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

# 统计token量
python token_count_local.py \
  --text_path result_data/verl_hydrated_clean.txt \
  --tokenizer-path /mnt/hdfs/tiktok_aiic_new/user/codeai/hf_models/Qwen2.5-32B-Instruct \
  --add-special-tokens

python utils/vllm_load_test.py \
   --urls "http://[2605:340:cd51:7700:cd6d:b7d4:77a1:d61e]:8000/v1/chat/completions" \
   --model gpt-oss-120b \
   --prompt "Summarize: hello world." \
   --max-tokens 131072 \
   --temperature 0.2 \
   --warmup 5 \
   --requests 50 \
   --concurrency 16
