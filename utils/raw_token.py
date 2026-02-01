import os, glob
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import tiktoken

BASE = "/mnt/hdfs/userx/shanyong/code/code_wiki/deepwiki"
ENCODING = "cl100k_base"   # 需要的话改成 "o200k_base"
BATCH_SIZE = 4096           # 太大可能占内存；太小会慢

enc = tiktoken.get_encoding(ENCODING)

# 1) 统计总行数（用于进度条 total）
part_files = sorted(glob.glob(os.path.join(BASE, "part-*")))
if not part_files:
    raise SystemExit(f"No part-* files under {BASE}")

total_rows = 0
for fp in tqdm(part_files, desc="Reading parquet footers", unit="file", dynamic_ncols=True):
    total_rows += pq.ParquetFile(fp).metadata.num_rows

# 2) 流式读取 content 并统计 token
dataset = ds.dataset(BASE, format="parquet")
scanner = dataset.scanner(columns=["content"], batch_size=BATCH_SIZE)

total_tokens = 0
null_rows = 0
rows_seen = 0

pbar = tqdm(total=total_rows, desc=f"Tokenizing (tiktoken:{ENCODING})", unit="rows", dynamic_ncols=True)
for batch in scanner.to_batches():
    for s in batch.column(0).to_pylist():
        if s is None:
            null_rows += 1
            continue
        total_tokens += len(enc.encode_ordinary(s))  # 更快：不处理特殊token
    rows_seen += batch.num_rows
    pbar.update(batch.num_rows)
    pbar.set_postfix(tokens=total_tokens, nulls=null_rows)

pbar.close()
print("\nDONE")
print("base:", BASE)
print("files:", len(part_files))
print("total_rows:", total_rows)
print("rows_seen:", rows_seen)
print("null_content_rows:", null_rows)
print("total_tokens:", total_tokens)