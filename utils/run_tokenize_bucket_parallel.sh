#!/usr/bin/env bash
set -euo pipefail

# Parallel runner for utils/tokenize_and_bucket_parquet.py using sharding.
#
# Usage:
#   utils/run_tokenize_bucket_parallel.sh "<input_glob>" "<output_dir>" [num_shards]
#
# Example:
#   utils/run_tokenize_bucket_parallel.sh \
#     "/mnt/hdfs/.../0213_all_narratives.worker*.part*.parquet" \
#     "/mnt/hdfs/.../0213_buckets" \
#     64
#
# Notes:
# - Safe to rerun: uses --resume and per-file done markers.
# - For multiple processes sharing the same output dir, we enable --claim to avoid duplicate work.

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"<input_glob>\" \"<output_dir>\" [num_shards]" >&2
  exit 2
fi

INPUT_GLOB="$1"
OUTPUT_DIR="$2"
NUM_SHARDS="${3:-64}"

# You can override these via env vars.
HEARTBEAT_SECS="${HEARTBEAT_SECS:-60}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WRITE_BATCH="${WRITE_BATCH:-8}"
TOKENIZER_BACKEND="${TOKENIZER_BACKEND:-tiktoken}"
ENCODING="${ENCODING:-cl100k_base}"

LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/_resume/logs}"
mkdir -p "$LOG_DIR"

PIDS=()

cleanup() {
  # Best-effort: stop children on Ctrl-C / termination.
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

echo "[runner] input_glob=$INPUT_GLOB" >&2
echo "[runner] output_dir=$OUTPUT_DIR" >&2
echo "[runner] num_shards=$NUM_SHARDS" >&2
echo "[runner] logs=$LOG_DIR" >&2

for ((i=0; i<NUM_SHARDS; i++)); do
  log="$LOG_DIR/shard_${i}.log"
  (
    python utils/tokenize_and_bucket_parquet.py \
      --input-glob "$INPUT_GLOB" \
      --output-dir "$OUTPUT_DIR" \
      --tokenizer-backend "$TOKENIZER_BACKEND" --encoding "$ENCODING" \
      --batch-size "$BATCH_SIZE" --write-batch "$WRITE_BATCH" \
      --resume --claim \
      --num-shards "$NUM_SHARDS" --shard-index "$i" \
      --heartbeat-secs "$HEARTBEAT_SECS"
  ) >"$log" 2>&1 &
  PIDS+=("$!")
done

fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

echo "[runner] done. exit_code=$fail" >&2
echo "[runner] summarize:" >&2
python utils/summarize_bucket_run.py --output-dir "$OUTPUT_DIR" --write-json "$OUTPUT_DIR/summary.json" || true

exit "$fail"

