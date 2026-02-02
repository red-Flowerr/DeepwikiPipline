import argparse
import os
import sys
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def make_token_counter(
    tokenizer_backend: str,
    tiktoken_encoding: str,
    hf_model: str,
    hf_trust_remote_code: bool,
    hf_local_files_only: bool,
):
    if tokenizer_backend == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(tiktoken_encoding)

        def count_tokens(s: str) -> int:
            return len(enc.encode_ordinary(s))

        return count_tokens

    if tokenizer_backend == "hf":
        from transformers import AutoTokenizer

        if not hf_model:
            raise ValueError("--hf-model is required when --tokenizer-backend=hf")

        try:
            tok = AutoTokenizer.from_pretrained(
                hf_model,
                trust_remote_code=hf_trust_remote_code,
                local_files_only=hf_local_files_only,
                use_fast=True,
            )
        except Exception:
            tok = AutoTokenizer.from_pretrained(
                hf_model,
                trust_remote_code=hf_trust_remote_code,
                local_files_only=hf_local_files_only,
                use_fast=False,
            )

        # We only count tokens; we are not running the model. Avoid max_length warnings if possible.
        try:
            tok.model_max_length = 1 << 60
        except Exception:
            pass

        def count_tokens(s: str) -> int:
            return len(tok.encode(s, add_special_tokens=False))

        return count_tokens

    raise ValueError(f"unknown tokenizer backend: {tokenizer_backend}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read a parquet with a 'narrative' column and write a new parquet with an added 'narrative_tokens' column."
    )
    ap.add_argument("--input", required=True, help="Input parquet path")
    ap.add_argument("--output", required=True, help="Output parquet path (new file)")
    ap.add_argument("--narrative-col", default="narrative", help="Narrative column name (default: %(default)s)")
    ap.add_argument("--tokens-col", default="narrative_tokens", help="Tokens column name to add (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=64, help="Rows per processing batch (default: %(default)s)")

    ap.add_argument("--tokenizer-backend", default="tiktoken", choices=["tiktoken", "hf"])
    ap.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding name (default: %(default)s)")
    ap.add_argument("--hf-model", default="", help="HF model name/path (required when --tokenizer-backend=hf)")
    ap.add_argument("--hf-trust-remote-code", action="store_true")
    ap.add_argument("--hf-local-files-only", action="store_true", default=True)
    ap.add_argument("--hf-allow-download", action="store_false", dest="hf_local_files_only")
    args = ap.parse_args()

    if args.tokenizer_backend == "hf" and not args.hf_model:
        raise SystemExit("--hf-model is required when --tokenizer-backend=hf")

    count_tokens = make_token_counter(
        tokenizer_backend=args.tokenizer_backend,
        tiktoken_encoding=args.encoding,
        hf_model=args.hf_model,
        hf_trust_remote_code=args.hf_trust_remote_code,
        hf_local_files_only=args.hf_local_files_only,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Avoid noisy warnings when texts exceed model max length (we only count tokens).
    warnings.filterwarnings("ignore", message="Token indices sequence length is longer than*")

    pf = pq.ParquetFile(args.input)
    in_schema = pf.schema_arrow
    if args.narrative_col not in set(in_schema.names):
        raise SystemExit(f"Input parquet missing column: {args.narrative_col}")
    if args.tokens_col in set(in_schema.names):
        raise SystemExit(f"Input parquet already has column: {args.tokens_col}")

    out_schema = pa.schema(list(in_schema) + [pa.field(args.tokens_col, pa.int64())])
    writer = pq.ParquetWriter(args.output, out_schema, compression="zstd")

    total_rows = pf.metadata.num_rows
    total_tokens = 0
    rows_done = 0

    pbar = tqdm(
        total=total_rows,
        desc="Adding narrative_tokens",
        unit="rows",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    try:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            rb = pa.RecordBatch.from_struct_array(pa.StructArray.from_arrays(batch.columns, names=batch.schema.names))
            # rb is same as batch but explicit RecordBatch (safe for pyarrow ops)
            narratives = rb.column(rb.schema.get_field_index(args.narrative_col)).to_pylist()
            counts = []
            for s in narratives:
                if not s:
                    counts.append(0)
                    continue
                if not isinstance(s, str):
                    s = str(s)
                c = int(count_tokens(s))
                total_tokens += c
                counts.append(c)

            out_batch = rb.append_column(args.tokens_col, pa.array(counts, type=pa.int64()))
            writer.write_table(pa.Table.from_batches([out_batch], schema=out_schema))

            rows_done += out_batch.num_rows
            pbar.update(out_batch.num_rows)
            pbar.set_postfix(tokens=total_tokens)
    finally:
        pbar.close()
        writer.close()

    print("\nDONE")
    print("input:", args.input)
    print("output:", args.output)
    print("rows:", rows_done)
    print("total_tokens:", total_tokens)


if __name__ == "__main__":
    main()

