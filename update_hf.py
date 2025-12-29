from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/mlx_devbox/users/xingtianshun/playground/SWE-bench_Verified/Deepwiki_pipline/result_data/verl_narratives_merged.txt",     # 本地 txt
    path_in_repo="verl_narratives_merged.txt",              # HF repo 中的路径
    repo_id="xtsssss/deepwiki_case",
    repo_type="dataset",
)
