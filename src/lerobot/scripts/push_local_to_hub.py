from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/path/to/local/dataset",
    repo_id="Vertax/place_dual_shoes_demo_clean_arx5_repo",
    repo_type="dataset",
)
