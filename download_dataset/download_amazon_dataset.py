# download_all_categories.py
from pathlib import Path

from huggingface_hub import list_repo_files, hf_hub_download

REPO = "McAuley-Lab/Amazon-Reviews-2023"   # dataset repo on the Hub
OUT  = Path("amazon_reviews_2023_categories")  # download target
OUT.mkdir(exist_ok=True)

for path in list_repo_files(REPO, repo_type="dataset"):           # ① list every file
    if path.startswith("raw/review_categories/") and path.endswith(".jsonl"):
        hf_hub_download(                                         # ② grab it
            repo_id=REPO,
            repo_type="dataset",
            filename=path,
            local_dir=OUT,                    # files land in OUT/<same-basename>
            local_dir_use_symlinks=False,     # copy the real file, no symlink
            resume_download=True,             # pick up where you left off
        )
        print("✓", Path(path).name)