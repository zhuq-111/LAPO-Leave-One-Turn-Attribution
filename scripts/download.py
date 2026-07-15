import argparse
from pathlib import Path

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError as exc:
    raise ImportError(
        "ModelScope is required to download these datasets. "
        "Install it with: pip install modelscope"
    ) from exc


DEFAULT_INDEX_REPO_ID = "yamseyoung/wiki-18-e5-index"
DEFAULT_CORPUS_REPO_ID = "zhuoran997/wiki-18-corpus"
INDEX_FILES = ["part_aa", "part_ab"]
CORPUS_FILE = "wiki-18.jsonl.gz"


def download_dataset_files(repo_id: str, files: list[str], save_path: str, max_workers: int) -> None:
    for filename in files:
        print(f"Downloading {repo_id}/{filename} ...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_file_pattern=filename,
            local_dir=save_path,
            max_workers=max_workers,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download wiki-18 index and corpus files from ModelScope."
    )
    parser.add_argument("--save_path", type=str, default="/mnt/si001036p20k/default/shaobo/Search-R1/data/wiki_data", help="Local directory to save files")
    parser.add_argument(
        "--index_repo_id",
        type=str,
        default=DEFAULT_INDEX_REPO_ID,
        help="ModelScope dataset repo ID for the e5 index",
    )
    parser.add_argument(
        "--corpus_repo_id",
        type=str,
        default=DEFAULT_CORPUS_REPO_ID,
        help="ModelScope dataset repo ID for the wiki-18 corpus",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of parallel download workers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    download_dataset_files(args.index_repo_id, INDEX_FILES, str(save_path), args.max_workers)
    download_dataset_files(args.corpus_repo_id, [CORPUS_FILE], str(save_path), args.max_workers)
    print(f"Downloaded files to {save_path.resolve()}")


if __name__ == "__main__":
    main()
