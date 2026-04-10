#!/usr/bin/env python
"""
Push a local LeRobot dataset to the Hugging Face Hub.

This script is useful when:
1. The push_to_hub step failed during recording (e.g., network issues, SSL errors)
2. You want to push a previously recorded dataset to the Hub
3. You want to re-upload a dataset with different settings

Usage:
    # Basic usage (requires both --repo-id and --dataset-path)
    python -m lerobot.scripts.push_dataset_to_hub \\
        --repo-id Vertax/xense_flare_pick_and_place \\
        --dataset-path ~/.cache/huggingface/lerobot/Vertax/xense_flare_pick_and_place

    # Use upload_large_folder for large datasets (recommended)
    python -m lerobot.scripts.push_dataset_to_hub \
        --repo-id Xense/assemble_box_with_phone_stand0410 \
        --dataset-path ~/.cache/huggingface/lerobot/Xense/assemble_box_with_phone_stand0410 \
        --upload-large-folder

    # Push as private dataset
    python -m lerobot.scripts.push_dataset_to_hub \\
        --repo-id Vertax/xense_flare_pick_and_place \\
        --dataset-path ~/.cache/huggingface/lerobot/Vertax/xense_flare_pick_and_place \\
        --private

    # Skip pushing videos (only push metadata and parquet files)
    python -m lerobot.scripts.push_dataset_to_hub \\
        --repo-id Vertax/xense_flare_pick_and_place \\
        --dataset-path ~/.cache/huggingface/lerobot/Vertax/xense_flare_pick_and_place \\
        --no-videos

Examples:
    # First login to Hugging Face
    huggingface-cli login

    # Push xense_flare dataset with large folder API
    python -m lerobot.scripts.push_dataset_to_hub \\
        --repo-id Vertax/xense_flare_pick_and_place_cubes_20260104 \\
        --dataset-path ~/.cache/huggingface/lerobot/Vertax/xense_flare_pick_and_place_cubes_20260104 \\
        --upload-large-folder
"""

import argparse
import logging
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def push_dataset_to_hub(
    dataset_path: Path,
    repo_id: str,
    branch: str | None = None,
    tags: list | None = None,
    license: str | None = "apache-2.0",
    tag_version: bool = True,
    push_videos: bool = True,
    private: bool = False,
    allow_patterns: list[str] | str | None = None,
    upload_large_folder: bool = False,
    **card_kwargs,
) -> None:
    """
    Push a local dataset to the Hugging Face Hub.

    Args:
        dataset_path: Path to the local dataset directory
        repo_id: Hub repository ID (e.g., "Vertax/xense_flare_pick_and_place")
        branch: Git branch to push to (default: main)
        tags: Tags to add to the dataset card
        license: License for the dataset
        tag_version: Whether to tag with codebase version
        push_videos: Whether to push video files
        private: Whether to make the repository private
        allow_patterns: Patterns of files to include
        upload_large_folder: Use upload_large_folder API for large datasets
        **card_kwargs: Additional arguments for the dataset card
    """
    dataset_path = Path(dataset_path).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    logging.info(f"Pushing dataset to: {repo_id}")
    logging.info(f"Dataset path: {dataset_path}")
    logging.info(f"Private: {private}")
    logging.info(f"Push videos: {push_videos}")
    logging.info(f"Upload large folder: {upload_large_folder}")
    if not push_videos:
        logging.info("Skipping video files")

    logging.info("Loading local LeRobot dataset...")
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)

    logging.info("Starting upload...")
    try:
        dataset.push_to_hub(
            branch=branch,
            tags=tags,
            license=license,
            tag_version=tag_version,
            push_videos=push_videos,
            private=private,
            allow_patterns=allow_patterns,
            upload_large_folder=upload_large_folder,
            **card_kwargs,
        )
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        logging.info("Tips:")
        logging.info("  - Try using --upload-large-folder for large datasets")
        logging.info("  - Check your network connection")
        logging.info("  - Make sure you have write access to the repository")
        logging.info("  - Make sure you are logged in with: huggingface-cli login")
        raise

    logging.info(
        f"✅ Dataset successfully pushed to: https://huggingface.co/datasets/{repo_id}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Push a local LeRobot dataset to the Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the local dataset directory (e.g., ~/.cache/huggingface/lerobot/username/dataset-name)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hub repository ID (e.g., 'Vertax/xense_flare_pick_and_place_cubes_20260104')",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Git branch to push to (default: main)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=None,
        help="Tags to add to the dataset card",
    )
    parser.add_argument(
        "--license",
        type=str,
        default="apache-2.0",
        help="License for the dataset (default: apache-2.0)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip pushing video files",
    )
    parser.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="Use upload_large_folder API (recommended for large datasets with many files)",
    )
    parser.add_argument(
        "--no-tag-version",
        action="store_true",
        help="Do not tag with codebase version",
    )

    args = parser.parse_args()

    setup_logging()

    try:
        push_dataset_to_hub(
            dataset_path=args.dataset_path,
            repo_id=args.repo_id,
            branch=args.branch,
            tags=args.tags,
            license=args.license,
            tag_version=not args.no_tag_version,
            push_videos=not args.no_videos,
            private=args.private,
            upload_large_folder=args.upload_large_folder,
        )
    except KeyboardInterrupt:
        logging.info("\nUpload cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to push dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
