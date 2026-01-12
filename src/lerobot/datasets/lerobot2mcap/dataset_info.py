"""Dataset information parser for LeRobot datasets.
Parses the info.json file to extract metadata about the dataset.

Supports both v2.x and v3.0 LeRobot dataset formats:
- v2.x: One parquet/video file per episode
- v3.0: Multiple episodes merged into single parquet/video files,
        with episode boundaries stored in meta/episodes/ parquet files

TODO: Boilerplate Reduction (Low Priority)
    Consider using Pydantic model or __getattr__ to reduce repetitive getter methods.
    See REVIEW.md for suggested approaches.

TODO: Add validate_files() Method (Low Priority)
    Optional method to verify files exist on disk before conversion:
    def validate_files(self, dataset_root: Path) -> dict:
        Returns {"missing_parquet": [...], "missing_videos": {...}, "extra_files": [...]}
"""

import json
import math
from pathlib import Path

import pandas as pd

from lerobot.datasets.lerobot2mcap.logger import get_logger

logger = get_logger("dataset_info")

# Official LeRobot defaults (from lerobot.datasets.utils)
DEFAULT_CHUNK_SIZE = 1000  # Max number of episodes per chunk
DEFAULT_CODEC = "h264"  # Supported: "h264", "av1"
DEFAULT_FILE_INDEX = 0  # First file in a chunk


class DatasetInfo:
    """Parses and holds LeRobot dataset metadata from info.json."""

    def __init__(self, info_json_path: Path):
        """
        Initialize DatasetInfo by parsing info.json.

        Args:
            info_json_path: Path to the info.json file (usually in meta/info.json)
        """
        self.info_json_path = info_json_path  # Assign the info json address
        self.dataset_root = info_json_path.parent.parent  # meta/info.json -> dataset_root
        self.data = (
            self._parse_info_json()
        )  # Load information from the info.json into the data member variable
        self.video_keys = (
            self._extract_video_keys()
        )  # Extract keys of video dict - the names of your camera feeds

        # Load episodes metadata for v3.0 datasets
        self._episodes_df: pd.DataFrame | None = None
        if self.is_v3_format():
            self._load_episodes_metadata()

        logger.info(
            f"Loaded dataset info from {info_json_path}"
        )  # Record info.json data extraction in file logs
        logger.info(
            f"Found {len(self.video_keys)} video streams: {self.video_keys}"
        )  # Record number of keys and the key name strings

    def _parse_info_json(self) -> dict:
        """Parse the info.json file."""
        if not self.info_json_path.exists():
            raise FileNotFoundError(f"info.json not found at {self.info_json_path}")

        with open(self.info_json_path) as f:
            return json.load(f)

    def _extract_video_keys(self) -> list[str]:
        """
        Extract video keys from features.
        Video features have dtype="video" in info.json.
        Example: "observation.images.front", "observation.images.external"

        Raises:
            KeyError: If 'features' field is missing from info.json
        """
        if "features" not in self.data:
            raise KeyError(
                f"Required field 'features' not found in {self.info_json_path}. "
                "This field is required in LeRobot datasets."
            )

        video_keys = []
        features = self.data["features"]

        for feature_name, feature_info in features.items():
            if isinstance(feature_info, dict) and feature_info.get("dtype") == "video":
                video_keys.append(feature_name)

        return video_keys

    def is_v3_format(self) -> bool:
        """
        Check if the dataset is in v3.0 format.

        v3.0 format merges multiple episodes into single parquet/video files.

        Returns:
            True if dataset is v3.0 format, False otherwise
        """
        version = self.get_codebase_version()
        return version.startswith("v3")

    def _load_episodes_metadata(self) -> None:
        """
        Load episodes metadata from meta/episodes/ parquet files.

        For v3.0 datasets, this contains episode boundaries (from_index, to_index)
        and video timestamp ranges for each episode.
        """
        episodes_dir = self.dataset_root / "meta" / "episodes"
        if not episodes_dir.exists():
            logger.warn(
                f"Episodes metadata directory not found: {episodes_dir}. This is required for v3.0 datasets."
            )
            return

        # Find all episode metadata parquet files
        episode_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
        if not episode_files:
            logger.warn(f"No episode metadata files found in {episodes_dir}")
            return

        # Load and concatenate all episode metadata
        dfs = []
        for ep_file in episode_files:
            try:
                df = pd.read_parquet(ep_file)
                dfs.append(df)
            except Exception as e:
                logger.warn(f"Failed to load episode metadata from {ep_file}: {e}")

        if dfs:
            self._episodes_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded metadata for {len(self._episodes_df)} episodes")
        else:
            logger.warn("No episode metadata could be loaded")

    def get_episode_info(self, episode_index: int) -> dict:
        """
        Get detailed info for a specific episode (v3.0 format).

        Args:
            episode_index: The episode index

        Returns:
            Dictionary with episode info:
            {
                "episode_index": int,
                "length": int,
                "data_chunk_index": int,
                "data_file_index": int,
                "from_index": int,  # Start index in the data file
                "to_index": int,    # End index in the data file
                "videos": {
                    "video_key": {
                        "chunk_index": int,
                        "file_index": int,
                        "from_timestamp": float,
                        "to_timestamp": float,
                    },
                    ...
                }
            }

        Raises:
            ValueError: If episode_index is not found or v3.0 metadata not loaded
        """
        if self._episodes_df is None:
            raise ValueError("Episodes metadata not loaded. This method is only available for v3.0 datasets.")

        episode_row = self._episodes_df[self._episodes_df["episode_index"] == episode_index]
        if episode_row.empty:
            raise ValueError(f"Episode {episode_index} not found in metadata")

        row = episode_row.iloc[0]

        info = {
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "data_chunk_index": int(row["data/chunk_index"]),
            "data_file_index": int(row["data/file_index"]),
            "from_index": int(row["dataset_from_index"]),
            "to_index": int(row["dataset_to_index"]),
            "videos": {},
        }

        # Extract video info for each video key
        for video_key in self.video_keys:
            video_prefix = f"videos/{video_key}/"
            try:
                info["videos"][video_key] = {
                    "chunk_index": int(row[f"{video_prefix}chunk_index"]),
                    "file_index": int(row[f"{video_prefix}file_index"]),
                    "from_timestamp": float(row[f"{video_prefix}from_timestamp"]),
                    "to_timestamp": float(row[f"{video_prefix}to_timestamp"]),
                }
            except KeyError:
                logger.warn(f"Video metadata for {video_key} not found in episode {episode_index}")

        return info

    @property
    def data_path_template(self) -> str:
        """
        Get the data (parquet) path template from dataset metadata.

        Returns:
            Path template string (e.g., "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")

        Raises:
            KeyError: If 'data_path' field is missing from info.json
        """
        if "data_path" not in self.data:
            raise KeyError(
                f"Required field 'data_path' not found in {self.info_json_path}. "
                "This field is required in LeRobot datasets."
            )
        return self.data["data_path"]

    @property
    def video_path_template(self) -> str:
        """
        Get the video path template from dataset metadata.

        Returns:
            Path template string (e.g., "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

        Raises:
            KeyError: If 'video_path' field is missing from info.json when video features exist
        """
        if "video_path" not in self.data:
            if self.video_keys:  # Only required if dataset has video features
                raise KeyError(
                    f"Required field 'video_path' not found in {self.info_json_path}. "
                    "This field is required when the dataset contains video features."
                )
            # Return empty string if no videos (though this shouldn't be called in that case)
            return ""
        return self.data["video_path"]

    def get_chunk_files(self, chunk_index: int, dataset_root: Path) -> dict:
        """
        Get file paths for a specific chunk.

        Args:
            chunk_index: The chunk index (e.g., 0 for chunk-000)
            dataset_root: Root directory of the dataset

        Returns:
            Dictionary with:
            {
                "parquet": Path to parquet file,
                "videos": {
                    "observation.images.front": Path to video file,
                    "observation.images.external": Path to video file,
                    ...
                }
            }
        """
        chunk_files = {}

        # Determine format parameters based on dataset version
        version = self.get_codebase_version()
        if version.startswith("v2"):
            # v2.1 format: use episode_chunk and episode_index, null out v3 params
            format_params = {
                "episode_chunk": chunk_index,
                "episode_index": DEFAULT_FILE_INDEX,
                "chunk_index": None,
                "file_index": None,
            }
        else:
            # v3.0+ format: use chunk_index and file_index, null out v2 params
            format_params = {
                "episode_chunk": None,
                "episode_index": None,
                "chunk_index": chunk_index,
                "file_index": DEFAULT_FILE_INDEX,
            }

        # Get parquet file path
        parquet_path = self.data_path_template.format(**format_params)
        chunk_files["parquet"] = dataset_root / parquet_path

        # Get video file paths
        chunk_files["videos"] = {}
        for video_key in self.video_keys:
            video_path = self.video_path_template.format(
                video_key=video_key,
                **format_params,
            )
            chunk_files["videos"][video_key] = dataset_root / video_path

        return chunk_files

    def get_episode_files(self, episode_index: int, chunk_index: int, dataset_root: Path) -> dict:
        """
        Get file paths for a specific episode within a chunk.

        For v3.0 datasets, this returns the shared file paths plus index/timestamp ranges
        to extract the specific episode from the merged files.

        Args:
            episode_index: The episode index (e.g., 0 for file-000, 1 for file-001)
            chunk_index: The chunk index (e.g., 0 for chunk-000)
            dataset_root: Root directory of the dataset

        Returns:
            Dictionary with:
            {
                "parquet": Path to parquet file,
                "videos": {
                    "observation.images.front": Path to video file,
                    ...
                },
                # v3.0 only fields:
                "from_index": int,  # Start index in parquet
                "to_index": int,    # End index in parquet
                "video_ranges": {   # Timestamp ranges for video extraction
                    "observation.images.front": {
                        "from_timestamp": float,
                        "to_timestamp": float,
                    },
                    ...
                }
            }
        """
        episode_files = {}

        # Determine format parameters based on dataset version
        version = self.get_codebase_version()
        if version.startswith("v2"):
            # v2.1 format: use episode_chunk and episode_index, null out v3 params
            format_params = {
                "episode_chunk": chunk_index,
                "episode_index": episode_index,
                "chunk_index": None,
                "file_index": None,
            }

            # Get parquet file path
            parquet_path = self.data_path_template.format(**format_params)
            episode_files["parquet"] = dataset_root / parquet_path

            # Get video file paths
            episode_files["videos"] = {}
            for video_key in self.video_keys:
                video_path = self.video_path_template.format(
                    video_key=video_key,
                    **format_params,
                )
                episode_files["videos"][video_key] = dataset_root / video_path
        else:
            # v3.0+ format: use episode metadata for correct file paths and ranges
            if self._episodes_df is None:
                raise ValueError("Episodes metadata not loaded. Cannot get episode files for v3.0 dataset.")

            # Get episode info from metadata
            ep_info = self.get_episode_info(episode_index)

            # Build format params from episode metadata
            format_params = {
                "episode_chunk": None,
                "episode_index": None,
                "chunk_index": ep_info["data_chunk_index"],
                "file_index": ep_info["data_file_index"],
            }

            # Get parquet file path
            parquet_path = self.data_path_template.format(**format_params)
            episode_files["parquet"] = dataset_root / parquet_path

            # Add index range for filtering parquet data
            episode_files["from_index"] = ep_info["from_index"]
            episode_files["to_index"] = ep_info["to_index"]

            # Get video file paths and timestamp ranges
            episode_files["videos"] = {}
            episode_files["video_ranges"] = {}
            for video_key in self.video_keys:
                if video_key in ep_info["videos"]:
                    video_info = ep_info["videos"][video_key]
                    video_format_params = {
                        "episode_chunk": None,
                        "episode_index": None,
                        "chunk_index": video_info["chunk_index"],
                        "file_index": video_info["file_index"],
                    }
                    video_path = self.video_path_template.format(
                        video_key=video_key,
                        **video_format_params,
                    )
                    episode_files["videos"][video_key] = dataset_root / video_path
                    episode_files["video_ranges"][video_key] = {
                        "from_timestamp": video_info["from_timestamp"],
                        "to_timestamp": video_info["to_timestamp"],
                    }

        return episode_files

    def get_total_chunks(self) -> int:
        """
        Get total number of chunks in the dataset.

        Calculates from metadata using: ceil(total_episodes / chunks_size)
        where chunks_size is the maximum number of episodes per chunk.

        Returns:
            Number of chunks in the dataset (minimum 1)

        Raises:
            KeyError: If 'total_episodes' field is missing from info.json
        """
        if "total_episodes" not in self.data:
            raise KeyError(
                f"Required field 'total_episodes' not found in {self.info_json_path}. "
                "This field is required in LeRobot datasets."
            )

        total_episodes = self.data["total_episodes"]
        # chunks_size has an official default value
        chunks_size = self.data.get("chunks_size", DEFAULT_CHUNK_SIZE)

        if total_episodes > 0 and chunks_size > 0:
            return math.ceil(total_episodes / chunks_size)

        # Fallback to 1 chunk if calculation not possible
        return 1

    def get_fps(self) -> int:
        """
        Get frames per second from dataset info.

        Returns:
            Frames per second as an integer

        Raises:
            KeyError: If 'fps' field is missing from info.json
        """
        if "fps" not in self.data:
            raise KeyError(
                f"Required field 'fps' not found in {self.info_json_path}. "
                "This field is required in LeRobot datasets."
            )
        return self.data["fps"]

    def get_codebase_version(self) -> str:
        """
        Get the LeRobot codebase version (dataset format version).

        This indicates the schema version of the dataset and parquet files.
        Examples: "v2.0", "v2.1", "v3.0"

        Returns:
            Codebase version string

        Raises:
            KeyError: If 'codebase_version' field is missing from info.json
        """
        if "codebase_version" not in self.data:
            raise KeyError(
                f"Required field 'codebase_version' not found in {self.info_json_path}. "
                "This field is required in LeRobot datasets to identify the dataset format version."
            )
        return self.data["codebase_version"]

    def get_writer_format(self) -> str:
        """
        Get MCAP writer format from dataset metadata.

        Returns:
            Writer format (e.g., "ros1", "ros2", "json", "protobuf")
            Defaults to "ros2" for LeRobot datasets.
        """
        return self.data.get("writer_format", "ros2")

    def get_video_frame_id(self, video_key: str) -> str:
        """
        Generate frame_id for a video stream.
        Args:
            video_key: The video key (e.g., "observation.images.front")
        Returns:
            Frame ID (e.g., "front_camera")
        """
        # Extract the last part of the video key as frame_id
        # "observation.images.front" -> "front_camera"
        camera_name = video_key.split(".")[-1] if "." in video_key else "camera"
        return f"{camera_name}_camera"

    def get_total_episodes(self) -> int:
        """
        Get total number of episodes in the dataset.

        Returns:
            Total number of episodes as an integer

        Raises:
            KeyError: If 'total_episodes' field is missing from info.json
        """
        if "total_episodes" not in self.data:
            raise KeyError(
                f"Required field 'total_episodes' not found in {self.info_json_path}. "
                "This field is required in LeRobot datasets."
            )
        return self.data["total_episodes"]

    def __repr__(self) -> str:
        return (
            f"DatasetInfo(version={self.get_codebase_version()}, "
            f"episodes={self.get_total_episodes()}, "
            f"chunks={self.get_total_chunks()}, "
            f"fps={self.get_fps()}, "
            f"videos={len(self.video_keys)})"
        )
