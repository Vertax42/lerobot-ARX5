"""Main converter orchestrator for LeRobot to MCAP conversion."""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from lerobot.datasets.lerobot2mcap.config_generator import ConfigGenerator
from lerobot.datasets.lerobot2mcap.dataset_info import DatasetInfo
from lerobot.datasets.lerobot2mcap.logger import get_logger
from lerobot.datasets.lerobot2mcap.sorted_mcap_converter import SortedMcapConverter

logger = get_logger("converter")

# Display formatting constants
SEPARATOR_WIDTH = 60

# Project root (for writing back the latest example config)
PROJECT_ROOT = Path(__file__).resolve().parent


class LeRobotConverter:
    """Main converter orchestrator that manages the conversion process."""

    def __init__(self, dataset_root: Path, converter_functions_path: Path):
        """
        Initialize LeRobotConverter.
        Args:
            dataset_root: Root directory of the LeRobot dataset
            converter_functions_path: Path to converter_functions.yaml
        """
        self.dataset_root = dataset_root
        self.converter_functions_path = converter_functions_path

        info_json_path = dataset_root / "meta" / "info.json"
        if not info_json_path.exists():
            raise FileNotFoundError(
                f"Dataset info.json not found at {info_json_path}. Is {dataset_root} a valid LeRobot dataset?"
            )

        # Import Dataset Info from dataset_info.py
        self.dataset_info = DatasetInfo(info_json_path)

        # Import Config Generator from config_generator.py
        self.config_generator = ConfigGenerator(self.dataset_info)

        # Find log file
        self.log_file = self._find_log_file()
        if self.log_file:
            logger.info(f"Found log file: {self.log_file}")
        else:
            logger.info("No log file found in dataset root")

        logger.info(f"Initialized converter for {self.dataset_info}")

    def _find_log_file(self) -> Path | None:
        """
        Find the .log file in the dataset root directory.
        Returns:
            Path to the log file, or None if not found
        """
        log_files = list(self.dataset_root.glob("*.log"))

        if not log_files:
            return None

        if len(log_files) > 1:
            logger.warn(
                f"Multiple log files found: {[f.name for f in log_files]}. "
                f"Using the first one: {log_files[0].name}"
            )

        return log_files[0]

    def convert(
        self,
        output_dir: Path,
        chunks: list[int] | None = None,
        episodes: list[int] | None = None,
        num_workers: int = 1,
    ) -> bool:
        """
        Convert the LeRobot dataset to MCAP format.
        Iterates through each chunk and converts all episodes within that chunk.
        Each episode produces a separate MCAP file in its own directory.

        Args:
            output_dir: Directory where MCAP files will be saved
            chunks: List of chunk indices to convert (None = all chunks)
            episodes: List of episode indices to convert (None = all episodes)
            num_workers: Number of parallel workers (default: 1)
        Returns:
            True if conversion succeeded, False otherwise
        """
        logger.info("=" * SEPARATOR_WIDTH)
        logger.info("LeRobot to MCAP Conversion")
        logger.info("=" * SEPARATOR_WIDTH)
        logger.info(f"Dataset: {self.dataset_root}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Dataset info: {self.dataset_info}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine which episodes to convert
        if episodes is None:
            total_episodes = self.dataset_info.get_total_episodes()
            episodes = list(range(total_episodes))
            logger.info(f"Converting all {total_episodes} episodes")
        else:
            logger.info(f"Converting episodes: {episodes}")

        # For v3.0 format, chunks parameter is ignored since all episodes
        # are in shared files. For v2.x, we still support chunk filtering.
        is_v3 = self.dataset_info.is_v3_format()
        if is_v3:
            logger.info("Detected v3.0 dataset format (merged episodes)")
            if chunks is not None:
                logger.warn(
                    "Chunks parameter is ignored for v3.0 datasets. All episodes are in shared files."
                )
            # For v3.0, use chunk 0 as placeholder (actual chunk comes from metadata)
            all_conversions = [(0, episode_idx) for episode_idx in episodes]
        else:
            # v2.x format: each episode has its own file
            if chunks is None:
                total_chunks = self.dataset_info.get_total_chunks()
                chunks = list(range(total_chunks))
                logger.info(f"Converting all {total_chunks} chunks")
            else:
                logger.info(f"Converting chunks: {chunks}")

            # Create a flat list of (chunk_idx, episode_idx) pairs for all conversions
            all_conversions = [(chunk_idx, episode_idx) for chunk_idx in chunks for episode_idx in episodes]

        # Convert each episode
        success_count = 0
        fail_count = 0
        last_episode_config = None

        def convert_one(args):
            """Convert a single episode. Returns (success, config, error)."""
            chunk_idx, episode_idx = args
            try:
                if is_v3:
                    cfg = self._convert_episode_v3(episode_idx, output_dir)
                else:
                    cfg = self._convert_episode(episode_idx, chunk_idx, output_dir)
                return (True, cfg, None, episode_idx, chunk_idx)
            except Exception as e:
                return (False, None, str(e), episode_idx, chunk_idx)

        if num_workers > 1:
            # Parallel conversion using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(convert_one, args): args for args in all_conversions}
                for future in tqdm(
                    as_completed(futures),
                    total=len(all_conversions),
                    desc=f"Converting episodes ({num_workers} workers)",
                    unit="episode",
                ):
                    success, cfg, error, episode_idx, chunk_idx = future.result()
                    if success:
                        last_episode_config = cfg
                        success_count += 1
                    else:
                        logger.warn(f"Skipping episode {episode_idx} in chunk {chunk_idx}: {error}")
                        fail_count += 1
        else:
            # Sequential conversion
            for chunk_idx, episode_idx in tqdm(
                all_conversions,
                desc="Converting episodes",
                unit="episode",
            ):
                success, cfg, error, ep_idx, ch_idx = convert_one((chunk_idx, episode_idx))
                if success:
                    last_episode_config = cfg
                    success_count += 1
                else:
                    logger.warn(f"Skipping episode {ep_idx} in chunk {ch_idx}: {error}")
                    fail_count += 1

        # Save a single config file named after output directory
        if last_episode_config:
            config_name = f"config_{output_dir.name}.yaml"
            config_path = output_dir / config_name
            with open(config_path, "w") as config_file:
                yaml.dump(
                    last_episode_config.model_dump(mode="python", exclude_none=True),
                    config_file,
                    default_flow_style=False,
                    sort_keys=False,
                )
            logger.info(f"Saved config: {config_path}")

            # Also update example config in configs/
            self._update_example_config(config_path)

        # Summary
        logger.info("=" * SEPARATOR_WIDTH)
        logger.info("Conversion Summary")
        logger.info("=" * SEPARATOR_WIDTH)
        logger.info(f"Successfully converted: {success_count} episodes")
        if fail_count > 0:
            logger.warn(f"Failed/Skipped: {fail_count} episodes")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * SEPARATOR_WIDTH)

        return success_count > 0

    def _convert_episode(self, episode_idx: int, chunk_idx: int, output_dir: Path) -> Path:
        """
        Convert a single episode to MCAP.
        Args:
            episode_idx: The episode index (used as file_index)
            chunk_idx: The chunk index
            output_dir: Output directory for MCAP files
        Raises:
            Exception: If episode conversion fails
        """
        # Check if episode files exist
        episode_files = self.dataset_info.get_episode_files(episode_idx, chunk_idx, self.dataset_root)

        if not episode_files["parquet"].exists():
            raise FileNotFoundError(f"Parquet file not found: {episode_files['parquet']}")

        # Check video files
        missing_videos = []
        for video_key, video_path in episode_files["videos"].items():
            if not video_path.exists():
                missing_videos.append(video_key)

        if missing_videos:
            logger.warn(f"Episode {episode_idx}: Missing videos: {missing_videos}. These will be skipped.")

        # Generate dynamic configuration for this episode
        episode_config = self.config_generator.generate_episode_config(episode_idx, chunk_idx)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config to temp file for tabular2mcap
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            yaml.dump(
                episode_config.model_dump(mode="python", exclude_none=True),
                config_file,
                default_flow_style=False,
                sort_keys=False,
            )
            config_path = Path(config_file.name)

        try:
            # Use SortedMcapConverter for timestamp-ordered MCAP output
            mcap_converter = SortedMcapConverter(config_path, self.converter_functions_path)

            # Output MCAP path: mcap_output/episode_000.mcap
            output_mcap = output_dir / f"episode_{episode_idx:03d}.mcap"

            logger.info(f"Converting episode {episode_idx} -> {output_mcap.name}")

            # Convert
            mcap_converter.convert(self.dataset_root, output_mcap)
        finally:
            # Clean up temp config file
            config_path.unlink(missing_ok=True)

        return episode_config

    def _convert_episode_v3(self, episode_idx: int, output_dir: Path) -> Path:
        """
        Convert a single episode to MCAP (v3.0 format).

        For v3.0 datasets, episodes are merged into shared files.
        This method extracts the episode data to temporary files before conversion.

        Uses simplified directory structure for clean topic names:
        - data.parquet -> /robot_data topic
        - videos/{camera_name}.mp4 -> /{camera_name} topic

        Args:
            episode_idx: The episode index
            output_dir: Output directory for MCAP files
        Raises:
            Exception: If episode conversion fails
        """
        from .video_loader_av import save_video_slice

        # Get episode info from metadata
        ep_info = self.dataset_info.get_episode_info(episode_idx)
        chunk_idx = ep_info["data_chunk_index"]

        # Get episode files (shared files with index ranges)
        episode_files = self.dataset_info.get_episode_files(episode_idx, chunk_idx, self.dataset_root)

        if not episode_files["parquet"].exists():
            raise FileNotFoundError(f"Parquet file not found: {episode_files['parquet']}")

        # Check video files
        missing_videos = []
        for video_key, video_path in episode_files["videos"].items():
            if not video_path.exists():
                missing_videos.append(video_key)

        if missing_videos:
            logger.warn(f"Episode {episode_idx}: Missing videos: {missing_videos}. These will be skipped.")

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for extracted episode data
        with tempfile.TemporaryDirectory(prefix=f"lerobot2mcap_ep{episode_idx}_") as temp_dir:
            temp_path = Path(temp_dir)

            # Extract episode data from shared parquet file
            # Split into separate files for semantic topic names:
            # - action -> /robot_state/action/data
            # - observation.state -> /observation/state/data
            self._extract_episode_parquet_split(
                episode_files["parquet"],
                episode_files["from_index"],
                episode_files["to_index"],
                temp_path,
                episode_idx,
            )

            # Extract video slices for each camera
            # Use semantic path structure: observation/images/{camera}.mp4
            # This creates topic: /observation/images/{camera}
            video_ranges = episode_files.get("video_ranges", {})

            for video_key, video_path in episode_files["videos"].items():
                if video_path.exists() and video_key in video_ranges:
                    # Convert video_key to path structure
                    # e.g., "observation.images.wrist_cam" -> "observation/images/wrist_cam.mp4"
                    video_rel_path = video_key.replace(".", "/") + ".mp4"
                    temp_video_path = temp_path / video_rel_path
                    temp_video_path.parent.mkdir(parents=True, exist_ok=True)

                    # Get timestamp range for this episode
                    video_range = video_ranges[video_key]
                    from_ts = video_range["from_timestamp"]
                    to_ts = video_range["to_timestamp"]

                    # Extract video slice
                    logger.debug(f"Extracting video slice for {video_key}: {from_ts:.3f}s - {to_ts:.3f}s")
                    save_video_slice(
                        video_path,
                        temp_video_path,
                        from_ts,
                        to_ts,
                        codec="libx264",
                    )

            # Copy log file if exists
            if self.log_file:
                temp_log_path = temp_path / self.log_file.name
                shutil.copy2(self.log_file, temp_log_path)

            # Generate v3.0 specific configuration with clean topic names
            episode_config = self.config_generator.generate_episode_config_v3(self.dataset_info.video_keys)

            # Save config to temp file for tabular2mcap
            config_path = temp_path / "config.yaml"
            with open(config_path, "w") as config_file:
                yaml.dump(
                    episode_config.model_dump(mode="python", exclude_none=True),
                    config_file,
                    default_flow_style=False,
                    sort_keys=False,
                )

            # Use SortedMcapConverter for timestamp-ordered MCAP output
            mcap_converter = SortedMcapConverter(config_path, self.converter_functions_path)

            # Output MCAP path: mcap_output/episode_000.mcap
            output_mcap = output_dir / f"episode_{episode_idx:03d}.mcap"

            logger.info(f"Converting episode {episode_idx} -> {output_mcap.name}")

            # Convert from temp directory
            # Use strip_file_suffix to drop .parquet from topic base
            mcap_converter.convert(temp_path, output_mcap, strip_file_suffix=True)

            return episode_config

    def _extract_episode_parquet(
        self,
        source_parquet: Path,
        from_index: int,
        to_index: int,
        temp_dir: Path,
        episode_idx: int,
        chunk_idx: int,
    ):
        """
        Extract episode data from a shared parquet file to a temporary file.
        (Legacy method for v2.x compatibility)

        Args:
            source_parquet: Path to the shared parquet file
            from_index: Start index (inclusive)
            to_index: End index (exclusive)
            temp_dir: Temporary directory to save extracted data
            episode_idx: Original episode index (for logging)
            chunk_idx: Chunk index (for path structure)
        """
        # Read the shared parquet file
        df = pd.read_parquet(source_parquet)

        # Filter rows by index range
        # The 'index' column contains the global dataset index
        episode_df = df[(df["index"] >= from_index) & (df["index"] < to_index)].copy()

        if episode_df.empty:
            raise ValueError(f"No data found for episode {episode_idx} (index range {from_index}:{to_index})")

        logger.debug(
            f"Extracted {len(episode_df)} rows for episode {episode_idx} "
            f"(index range {from_index}:{to_index})"
        )

        # Create the same directory structure as the original
        # Use chunk-000/file-000.parquet for the temp file
        temp_parquet_dir = temp_dir / "data" / "chunk-000"
        temp_parquet_dir.mkdir(parents=True, exist_ok=True)
        temp_parquet_path = temp_parquet_dir / "file-000.parquet"

        # Save extracted data
        episode_df.to_parquet(temp_parquet_path, index=False)
        logger.debug(f"Saved extracted parquet to {temp_parquet_path}")

    def _extract_episode_parquet_v3(
        self,
        source_parquet: Path,
        from_index: int,
        to_index: int,
        temp_dir: Path,
        episode_idx: int,
    ):
        """
        Extract episode data from a shared parquet file to a temporary file.
        (v3.0 format with semantic naming for clean topic names)

        Args:
            source_parquet: Path to the shared parquet file
            from_index: Start index (inclusive)
            to_index: End index (exclusive)
            temp_dir: Temporary directory to save extracted data
            episode_idx: Original episode index (for logging)
        """
        # Read the shared parquet file
        df = pd.read_parquet(source_parquet)

        # Filter rows by index range
        # The 'index' column contains the global dataset index
        episode_df = df[(df["index"] >= from_index) & (df["index"] < to_index)].copy()

        if episode_df.empty:
            raise ValueError(f"No data found for episode {episode_idx} (index range {from_index}:{to_index})")

        logger.debug(
            f"Extracted {len(episode_df)} rows for episode {episode_idx} "
            f"(index range {from_index}:{to_index})"
        )

        # Use semantic path: data.parquet -> /dataparquet/robot_state topic
        temp_parquet_path = temp_dir / "data.parquet"

        # Save extracted data
        episode_df.to_parquet(temp_parquet_path, index=False)
        logger.debug(f"Saved extracted parquet to {temp_parquet_path}")

    def _extract_episode_parquet_split(
        self,
        source_parquet: Path,
        from_index: int,
        to_index: int,
        temp_dir: Path,
        episode_idx: int,
    ):
        """
        Extract episode data and split into separate parquet files by data type.

        Creates:
        - robot_state/action.parquet -> /robot_state/action/data topic
        - observation/state.parquet -> /observation/state/data topic

        Args:
            source_parquet: Path to the shared parquet file
            from_index: Start index (inclusive)
            to_index: End index (exclusive)
            temp_dir: Temporary directory to save extracted data
            episode_idx: Original episode index (for logging)
        """
        # Read the shared parquet file
        df = pd.read_parquet(source_parquet)

        # Filter rows by index range
        episode_df = df[(df["index"] >= from_index) & (df["index"] < to_index)].copy()

        if episode_df.empty:
            raise ValueError(f"No data found for episode {episode_idx} (index range {from_index}:{to_index})")

        logger.debug(
            f"Extracted {len(episode_df)} rows for episode {episode_idx} "
            f"(index range {from_index}:{to_index})"
        )

        # Common metadata columns (included in all outputs)
        metadata_cols = [
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]

        # 1. Extract action data -> /robot/actions/data
        # File: robot/actions.parquet (strip suffix later for topic name)
        action_dir = temp_dir / "robot"
        action_dir.mkdir(parents=True, exist_ok=True)
        action_cols = ["action"] + [c for c in metadata_cols if c in episode_df.columns]
        action_df = episode_df[action_cols].copy()
        action_path = action_dir / "actions.parquet"
        action_df.to_parquet(action_path, index=False)
        logger.debug(f"Saved action data to {action_path}")

        # 2. Extract observation.state -> /robot/states/data
        state_dir = temp_dir / "robot"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_cols = ["observation.state"] + [c for c in metadata_cols if c in episode_df.columns]
        state_df = episode_df[state_cols].copy()
        state_path = state_dir / "states.parquet"
        state_df.to_parquet(state_path, index=False)
        logger.debug(f"Saved observation state to {state_path}")

    def _update_example_config(self, config_path: Path) -> None:
        """
        Copy the latest generated config to configs/config.yaml for reference.
        """
        dest = PROJECT_ROOT / "configs" / "config.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(config_path, dest)
            logger.info(f"Updated example config: {dest}")
        except Exception as e:
            logger.warn(f"Failed to update example config: {e}")

    def get_conversion_plan(self, chunks: list[int] | None = None) -> str:
        """
        Generate a human-readable conversion plan.
        Args:
            chunks: List of chunk indices to include in plan (None = all)
        Returns:
            Formatted string describing the conversion plan
        """
        if chunks is None:
            chunks = list(range(self.dataset_info.get_total_chunks()))

        plan = [
            "=" * SEPARATOR_WIDTH,
            "LeRobot to MCAP Conversion Plan",
            "=" * SEPARATOR_WIDTH,
            f"Dataset: {self.dataset_root.name}",
            f"Total episodes: {self.dataset_info.get_total_episodes()}",
            f"Total chunks: {self.dataset_info.get_total_chunks()}",
            f"FPS: {self.dataset_info.get_fps()}",
            f"Video streams: {len(self.dataset_info.video_keys)}",
        ]

        plan.extend(
            [
                f"Chunks to convert: {len(chunks)}",
                "",
            ]
        )

        # Show config for first chunk as example
        if chunks:
            plan.append(self.config_generator.generate_config_summary(chunks[0]))

        plan.append("=" * SEPARATOR_WIDTH)

        return "\n".join(plan)
