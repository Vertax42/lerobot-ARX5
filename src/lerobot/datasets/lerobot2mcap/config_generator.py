"""Configuration generator for tabular2mcap conversion."""

from tabular2mcap.loader.models import (
    AttachmentConfig,
    CompressedVideoMappingConfig,
    ConverterFunctionConfig,
    McapConversionConfig,
    TabularMappingConfig,
)

from lerobot.datasets.lerobot2mcap.dataset_info import DEFAULT_CODEC, DatasetInfo
from lerobot.datasets.lerobot2mcap.logger import get_logger

logger = get_logger("config_generator")


class ConfigGenerator:
    """
    Generates tabular2mcap configuration for each episodes based on metadata info.

    Configurations are created per-episode (not per-chunk), with specific
    file paths for each episode's data files, videos, and optional attachments.
    """

    def __init__(self, dataset_info: DatasetInfo):
        """
        Initialize ConfigGenerator with dataset information.
        Args:
            dataset_info: DatasetInfo instance containing dataset metadata
        """
        self.dataset_info = dataset_info

        # Create a base template config with default values from metadata
        # This will be copied and modified for each episode
        self._base_config_template = McapConversionConfig(
            writer_format=dataset_info.get_writer_format(),
            tabular_mappings=[],
            other_mappings=[],
            attachments=[],
            metadata=[],
        )

        logger.info(
            f"Initialized ConfigGenerator for dataset with {len(dataset_info.video_keys)} video streams"
        )

    @staticmethod
    def _format_episode_pattern(template: str, chunk_index: int, episode_index: int, **kwargs) -> str:
        """
        Format a file path template for a specific episode (no wildcards).
        Supports both v2.1 and v3 LeRobot formats.
        Args:
            template: Path template with placeholders
            chunk_index: The chunk index to substitute
            episode_index: The episode index to use as file_index
            **kwargs: Additional key-value pairs to substitute in the template
        Returns:
            Formatted pattern with specific file path
        """
        # Support both v2.1 and v3 naming conventions
        format_params = {
            "episode_chunk": chunk_index,  # v2.1 format
            "episode_index": episode_index,  # v2.1 format
            "chunk_index": chunk_index,  # v3 format
            "file_index": episode_index,  # v3 format
            **kwargs,
        }

        # Format with specific file_index (episode_index)
        formatted = template.format(**format_params)
        return formatted

    def generate_episode_config(
        self, episode_index: int, chunk_index: int, include_log: bool = True
    ) -> McapConversionConfig:
        """
        Generate configuration for a specific episode using Pydantic models.

        Builds from the base template and updates with episode-specific mappings.

        Args:
            episode_index: The episode index (used as file_index in paths)
            chunk_index: The chunk index
            include_log: Whether to include log file as MCAP attachment (default: True,
                tabular2mcap will auto-detect if the file exists)

        Returns:
            McapConversionConfig Pydantic model with validated configuration
        """
        # Generate episode-specific mappings
        tabular_mappings = self._generate_tabular_mappings_for_episode(episode_index, chunk_index)
        other_mappings = self._generate_video_mappings_for_episode(episode_index, chunk_index)

        # Generate attachments (log file if requested)
        attachments = []
        if include_log:
            attachments.append(
                AttachmentConfig(
                    file_pattern="**/*.log",
                    mime_type="text/plain",
                )
            )

        # Build from template using model_copy with updates
        config = self._base_config_template.model_copy(
            update={
                "tabular_mappings": tabular_mappings,
                "other_mappings": other_mappings,
                "attachments": attachments,
            }
        )

        logger.debug(
            f"Generated config for episode {episode_index} in chunk {chunk_index}: "
            f"{len(config.tabular_mappings)} tabular, "
            f"{len(config.other_mappings)} other mappings, "
            f"{len(config.attachments)} attachments"
        )

        return config

    def _generate_tabular_mappings_for_episode(
        self, episode_index: int, chunk_index: int
    ) -> list[TabularMappingConfig]:
        """
        Generate tabular mapping configurations for a specific episode.
        Creates mappings for a specific parquet data file (not wildcards).
        Args:
            episode_index: The episode index (used as file_index)
            chunk_index: The chunk index
        Returns:
            List of tabular mapping configurations using Pydantic models
        """
        mappings = []

        # Generate parquet file pattern for specific episode (no wildcards)
        parquet_pattern = self._format_episode_pattern(
            self.dataset_info.data_path_template, chunk_index, episode_index
        )

        mappings.append(
            TabularMappingConfig(
                file_pattern=f"**/{parquet_pattern}",
                converter_functions=[
                    ConverterFunctionConfig(
                        function_name="row_to_message_with_timestamp",
                        schema_name=None,  # Will auto-generate schema
                        topic_suffix="robot_data",
                        exclude_columns=["timestamp"],
                    )
                ],
            )
        )

        return mappings

    def _generate_video_mappings_for_episode(
        self, episode_index: int, chunk_index: int
    ) -> list[CompressedVideoMappingConfig]:
        """
        Generate video mapping configurations for a specific episode.
        Args:
            episode_index: The episode index (used as file_index)
            chunk_index: The chunk index
        Returns:
            List of CompressedVideoMappingConfig Pydantic models
        """
        mappings = []

        for video_key in self.dataset_info.video_keys:
            # Always use h264 codec for optimal speed
            video_format = DEFAULT_CODEC

            # Generate video file pattern for specific episode (no wildcards)
            video_pattern = self._format_episode_pattern(
                self.dataset_info.video_path_template,
                chunk_index,
                episode_index,
                video_key=video_key,
            )

            # Generate topic suffix from video_key
            # "observation.images.front" -> "observation/images/front"
            topic_suffix = video_key.replace(".", "/")

            # Get frame_id
            frame_id = self.dataset_info.get_video_frame_id(video_key)

            mappings.append(
                CompressedVideoMappingConfig(
                    file_pattern=f"**/{video_pattern}",
                    topic_suffix=topic_suffix,
                    frame_id=frame_id,
                    format=video_format,
                )
            )

        return mappings

    def generate_episode_config_v3(
        self, video_keys: list[str], include_log: bool = True
    ) -> McapConversionConfig:
        """
        Generate configuration for v3.0 format with semantic topic names.

        Uses directory structure to generate clean topic names:
        - robot_data.parquet -> /robot_data
        - observation/images/wrist_cam.mp4 -> /observation/images/wrist_cam

        The topic name is derived from the file path (without extension),
        so we use empty topic_suffix and rely on file structure.

        Args:
            video_keys: List of video keys (e.g., ["observation.images.wrist_cam"])
            include_log: Whether to include log file as MCAP attachment

        Returns:
            McapConversionConfig Pydantic model with validated configuration
        """
        # Tabular mappings: Split parquet data into separate topics
        # Use strip_file_suffix=True in converter to remove .parquet from topic base
        tabular_mappings = [
            # Action data: /robot/actions/data
            TabularMappingConfig(
                file_pattern="robot/actions.parquet",
                converter_functions=[
                    ConverterFunctionConfig(
                        function_name="row_to_message_with_timestamp",
                        schema_name=None,
                        topic_suffix="data",  # /robot/actions/data
                        exclude_columns=["timestamp"],
                    )
                ],
            ),
            # Observation state: /robot/states/data
            TabularMappingConfig(
                file_pattern="robot/states.parquet",
                converter_functions=[
                    ConverterFunctionConfig(
                        function_name="row_to_message_with_timestamp",
                        schema_name=None,
                        topic_suffix="data",  # /robot/states/data
                        exclude_columns=["timestamp"],
                    )
                ],
            ),
        ]

        # Video mappings: observation/images/{camera}.mp4 -> /observation/images/{camera}/video
        other_mappings = []
        for video_key in video_keys:
            # Convert video_key to path: "observation.images.wrist_cam" -> "observation/images/wrist_cam"
            video_path = video_key.replace(".", "/")

            # Get frame_id
            frame_id = self.dataset_info.get_video_frame_id(video_key)

            other_mappings.append(
                CompressedVideoMappingConfig(
                    file_pattern=f"{video_path}.mp4",
                    topic_suffix="video",  # /observation/images/wrist_cam/video
                    frame_id=frame_id,
                    format=DEFAULT_CODEC,
                )
            )

        # Attachments
        attachments = []
        if include_log:
            attachments.append(
                AttachmentConfig(
                    file_pattern="**/*.log",
                    mime_type="text/plain",
                )
            )

        # Build config
        config = self._base_config_template.model_copy(
            update={
                "tabular_mappings": tabular_mappings,
                "other_mappings": other_mappings,
                "attachments": attachments,
            }
        )

        logger.debug(
            f"Generated v3.0 config: "
            f"{len(config.tabular_mappings)} tabular, "
            f"{len(config.other_mappings)} video mappings"
        )

        return config

    def generate_config_summary(self, chunk_index: int) -> str:
        """
        Generate a human-readable summary of the configuration.

        Shows an example episode configuration (episode 0) to give users
        a preview of what will be generated for each episode in the chunk.

        Args:
            chunk_index: The chunk index

        Returns:
            Formatted string describing the configuration
        """
        # Generate an example config using episode 0
        config = self.generate_episode_config(episode_index=0, chunk_index=chunk_index, include_log=True)

        summary = [
            f"Example configuration for chunk-{chunk_index:03d} (episode 0):",
            f"  Writer format: {config.writer_format}",
            f"  Tabular mappings: {len(config.tabular_mappings)}",
        ]

        for i, mapping in enumerate(config.tabular_mappings, 1):
            summary.append(f"    {i}. {mapping.file_pattern}")

        summary.append(f"  Other mappings: {len(config.other_mappings)}")
        for i, mapping in enumerate(config.other_mappings, 1):
            mapping_type = mapping.type if hasattr(mapping, "type") else "unknown"
            topic = mapping.topic_suffix
            if mapping_type == "compressed_video":
                details = f"{topic} ({mapping.format}, frame_id={mapping.frame_id})"
            else:
                details = topic
            summary.append(f"    {i}. [{mapping_type}] {details}")

        if config.attachments:
            summary.append(f"  Attachments: {len(config.attachments)}")
            for i, attachment in enumerate(config.attachments, 1):
                mime_type = attachment.mime_type or "auto-detected"
                summary.append(f"    {i}. {attachment.file_pattern} (MIME: {mime_type})")

        return "\n".join(summary)
