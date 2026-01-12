"""Sorted MCAP converter that ensures messages are written in timestamp order."""

from pathlib import Path
from typing import Any

from mcap.writer import Writer as McapWriter
from mcap_ros2.writer import Writer as McapRos2Writer
from tabular2mcap.converter import (
    ConvertedRow,
    JsonConverter,
    Ros2Converter,
)
from tabular2mcap.converter.others import (
    compressed_video_message_iterator,
)
from tabular2mcap.loader import (
    CompressedImageMappingConfig,
    CompressedVideoMappingConfig,
    load_video_data,
)
from tabular2mcap.mcap_converter import McapConverter
from tqdm import tqdm

from lerobot.datasets.lerobot2mcap.logger import get_logger

logger = get_logger("sorted_mcap_converter")


class SortedMcapConverter(McapConverter):
    """
    MCAP converter that collects all messages and writes them in sorted timestamp order.

    MCAP requires messages to be written with monotonically increasing log_time.
    This converter buffers all messages from all sources (tabular + video) and
    sorts them by timestamp before writing.
    """

    def convert(
        self,
        input_path: Path,
        output_path: Path,
        topic_prefix: str = "",
        test_mode: bool = False,
        best_effort: bool = False,
        strip_file_suffix: bool = False,
    ) -> None:
        """
        Convert tabular and multimedia data to MCAP format with sorted timestamps.

        Overrides the parent class to collect all messages first, sort by timestamp,
        then write in sorted order.
        """
        logger.info(f"Input directory: {input_path}")
        logger.info(f"Output MCAP: {output_path}")
        logger.info("Using sorted timestamp mode for MCAP compliance")

        if self.mcap_config.writer_format not in ["json", "ros2"]:
            raise ValueError(
                f"Writer format {self.mcap_config.writer_format} is not supported by SortedMcapConverter"
            )

        # Collect all messages with their metadata
        all_messages: list[tuple[int, str, int, ConvertedRow]] = []  # (log_time_ns, topic, schema_id, row)

        # Prepare data for processing
        mapping_tuples = self._prepare_mapping_tuples(input_path)

        # Temporary storage for schema info
        self._pending_schemas: dict[str, tuple[int, Any]] = {}  # topic -> (schema_id, schema_info)

        with open(output_path, "wb") as f:
            if self.mcap_config.writer_format == "json":
                self._writer = McapWriter(f)
                self._writer.start()
                self._converter = JsonConverter(self._writer)
            elif self.mcap_config.writer_format == "ros2":
                self._writer = McapRos2Writer(f)
                self._converter = Ros2Converter(self._writer)

            # Print conversion plan
            logger.info("\n" + "=" * 60)
            logger.info("MCAP Conversion Plan (Sorted Mode)")
            logger.info("=" * 60)
            logger.info(f"Tabular mappings:      {len(self.mcap_config.tabular_mappings)}")
            logger.info(f"Other mappings:        {len(self.mcap_config.other_mappings)}")
            logger.info(f"Attachments:           {len(self.mcap_config.attachments)}")
            logger.info("=" * 60 + "\n")

            # Collect tabular messages
            self._collect_tabular_messages(
                mapping_tuples["tabular"],
                input_path,
                topic_prefix,
                test_mode,
                best_effort,
                strip_file_suffix,
                all_messages,
            )

            # Collect video messages
            self._collect_video_messages(
                mapping_tuples["other"],
                input_path,
                topic_prefix,
                best_effort,
                all_messages,
            )

            # Sort all messages by log_time_ns
            logger.info(f"Sorting {len(all_messages)} messages by timestamp...")
            all_messages.sort(key=lambda x: x[0])

            # Write sorted messages
            logger.info("Writing sorted messages to MCAP...")
            self._write_sorted_messages(all_messages)

            # Process attachments and metadata (these don't affect message ordering)
            self._process_attachments(mapping_tuples["attachments"], input_path, best_effort)
            self._process_metadata(mapping_tuples["metadata"], input_path, best_effort)

            # Finish writing
            print("\n" + "=" * 60)
            print("Finalizing MCAP file...")
            print("=" * 60)
            self._writer.finish()

            # Print summary
            print("\n" + "=" * 60)
            print("[OK] Conversion completed successfully!")
            print(f"[OK] Output file: {output_path}")
            print(f"[OK] File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
            print(f"[OK] Total messages: {len(all_messages)}")
            print("=" * 60)

    def _collect_tabular_messages(
        self,
        mapping_tuples: list,
        input_path: Path,
        topic_prefix: str,
        test_mode: bool,
        best_effort: bool,
        strip_file_suffix: bool,
        all_messages: list,
    ):
        """Collect tabular data messages without writing them."""
        for file_mapping, input_file in tqdm(
            mapping_tuples,
            desc="Collecting tabular data",
            leave=False,
            unit="file",
        ):
            try:
                relative_path = input_file.relative_to(input_path)
                df = self._load_dataframe(input_file)

                if test_mode:
                    df = df.head(5)

                path_str = str(relative_path.with_suffix("")) if strip_file_suffix else str(relative_path)
                relative_path_no_ext = self._clean_string(path_str)
                topic_base = f"{topic_prefix}{relative_path_no_ext}/"

                for converter_function in file_mapping.converter_functions:
                    topic_name = f"{topic_base}{converter_function.topic_suffix}"

                    # Get converter definition
                    if converter_function.function_name not in self.converter_functions:
                        raise ValueError(f"Unknown converter function: {converter_function.function_name}")
                    converter_def = self.converter_functions[converter_function.function_name]

                    # Register schema and get convert_row function
                    schema_id, convert_row = self._register_schema(
                        df, topic_name, converter_function, converter_def
                    )

                    if schema_id is None:
                        continue

                    # Collect messages
                    for _, row in df.iterrows():
                        converted = convert_row(row)
                        all_messages.append(
                            (
                                converted.log_time_ns,
                                topic_name,
                                schema_id,
                                converted,
                            )
                        )

            except Exception as e:
                if best_effort:
                    logger.exception(f"Error collecting tabular file {input_file}: {e}")
                else:
                    raise

    def _collect_video_messages(
        self,
        mapping_tuples: list,
        input_path: Path,
        topic_prefix: str,
        best_effort: bool,
        all_messages: list,
    ):
        """Collect video messages without writing them."""
        for other_mapping, input_file in tqdm(
            mapping_tuples,
            desc="Collecting video data",
            leave=False,
            unit="file",
        ):
            try:
                if not isinstance(
                    other_mapping,
                    (CompressedImageMappingConfig, CompressedVideoMappingConfig),
                ):
                    continue

                relative_path = input_file.relative_to(input_path)
                relative_path_no_ext = self._clean_string(str(relative_path.with_suffix("")))
                topic_name_prefix = f"{topic_prefix}{relative_path_no_ext}/"
                topic_name = f"{topic_name_prefix}{other_mapping.topic_suffix}"

                # Get or register schema
                schema_name = other_mapping.set_default_schema_name(self.mcap_config.writer_format)
                if schema_name in self._schema_ids:
                    schema_id = self._schema_ids[schema_name]
                else:
                    schema_id = self._converter.register_schema(schema_name=schema_name)
                    self._schema_ids[schema_name] = schema_id

                # Load video data
                video_frames, video_properties = load_video_data(input_file)
                logger.debug(f"Loaded video: {input_file.name}, {len(video_frames)} frames")

                # Generate frame messages
                frame_iterator = compressed_video_message_iterator(
                    video_frames=video_frames,
                    fps=video_properties["fps"],
                    format=other_mapping.format,
                    frame_id=other_mapping.frame_id,
                    use_foxglove_format=schema_name.startswith("foxglove"),
                    writer_format=self.mcap_config.writer_format,
                )

                # Collect messages
                for converted in frame_iterator:
                    all_messages.append(
                        (
                            converted.log_time_ns,
                            topic_name,
                            schema_id,
                            converted,
                        )
                    )

            except Exception as e:
                if best_effort:
                    logger.error(f"Error collecting video {input_file}: {e}")
                else:
                    raise

    def _write_sorted_messages(self, all_messages: list):
        """Write all collected messages in sorted order.

        Uses the same write pattern as tabular2mcap's write_messages_from_iterator
        to support both JSON and ROS2 formats.
        """
        # Track message sequence per topic
        topic_sequences: dict[str, int] = {}

        for log_time_ns, topic_name, schema_id, converted in tqdm(
            all_messages,
            desc="Writing sorted messages",
            unit="msg",
        ):
            # Get or initialize sequence for this topic
            if topic_name not in topic_sequences:
                topic_sequences[topic_name] = 0
            sequence = topic_sequences[topic_name]
            topic_sequences[topic_name] += 1

            # Write message using the converter's writer
            # This works for both JSON and ROS2 formats
            self._writer.write_message(
                topic=topic_name,
                schema=schema_id,
                message=converted.data,
                log_time=log_time_ns,
                publish_time=converted.publish_time_ns,
                sequence=sequence,
            )
