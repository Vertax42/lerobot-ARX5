"""LeRobot to MCAP converter.

Convert LeRobot datasets to MCAP format with automatic configuration generation
from dataset metadata. No manual configuration required – just point it at your
dataset and go.

This module is primarily focused on LeRobot v3.0 merged-format datasets
(parquet+video merged, per-episode slicing), while remaining backward
compatible with v2.1/v2.x. AV1 is supported via a PyAV monkey-patch.

Usage:
    # Download and convert from command line
    lerobot2mcap download lerobot/pusht

    # Convert existing dataset
    lerobot2mcap convert ~/.cache/huggingface/lerobot/pusht -o ./mcap_output

    # Python API
    from lerobot.datasets.lerobot2mcap import download_dataset, convert_dataset
    dataset_root = download_dataset("lerobot/pusht")
    convert_dataset(dataset_root, output_dir, converter_functions_path)
"""

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

# Fix tabular2mcap bugs
import tabular2mcap.converter.others
import tabular2mcap.mcap_converter
from tabular2mcap.converter.common import ConvertedRow

from lerobot.datasets.lerobot2mcap.converter import LeRobotConverter
from lerobot.datasets.lerobot2mcap.logger import get_logger
from lerobot.datasets.lerobot_dataset import LeRobotDataset

__version__ = "0.1.0"

logger = get_logger("lerobot2mcap")

# Get the package root directory
PACKAGE_ROOT = Path(__file__).parent
DEFAULT_CONVERTER_FUNCTIONS = str(PACKAGE_ROOT / "configs" / "converter_functions.yaml")


def _fixed_create_foxglove_compressed_image_data(
    frame_timestamp: float, frame_id: str, encoded_data: bytes, format: str
) -> dict:
    """Fixed version that uses 'nanosec' for ROS2 compatibility."""
    return {
        "timestamp": {
            "sec": int(frame_timestamp),
            "nanosec": int((frame_timestamp % 1) * 1_000_000_000),
        },
        "frame_id": frame_id,
        "data": encoded_data,
        "format": format,
    }


def _fixed_compressed_video_message_iterator(
    video_frames: list[np.ndarray],
    fps: float,
    format: str,
    frame_id: str,
    use_foxglove_format: bool = True,
    writer_format: str = "json",
) -> Iterable[ConvertedRow]:
    """
    Fixed version using ffmpeg to encode video with every frame as keyframe.
    Supports H.264 and AV1 formats. Each frame can be decoded independently.
    """
    import subprocess
    import tempfile

    height, width = video_frames[0].shape[:2]

    # Codec configuration for different formats
    codec_configs = {
        "h264": {
            "encoder": "libx264",
            "ext": "h264",
            "extra_args": [
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-bf",
                "0",
                "-flags",
                "+global_header",
                "-bsf:v",
                "dump_extra",
            ],
            "frame_marker": 7,  # SPS NAL type
        },
        "av1": {
            "encoder": "libsvtav1",
            "ext": "obu",
            "extra_args": [
                "-svtav1-params",
                "keyint=1:lookahead=0",
            ],
            "frame_marker": 1,  # OBU_SEQUENCE_HEADER
        },
    }

    # Default to h264 if format not supported
    if format not in codec_configs:
        format = "h264"

    config = codec_configs[format]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write frames as raw video
        raw_video = os.path.join(tmpdir, "raw.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(raw_video, fourcc, int(fps), (width, height))
        for frame in video_frames:
            writer.write(frame)
        writer.release()

        # Encode with ffmpeg: every frame is keyframe
        encoded_file = os.path.join(tmpdir, f"encoded.{config['ext']}")
        cmd = (
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_video,
                "-c:v",
                config["encoder"],
                "-pix_fmt",
                "yuv420p",
                "-g",
                "1",  # GOP=1, every frame is keyframe
                "-keyint_min",
                "1",
            ]
            + config["extra_args"]
            + [
                "-f",
                config["ext"] if format == "h264" else "ivf",
                "-loglevel",
                "error",
                encoded_file,
            ]
        )
        subprocess.run(cmd, check=True, capture_output=True)

        # Read encoded bitstream
        with open(encoded_file, "rb") as f:
            bitstream = f.read()

        # Split into frames based on format
        if format == "h264":
            frames_data = _split_h264_frames(bitstream)
        else:  # av1
            frames_data = _split_av1_frames(bitstream)

        # Generate messages
        frame_timestamp: float = 0
        frame_timestamp_step = 1 / fps

        for frame_data in frames_data:
            yield ConvertedRow(
                data=_fixed_create_foxglove_compressed_image_data(
                    frame_timestamp=frame_timestamp,
                    frame_id=frame_id,
                    encoded_data=frame_data,
                    format=format,
                ),
                log_time_ns=int(frame_timestamp * 1_000_000_000),
                publish_time_ns=int(frame_timestamp * 1_000_000_000),
            )
            frame_timestamp += frame_timestamp_step


def _split_h264_frames(bitstream: bytes) -> list[bytes]:
    """Split H.264 Annex B bitstream into frames. Each frame starts with SPS."""
    frames_data = []

    # Find all start code positions
    start_positions = []
    i = 0
    while i < len(bitstream) - 4:
        if bitstream[i : i + 4] == b"\x00\x00\x00\x01":
            start_positions.append(i)
            i += 4
        elif bitstream[i : i + 3] == b"\x00\x00\x01":
            start_positions.append(i)
            i += 3
        else:
            i += 1

    # Group NAL units into frames (each frame starts with SPS, NAL type 7)
    current_frame_start = None
    for pos in start_positions:
        # Get NAL type
        if bitstream[pos : pos + 4] == b"\x00\x00\x00\x01":
            nal_type = bitstream[pos + 4] & 0x1F
        else:
            nal_type = bitstream[pos + 3] & 0x1F

        if nal_type == 7:  # SPS - new frame starts
            if current_frame_start is not None:
                frames_data.append(bitstream[current_frame_start:pos])
            current_frame_start = pos

    # Don't forget the last frame
    if current_frame_start is not None:
        frames_data.append(bitstream[current_frame_start:])

    return frames_data


def _split_av1_frames(bitstream: bytes) -> list[bytes]:
    """Split AV1 IVF container into individual frames with sequence header."""
    import struct

    frames_data = []

    # IVF header is 32 bytes
    if len(bitstream) < 32:
        return frames_data

    # Parse IVF header
    signature = bitstream[0:4]
    if signature != b"DKIF":
        # Not IVF format, try raw OBU
        return _split_av1_obu_frames(bitstream)

    # Skip IVF header (32 bytes)
    pos = 32

    # Extract sequence header from first frame (we'll prepend it to all frames)
    seq_header = None

    while pos < len(bitstream):
        if pos + 12 > len(bitstream):
            break

        # IVF frame header: 4 bytes size, 8 bytes timestamp
        frame_size = struct.unpack("<I", bitstream[pos : pos + 4])[0]
        pos += 12  # Skip frame header

        if pos + frame_size > len(bitstream):
            break

        frame_data = bitstream[pos : pos + frame_size]

        # Check if this frame contains sequence header OBU
        if frame_data and (frame_data[0] & 0x78) >> 3 == 1:  # OBU_SEQUENCE_HEADER
            # Extract sequence header for prepending to other frames
            has_size = (frame_data[0] & 0x02) != 0
            if has_size and len(frame_data) > 1:
                # Find OBU size using LEB128
                obu_size = 0
                shift = 0
                idx = 1
                while idx < len(frame_data):
                    byte = frame_data[idx]
                    obu_size |= (byte & 0x7F) << shift
                    idx += 1
                    if (byte & 0x80) == 0:
                        break
                    shift += 7
                seq_header = frame_data[: idx + obu_size]

        # For all-keyframe mode, each frame should be independent
        # Prepend sequence header if not already present
        if seq_header and frame_data and (frame_data[0] & 0x78) >> 3 != 1:
            frame_data = seq_header + frame_data

        frames_data.append(frame_data)
        pos += frame_size

    return frames_data


def _split_av1_obu_frames(bitstream: bytes) -> list[bytes]:
    """Split raw AV1 OBU stream into frames."""
    # For raw OBU, just return the whole bitstream as one frame
    # This is a fallback - ideally we'd parse OBU properly
    return [bitstream] if bitstream else []


# Apply fixes to tabular2mcap
tabular2mcap.converter.others.create_foxglove_compressed_image_data = (
    _fixed_create_foxglove_compressed_image_data
)
tabular2mcap.converter.others.compressed_video_message_iterator = _fixed_compressed_video_message_iterator
tabular2mcap.mcap_converter.compressed_video_message_iterator = _fixed_compressed_video_message_iterator


def download_dataset(dataset_id: str, episodes: list[int] | None = None) -> Path | None:
    """
    Download a lerobot dataset from Hugging Face Hub.

    Downloads to default HuggingFace cache: ~/.cache/huggingface/lerobot

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "lerobot/pusht")
        episodes: Optional list of episode indices to download

    Returns:
        Path to the downloaded dataset root, or None if download failed.
    """
    logger.info(f"Downloading: {dataset_id}")
    if episodes:
        logger.info(f"  Episodes: {episodes}")

    try:
        # Use default cache location (don't pass root)
        dataset = LeRobotDataset(dataset_id, episodes=episodes)
        logger.info(
            f"Download complete - Episodes: {dataset.num_episodes}, "
            f"Frames: {dataset.num_frames}, FPS: {dataset.fps}"
        )
        # Return the actual dataset root path
        return Path(dataset.root)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def convert_dataset(
    dataset_root: Path,
    output_dir: Path,
    converter_functions_path: Path,
    chunks: list[int] | None = None,
    episodes: list[int] | None = None,
    num_workers: int = 1,
) -> bool:
    """
    Convert a LeRobot dataset to MCAP format.
    Iterates through each chunk and converts all episodes within that chunk.
    Each episode produces a separate MCAP file in its own directory.

    Args:
        dataset_root: Root directory of the LeRobot dataset
        output_dir: Output directory for MCAP files
        converter_functions_path: Path to converter_functions.yaml
        chunks: List of chunk indices to convert (None = all chunks)
        episodes: List of episode indices to convert (None = all episodes)
        num_workers: Number of parallel workers (default: 1)

    Returns:
        True if conversion succeeded, False otherwise
    """
    logger.info(f"Converting dataset: {dataset_root}")
    if episodes:
        logger.info(f"  Episodes: {episodes}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Converter functions: {converter_functions_path}")
    if num_workers > 1:
        logger.info(f"  Workers: {num_workers}")

    try:
        # Initialize the converter
        converter = LeRobotConverter(
            dataset_root=dataset_root, converter_functions_path=converter_functions_path
        )

        # Show conversion plan
        logger.info("\n" + converter.get_conversion_plan(chunks))

        # Perform conversion
        success = converter.convert(
            output_dir=output_dir,
            chunks=chunks,
            episodes=episodes,
            num_workers=num_workers,
        )

        return success

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return False


def main():
    """Main entry point for the lerobot2mcap CLI."""
    parser = argparse.ArgumentParser(
        prog="lerobot2mcap", description="Convert LeRobot datasets to MCAP format"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Default workers
    default_workers = max(1, os.cpu_count() // 4) if os.cpu_count() else 1

    # Define download parser arguments
    download_parser = subparsers.add_parser("download", help="Download a LeRobot dataset and convert to MCAP")
    download_parser.add_argument("dataset_id", help="Dataset ID (e.g., lerobot/pusht)")
    download_parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory for MCAP files (default: ./{dataset_name}_mcap)",
    )
    download_parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        nargs="+",
        help="Episode IDs to download (e.g., 0 1 2). If not specified, all episodes will be downloaded.",
    )
    download_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=default_workers,
        help=f"Number of parallel workers for conversion (default: {default_workers}, 1/4 of CPU cores)",
    )

    # Define convert parser
    convert_parser = subparsers.add_parser("convert", help="Convert a LeRobot dataset to MCAP format")
    convert_parser.add_argument(
        "input_dir",
        help="Input directory containing LeRobot dataset (dataset root with meta/info.json)",
    )
    convert_parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory for MCAP files (default: input_dir/mcap)",
    )
    convert_parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        nargs="+",
        help="Episode IDs to convert (e.g., 0 1 2). If not specified, all episodes will be converted.",
    )
    convert_parser.add_argument(
        "-f",
        "--converter-functions",
        default=DEFAULT_CONVERTER_FUNCTIONS,
        help=f"Path to converter_functions.yaml file (default: {DEFAULT_CONVERTER_FUNCTIONS})",
    )
    convert_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=default_workers,
        help=f"Number of parallel workers for conversion (default: {default_workers}, 1/4 of CPU cores)",
    )

    args = parser.parse_args()

    # Handle download command
    if args.command == "download":
        # Download to default HuggingFace cache (~/.cache/huggingface/lerobot)
        dataset_root = download_dataset(args.dataset_id, args.episodes)
        if dataset_root is None:
            return 1  # Download failed

        logger.info(f"Dataset location: {dataset_root}")

        # MCAP output directory: use -o or default to ./{dataset_name}_mcap
        dataset_name = args.dataset_id.replace("/", "_")
        mcap_output_dir = (
            Path(args.output_dir).expanduser() if args.output_dir else Path(f"./{dataset_name}_mcap")
        )
        converter_functions = Path(DEFAULT_CONVERTER_FUNCTIONS)
        chunks = None  # Convert all chunks
        episodes = args.episodes  # Use same episode filter as download
        num_workers = args.jobs

    elif args.command == "convert":
        # Set parameters from convert command arguments
        dataset_root = Path(args.input_dir).expanduser()
        mcap_output_dir = (
            Path(args.output_dir).expanduser() if args.output_dir else dataset_root / "mcap_conversion"
        )
        converter_functions = Path(args.converter_functions).expanduser()
        chunks = None  # Convert all chunks
        episodes = args.episodes
        num_workers = args.jobs

    else:
        # No command provided
        parser.print_help()
        return 0

    # Perform conversion (always happens after download, or standalone)
    if convert_dataset(
        dataset_root,
        mcap_output_dir,
        converter_functions,
        chunks,
        episodes,
        num_workers,
    ):
        return 0
    else:
        return 1


# Public API
__all__ = [
    "download_dataset",
    "convert_dataset",
    "LeRobotConverter",
    "main",
    "__version__",
    "DEFAULT_CONVERTER_FUNCTIONS",
]
