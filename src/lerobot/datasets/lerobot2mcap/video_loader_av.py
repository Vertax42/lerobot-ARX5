"""Video utilities for LeRobot v3.0 dataset conversion.

Provides video slicing functionality for LeRobot v3.0 datasets
where multiple episodes are merged into single video files.
"""

import subprocess
from pathlib import Path

from lerobot.datasets.lerobot2mcap.logger import get_logger

logger = get_logger("video_loader")


def save_video_slice(
    source_path: Path,
    output_path: Path,
    from_timestamp: float,
    to_timestamp: float,
    codec: str = "libx264",
) -> None:
    """
    Extract a slice of video and save it to a new file using ffmpeg.

    Args:
        source_path: Path to the source video file
        output_path: Path to save the output video
        from_timestamp: Start timestamp in seconds
        to_timestamp: End timestamp in seconds
        codec: Output codec (default: libx264 for H.264)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate duration
    duration = to_timestamp - from_timestamp

    # Use ffmpeg command line for reliable video slicing
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-ss",
        str(from_timestamp),  # Start time (before -i for fast seek)
        "-i",
        str(source_path),  # Input file
        "-t",
        str(duration),  # Duration
        "-c:v",
        codec,  # Video codec
        "-pix_fmt",
        "yuv420p",  # Pixel format
        "-an",  # No audio
        "-loglevel",
        "error",  # Reduce verbosity
        str(output_path),  # Output file
    ]

    logger.debug(f"Running ffmpeg: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}: {result.stderr}")

    logger.debug(f"Saved video slice to {output_path} ({from_timestamp:.3f}s - {to_timestamp:.3f}s)")
