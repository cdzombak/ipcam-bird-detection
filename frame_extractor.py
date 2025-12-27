"""Video frame extraction using ffmpeg."""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


class FrameExtractionError(Exception):
    """Error during frame extraction."""


@dataclass
class ExtractionResult:
    """Result of frame extraction."""

    frame_path: Path
    duration: float
    frame_time: float


def get_video_duration(video_path: Path) -> float:
    """Get the duration of a video file using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds.

    Raises:
        FrameExtractionError: If ffprobe fails.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration_str = data.get("format", {}).get("duration")
        if duration_str is None:
            raise FrameExtractionError("Could not determine video duration")
        return float(duration_str)
    except subprocess.CalledProcessError as e:
        raise FrameExtractionError(f"ffprobe failed: {e.stderr}") from e
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise FrameExtractionError(f"Failed to parse ffprobe output: {e}") from e


def extract_frame(
    video_path: Path,
    target_time: float = 6.0,
) -> ExtractionResult:
    """Extract a single frame from a video.

    Extracts frame at target_time seconds, or at 50% if video is shorter.

    Args:
        video_path: Path to the video file.
        target_time: Target time in seconds (default 6.0).

    Returns:
        ExtractionResult with frame path, duration, and actual frame time.

    Raises:
        FrameExtractionError: If extraction fails.
    """
    duration = get_video_duration(video_path)

    # Determine actual frame time
    if duration < target_time:
        frame_time = duration * 0.5
    else:
        frame_time = target_time

    # Create temp file for the frame
    fd, frame_path = tempfile.mkstemp(suffix=".jpg")
    # Close the file descriptor since ffmpeg will write to the path
    import os

    os.close(fd)
    frame_path = Path(frame_path)

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-ss",
        str(frame_time),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",  # High quality JPEG
        str(frame_path),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        frame_path.unlink(missing_ok=True)
        raise FrameExtractionError(f"ffmpeg failed: {e.stderr}") from e

    if not frame_path.exists() or frame_path.stat().st_size == 0:
        frame_path.unlink(missing_ok=True)
        raise FrameExtractionError("ffmpeg produced no output")

    return ExtractionResult(
        frame_path=frame_path,
        duration=duration,
        frame_time=frame_time,
    )
