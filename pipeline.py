"""Detection pipeline for ipcam-bird-detection.

This module encapsulates the video-to-detection pipeline, independent of
any specific video source (API, local file, etc.).
"""

from dataclasses import dataclass
from pathlib import Path

from detector import BirdDetector
from frame_extractor import extract_frames


@dataclass
class PipelineResult:
    """Complete result from the detection pipeline."""

    # Detection results
    has_bird: bool
    confidence: float | None = None
    bird_area_percent: float | None = None

    # Video/frame metadata
    video_duration: float | None = None
    frame_time: float | None = None  # Frame time where bird was detected

    # Error info (if failed)
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether the pipeline completed successfully."""
        return self.error is None


class DetectionPipeline:
    """Pipeline for detecting birds in videos.

    Handles frame extraction and bird detection, independent of video source.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.5,
        frame_times: list[float] | None = None,
        min_area_percent: float | None = None,
        max_area_percent: float | None = None,
    ):
        """Initialize the pipeline.

        Args:
            model_path: Path to YOLO model or model name.
            confidence_threshold: Minimum confidence for detections.
            frame_times: Target frame times in seconds (uses 50% if video shorter).
            min_area_percent: Minimum bird area as % of frame (None = no minimum).
            max_area_percent: Maximum bird area as % of frame (None = no maximum).
        """
        self.detector = BirdDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            min_area_percent=min_area_percent,
            max_area_percent=max_area_percent,
        )
        self.frame_times = frame_times if frame_times is not None else [6.0]

    def process(self, video_path: Path) -> PipelineResult:
        """Process a video file through the detection pipeline.

        Extracts frames at each configured time and checks for birds.
        If any frame contains a bird, returns the detection with the
        largest bird (by area percentage).

        Args:
            video_path: Path to the video file.

        Returns:
            PipelineResult with detection results or error info.
        """
        frame_paths = []

        try:
            # Extract frames
            extraction = extract_frames(video_path, target_times=self.frame_times)
            frame_paths = [f.frame_path for f in extraction.frames]

            # Run detection on each frame, track best result
            best_detection = None
            best_frame_time = None

            for frame_result in extraction.frames:
                detection = self.detector.detect(frame_result.frame_path)

                if detection.has_bird:
                    # Keep the detection with the largest bird
                    if best_detection is None or (
                        detection.bird_area_percent is not None
                        and (
                            best_detection.bird_area_percent is None
                            or detection.bird_area_percent
                            > best_detection.bird_area_percent
                        )
                    ):
                        best_detection = detection
                        best_frame_time = frame_result.frame_time

            if best_detection is not None:
                return PipelineResult(
                    has_bird=True,
                    confidence=best_detection.confidence,
                    bird_area_percent=best_detection.bird_area_percent,
                    video_duration=extraction.duration,
                    frame_time=best_frame_time,
                )
            else:
                return PipelineResult(
                    has_bird=False,
                    video_duration=extraction.duration,
                    frame_time=extraction.frames[0].frame_time
                    if extraction.frames
                    else None,
                )

        except Exception as e:
            return PipelineResult(
                has_bird=False,
                error=str(e),
            )

        finally:
            # Clean up extracted frames
            for frame_path in frame_paths:
                if frame_path.exists():
                    frame_path.unlink(missing_ok=True)
