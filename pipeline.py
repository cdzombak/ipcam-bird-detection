"""Detection pipeline for ipcam-bird-detection.

This module encapsulates the video-to-detection pipeline, independent of
any specific video source (API, local file, etc.).
"""

from dataclasses import dataclass
from pathlib import Path

from detector import BirdDetector, DetectionResult
from frame_extractor import ExtractionResult, extract_frame


@dataclass
class PipelineResult:
    """Complete result from the detection pipeline."""

    # Detection results
    has_bird: bool
    confidence: float | None = None
    bird_area_percent: float | None = None

    # Video/frame metadata
    video_duration: float | None = None
    frame_time: float | None = None

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
        frame_time: float = 6.0,
        min_area_percent: float | None = None,
        max_area_percent: float | None = None,
    ):
        """Initialize the pipeline.

        Args:
            model_path: Path to YOLO model or model name.
            confidence_threshold: Minimum confidence for detections.
            frame_time: Target frame time in seconds (uses 50% if video shorter).
            min_area_percent: Minimum bird area as % of frame (None = no minimum).
            max_area_percent: Maximum bird area as % of frame (None = no maximum).
        """
        self.detector = BirdDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            min_area_percent=min_area_percent,
            max_area_percent=max_area_percent,
        )
        self.frame_time = frame_time

    def process(self, video_path: Path) -> PipelineResult:
        """Process a video file through the detection pipeline.

        Args:
            video_path: Path to the video file.

        Returns:
            PipelineResult with detection results or error info.
        """
        frame_path = None

        try:
            # Extract frame
            extraction = extract_frame(video_path, target_time=self.frame_time)
            frame_path = extraction.frame_path

            # Run detection
            detection = self.detector.detect(frame_path)

            return PipelineResult(
                has_bird=detection.has_bird,
                confidence=detection.confidence,
                bird_area_percent=detection.bird_area_percent,
                video_duration=extraction.duration,
                frame_time=extraction.frame_time,
            )

        except Exception as e:
            return PipelineResult(
                has_bird=False,
                error=str(e),
            )

        finally:
            # Clean up extracted frame
            if frame_path and frame_path.exists():
                frame_path.unlink(missing_ok=True)
