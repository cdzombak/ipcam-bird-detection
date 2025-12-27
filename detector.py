"""Bird detection using Ultralytics YOLO."""

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

# COCO class ID for "bird"
BIRD_CLASS_ID = 14


@dataclass
class DetectionResult:
    """Result of bird detection."""

    has_bird: bool
    confidence: float | None = None
    bird_area_percent: float | None = None


class BirdDetector:
    """Detect birds in images using YOLO."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.5,
        min_area_percent: float | None = None,
        max_area_percent: float | None = None,
    ):
        """Initialize the detector.

        Args:
            model_path: Path to YOLO model or model name.
            confidence_threshold: Minimum confidence for detections.
            min_area_percent: Minimum bird area as % of frame (None = no minimum).
            max_area_percent: Maximum bird area as % of frame (None = no maximum).
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.min_area_percent = min_area_percent
        self.max_area_percent = max_area_percent

    def detect(self, image_path: Path) -> DetectionResult:
        """Detect birds in an image.

        Args:
            image_path: Path to the image file.

        Returns:
            DetectionResult with detection info.
        """
        # Run inference
        results = self.model(str(image_path), verbose=False)

        if not results or len(results) == 0:
            return DetectionResult(has_bird=False)

        result = results[0]

        # Get image dimensions for area calculation
        img_height, img_width = result.orig_shape

        # Filter for bird detections above threshold
        birds = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            if cls_id == BIRD_CLASS_ID and conf >= self.confidence_threshold:
                # Get bounding box coordinates (xyxy format)
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy

                # Calculate area
                bird_area = (x2 - x1) * (y2 - y1)
                frame_area = img_width * img_height
                area_percent = (bird_area / frame_area) * 100

                # Filter by area thresholds
                if self.min_area_percent is not None and area_percent < self.min_area_percent:
                    continue
                if self.max_area_percent is not None and area_percent > self.max_area_percent:
                    continue

                birds.append({
                    "confidence": conf,
                    "area_percent": area_percent,
                    "area": bird_area,
                })

        if not birds:
            return DetectionResult(has_bird=False)

        # Find the largest bird by area
        largest_bird = max(birds, key=lambda b: b["area"])

        return DetectionResult(
            has_bird=True,
            confidence=largest_bird["confidence"],
            bird_area_percent=largest_bird["area_percent"],
        )
