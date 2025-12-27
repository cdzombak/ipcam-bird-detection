"""Configuration loader for ipcam-bird-detection."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ApiConfig:
    """API connection configuration."""

    base_url: str
    timeout: int = 30


@dataclass
class DetectionConfig:
    """Bird detection configuration."""

    model: str = "yolo11n.pt"
    confidence_threshold: float = 0.5
    frame_time: float = 6.0
    min_area_percent: float | None = None  # Minimum bird area as % of frame
    max_area_percent: float | None = None  # Maximum bird area as % of frame


@dataclass
class DatabaseConfig:
    """Database configuration."""

    path: str = "bird_detections.db"


@dataclass
class OutputsConfig:
    """Output configuration."""

    directory: str | None = None  # Directory to save videos with birds


@dataclass
class Config:
    """Application configuration."""

    api: ApiConfig
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)


def load_config(config_path: str | Path) -> Config:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed Config object.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required configuration is missing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError("Configuration file is empty")

    if "api" not in data:
        raise ValueError("Missing required 'api' section in configuration")

    api_data = data["api"]
    if "base_url" not in api_data:
        raise ValueError("Missing required 'api.base_url' in configuration")

    api_config = ApiConfig(
        base_url=api_data["base_url"].rstrip("/"),
        timeout=api_data.get("timeout", 30),
    )

    detection_data = data.get("detection", {})
    detection_config = DetectionConfig(
        model=detection_data.get("model", "yolo11n.pt"),
        confidence_threshold=detection_data.get("confidence_threshold", 0.5),
        frame_time=detection_data.get("frame_time", 6.0),
        min_area_percent=detection_data.get("min_area_percent"),
        max_area_percent=detection_data.get("max_area_percent"),
    )

    database_data = data.get("database", {})
    database_config = DatabaseConfig(
        path=database_data.get("path", "bird_detections.db"),
    )

    outputs_data = data.get("outputs", {})
    outputs_config = OutputsConfig(
        directory=outputs_data.get("directory"),
    )

    return Config(
        api=api_config,
        detection=detection_config,
        database=database_config,
        outputs=outputs_config,
    )
