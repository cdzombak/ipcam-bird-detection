"""API client for ipcam-browser."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class MediaItem:
    """A media item from the API."""

    name: str
    path: str
    url: str
    proxy_url: str
    download_filename: str
    date: str
    media_type: str
    trigger: str
    timestamp: str
    size: str
    modified: str
    thumbnail_url: str | None = None


class ApiClient:
    """Client for the ipcam-browser API."""

    def __init__(self, base_url: str, timeout: int = 30):
        """Initialize the API client.

        Args:
            base_url: Base URL of the ipcam-browser API.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def get_videos(self) -> list[MediaItem]:
        """Fetch all video items from the API.

        Returns:
            List of MediaItem objects for videos only.

        Raises:
            requests.RequestException: If the API request fails.
        """
        response = self._session.get(
            f"{self.base_url}/api/media",
            timeout=self.timeout,
        )
        response.raise_for_status()

        items = []
        for item in response.json():
            if item.get("type") != "video":
                continue

            items.append(
                MediaItem(
                    name=item["name"],
                    path=item["path"],
                    url=item["url"],
                    proxy_url=item.get("proxyUrl", ""),
                    download_filename=item["downloadFilename"],
                    date=item["date"],
                    media_type=item["type"],
                    trigger=item["trigger"],
                    timestamp=item["timestamp"],
                    size=item["size"],
                    modified=item["modified"],
                    thumbnail_url=item.get("thumbnailUrl"),
                )
            )

        return items

    def download_video(self, proxy_url: str) -> Path:
        """Download a video via its proxy URL.

        Args:
            proxy_url: The proxyUrl from MediaItem (e.g., "/api/video/...").

        Returns:
            Path to the downloaded temporary file.

        Raises:
            requests.RequestException: If the download fails.
            ValueError: If proxy_url is empty.
        """
        if not proxy_url:
            raise ValueError("Empty proxy URL")

        url = f"{self.base_url}{proxy_url}"
        response = self._session.get(url, timeout=self.timeout, stream=True)
        response.raise_for_status()

        # Create temp file with .mp4 extension
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        try:
            with open(fd, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception:
            Path(temp_path).unlink(missing_ok=True)
            raise

        return Path(temp_path)

    def close(self) -> None:
        """Close the session."""
        self._session.close()

    def __enter__(self) -> "ApiClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
