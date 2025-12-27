"""SQLite database operations for ipcam-bird-detection."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class VideoRecord:
    """Record of a processed video."""

    filename: str
    path: str
    processed_at: str
    has_bird: bool
    confidence: float | None
    bird_area_percent: float | None
    video_duration: float | None
    frame_time: float | None


class Database:
    """SQLite database for storing video processing results."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE NOT NULL,
        path TEXT NOT NULL,
        processed_at TEXT NOT NULL,
        has_bird INTEGER NOT NULL,
        confidence REAL,
        bird_area_percent REAL,
        video_duration REAL,
        frame_time REAL
    );
    CREATE INDEX IF NOT EXISTS idx_videos_filename ON videos(filename);
    CREATE INDEX IF NOT EXISTS idx_videos_has_bird ON videos(has_bird);
    """

    def __init__(self, db_path: str | Path):
        """Initialize database connection.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Open the database connection and ensure schema exists."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Database":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def is_processed(self, filename: str) -> bool:
        """Check if a video has already been processed.

        Args:
            filename: The video filename to check.

        Returns:
            True if the video is already in the database.
        """
        if not self._conn:
            raise RuntimeError("Database not connected")

        cursor = self._conn.execute(
            "SELECT 1 FROM videos WHERE filename = ?",
            (filename,),
        )
        return cursor.fetchone() is not None

    def insert_result(
        self,
        filename: str,
        path: str,
        has_bird: bool,
        confidence: float | None = None,
        bird_area_percent: float | None = None,
        video_duration: float | None = None,
        frame_time: float | None = None,
    ) -> None:
        """Insert a video processing result.

        Args:
            filename: Original video filename.
            path: Full path from API.
            has_bird: Whether a bird was detected.
            confidence: Detection confidence (if bird found).
            bird_area_percent: Percentage of frame area (if bird found).
            video_duration: Video length in seconds.
            frame_time: Timestamp of extracted frame.
        """
        if not self._conn:
            raise RuntimeError("Database not connected")

        self._conn.execute(
            """
            INSERT INTO videos (
                filename, path, processed_at, has_bird,
                confidence, bird_area_percent, video_duration, frame_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                path,
                datetime.now().isoformat(),
                1 if has_bird else 0,
                confidence,
                bird_area_percent,
                video_duration,
                frame_time,
            ),
        )
        self._conn.commit()

    def get_stats(self) -> dict:
        """Get summary statistics.

        Returns:
            Dictionary with total, birds_found, and no_birds counts.
        """
        if not self._conn:
            raise RuntimeError("Database not connected")

        cursor = self._conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(has_bird) as birds_found
            FROM videos
            """
        )
        row = cursor.fetchone()
        total = row["total"] or 0
        birds_found = row["birds_found"] or 0

        return {
            "total": total,
            "birds_found": birds_found,
            "no_birds": total - birds_found,
        }
