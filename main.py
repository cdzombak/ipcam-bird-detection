#!/usr/bin/env python3
"""Main entry point for ipcam-bird-detection."""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from api_client import ApiClient, MediaItem
from config import load_config
from database import Database
from pipeline import DetectionPipeline

# Version is injected at build time by the Dockerfile
VERSION = "<dev>"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_video(video_path: Path, pipeline: DetectionPipeline) -> int:
    """Test the detection pipeline on a local video file.

    Args:
        video_path: Path to the video file.
        pipeline: Detection pipeline instance.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 1

    logger.info(f"Testing video: {video_path}")
    result = pipeline.process(video_path)

    if not result.success:
        logger.error(f"Pipeline failed: {result.error}")
        return 1

    logger.info("=" * 50)
    logger.info("Detection Results")
    logger.info("=" * 50)
    logger.info(f"  Video duration: {result.video_duration:.2f}s")
    logger.info(f"  Frame extracted at: {result.frame_time:.2f}s")
    logger.info(f"  Bird detected: {result.has_bird}")

    if result.has_bird:
        logger.info(f"  Confidence: {result.confidence:.3f}")
        logger.info(f"  Bird area: {result.bird_area_percent:.2f}% of frame")

    return 0


def process_video_from_api(
    video: MediaItem,
    api_client: ApiClient,
    pipeline: DetectionPipeline,
    db: Database,
    output_dir: Path | None = None,
) -> bool:
    """Process a single video from the API.

    Args:
        video: MediaItem to process.
        api_client: API client for downloading.
        pipeline: Detection pipeline instance.
        db: Database instance.
        output_dir: Optional directory to save videos with birds.

    Returns:
        True if processing succeeded, False otherwise.
    """
    video_path = None
    keep_video = False

    try:
        # Download video
        logger.info(f"Downloading: {video.name}")
        video_path = api_client.download_video(video.proxy_url)

        # Run through pipeline
        logger.info(f"Processing: {video.name}")
        result = pipeline.process(video_path)

        if not result.success:
            logger.error(f"Pipeline failed for {video.name}: {result.error}")
            return False

        # Store result
        db.insert_result(
            filename=video.name,
            path=video.path,
            has_bird=result.has_bird,
            confidence=result.confidence,
            bird_area_percent=result.bird_area_percent,
            video_duration=result.video_duration,
            frame_time=result.frame_time,
        )

        if result.has_bird:
            logger.info(
                f"Bird detected in {video.name}: "
                f"confidence={result.confidence:.2f}, "
                f"area={result.bird_area_percent:.1f}%"
            )
            # Save video to output directory
            if output_dir:
                dest_path = output_dir / video.download_filename
                shutil.copy2(video_path, dest_path)
                logger.info(f"Saved to: {dest_path}")
                keep_video = False  # We copied it, so we can delete the temp file
        else:
            logger.info(f"No bird in {video.name}")

        return True

    except Exception as e:
        logger.error(f"Failed to process {video.name}: {e}")
        return False

    finally:
        # Clean up downloaded video
        if video_path and video_path.exists() and not keep_video:
            video_path.unlink(missing_ok=True)


def run_batch(config, pipeline: DetectionPipeline) -> int:
    """Run batch processing of videos from the API.

    Args:
        config: Application configuration.
        pipeline: Detection pipeline instance.

    Returns:
        Exit code.
    """
    logger.info(f"Using API: {config.api.base_url}")
    logger.info(f"Using database: {config.database.path}")
    logger.info(f"Using model: {config.detection.model}")

    # Set up output directory if configured
    output_dir = None
    if config.outputs.directory:
        output_dir = Path(config.outputs.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving bird videos to: {output_dir}")

    with Database(config.database.path) as db:
        with ApiClient(config.api.base_url, timeout=config.api.timeout) as api:
            # Fetch video list
            logger.info("Fetching video list from API...")
            try:
                videos = api.get_videos()
            except Exception as e:
                logger.error(f"Failed to fetch videos: {e}")
                return 1

            logger.info(f"Found {len(videos)} videos")

            # Filter to unprocessed videos
            unprocessed = [v for v in videos if not db.is_processed(v.name)]
            logger.info(f"Videos to process: {len(unprocessed)}")

            if not unprocessed:
                logger.info("No new videos to process")
                return 0

            # Process each video
            success_count = 0
            fail_count = 0

            for i, video in enumerate(unprocessed, 1):
                logger.info(f"Processing [{i}/{len(unprocessed)}]: {video.name}")

                if process_video_from_api(video, api, pipeline, db, output_dir):
                    success_count += 1
                else:
                    fail_count += 1

            # Print summary
            stats = db.get_stats()
            logger.info("=" * 50)
            logger.info("Processing complete")
            logger.info(
                f"  Processed this run: {success_count} success, {fail_count} failed"
            )
            logger.info(f"  Total in database: {stats['total']}")
            logger.info(f"  Videos with birds: {stats['birds_found']}")
            logger.info(f"  Videos without birds: {stats['no_birds']}")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect birds in IP camera videos",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("./config.yaml"),
        help="Path to YAML configuration file (default: ./config.yaml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--test-video",
        type=Path,
        metavar="VIDEO",
        help="Test detection on a local video file (logs results, does not store in database)",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        return 1

    # Create pipeline
    pipeline = DetectionPipeline(
        model_path=config.detection.model,
        confidence_threshold=config.detection.confidence_threshold,
        frame_times=config.detection.frame_times,
        min_area_percent=config.detection.min_area_percent,
        max_area_percent=config.detection.max_area_percent,
    )

    # Test mode: process a single local video
    if args.test_video:
        return test_video(args.test_video, pipeline)

    # Batch mode: process videos from API
    return run_batch(config, pipeline)


if __name__ == "__main__":
    sys.exit(main())
