# ipcam-bird-detection

Detects birds in IP camera videos using YOLO object detection. Fetches videos from an [ipcam-browser](https://github.com/cdzombak/ipcam-browser) API, extracts a frame, runs detection, and stores results in SQLite.

## Requirements

- Python 3.11+
- ffmpeg
- [ipcam-browser](https://github.com/cdzombak/ipcam-browser) API (for batch mode)

## Installation

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

## Usage

**Batch mode** — process all videos from the API:

```bash
python main.py
python main.py -c /path/to/config.yaml
```

**Test mode** — run detection on a local video (logs results, no database):

```bash
python main.py --test-video /path/to/video.mp4
```

**Options:**

- `-c, --config` — config file path (default: `./config.yaml`)
- `-v, --verbose` — enable debug logging
- `--test-video` — test detection on a local video file
- `--version` — show version and exit

## Docker

Docker images are available for `linux/amd64` and `linux/arm64`:

```bash
docker pull cdzombak/ipcam-bird-detection:latest
```

Run with a config file and data directory mounted:

```bash
docker run --rm \
  -v /path/to/config.yaml:/data/config.yaml:ro \
  -v /path/to/data:/data \
  cdzombak/ipcam-bird-detection:latest
```

The container expects:

- Config file at `/data/config.yaml`
- Database will be written to `/data/` (configure `database.path` accordingly)
- Output videos saved to `/data/` (configure `outputs.directory` accordingly)

Example config for Docker:

```yaml
database:
  path: "/data/bird_detections.db"
outputs:
  directory: "/data/bird_videos"
```

## Configuration

See `config.example.yaml` for all options. Key settings:

```yaml
api:
  base_url: "http://localhost:8080" # ipcam-browser API URL

detection:
  model: "yolo11n.pt" # YOLO model (downloaded automatically)
  confidence_threshold: 0.5 # Minimum detection confidence
  frame_times: [6.0, 8.0] # Check these times for birds (or 50% if shorter)
  min_area_percent: 0.5 # Optional: ignore birds smaller than this % of frame
  max_area_percent: 50.0 # Optional: ignore birds larger than this % of frame

database:
  path: "bird_detections.db"

outputs:
  directory: "./bird_videos" # Optional: save videos with birds here
```

## Database Schema

Results are stored in SQLite with the following fields:

| Field               | Description                         |
| ------------------- | ----------------------------------- |
| `filename`          | Original video filename             |
| `has_bird`          | 1 if bird detected, 0 otherwise     |
| `confidence`        | Detection confidence (largest bird) |
| `bird_area_percent` | % of frame area (largest bird)      |
| `video_duration`    | Video length in seconds             |
| `frame_time`        | Timestamp of extracted frame        |
| `processed_at`      | Processing timestamp                |
