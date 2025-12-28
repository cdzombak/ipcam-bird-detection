FROM python:3.12-slim
ARG BIN_VERSION=<dev>

# Install ffmpeg for video frame extraction
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python source files
COPY api_client.py .
COPY config.py .
COPY database.py .
COPY detector.py .
COPY frame_extractor.py .
COPY pipeline.py .
COPY main.py .

# Inject version into the Python script
RUN sed -i "s/<dev>/${BIN_VERSION}/g" main.py

# Create a directory for config and data that can be mounted as a volume
RUN mkdir -p /data

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py", "--config", "/data/config.yaml"]

LABEL license="LGPL-3.0"
LABEL maintainer="Chris Dzombak <https://www.dzombak.com>"
LABEL org.opencontainers.image.authors="Chris Dzombak <https://www.dzombak.com>"
LABEL org.opencontainers.image.url="https://github.com/cdzombak/ipcam-bird-detection"
LABEL org.opencontainers.image.documentation="https://github.com/cdzombak/ipcam-bird-detection/blob/main/README.md"
LABEL org.opencontainers.image.source="https://github.com/cdzombak/ipcam-bird-detection.git"
LABEL org.opencontainers.image.version="${BIN_VERSION}"
LABEL org.opencontainers.image.licenses="LGPL-3.0"
LABEL org.opencontainers.image.title="ipcam-bird-detection"
LABEL org.opencontainers.image.description="Detect birds in IP camera videos using YOLO object detection"
