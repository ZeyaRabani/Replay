# Replay Highlights — one-container build (multi-arch: amd64 + arm64)
# Stage 1: build the React frontend
FROM node:20-slim AS frontend
WORKDIR /build
COPY highlights/app/frontend/package.json highlights/app/frontend/package-lock.json ./
RUN npm ci
COPY highlights/app/frontend/ ./
RUN npm run build

# Stage 2: runtime
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg nodejs ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY highlights/app/requirements.txt /tmp/app-req.txt
COPY highlights/pipeline/requirements.txt /tmp/pipeline-req.txt
RUN pip install --no-cache-dir -r /tmp/app-req.txt -r /tmp/pipeline-req.txt
# Multi-angle stage: YOLO on CPU. ultralytics is installed with --no-deps so it
# doesn't drag in the GUI opencv-python build (we use opencv-python-headless);
# its remaining deps are listed in the requirements file.
COPY highlights/multiangle/requirements.txt /tmp/ma-req.txt
ENV YOLO_CONFIG_DIR=/tmp/yolo
RUN grep -v '^ultralytics==' /tmp/ma-req.txt > /tmp/ma-base.txt \
    && pip install --no-cache-dir -r /tmp/ma-base.txt \
    && pip install --no-cache-dir --no-deps "$(grep '^ultralytics==' /tmp/ma-req.txt)" \
    && mkdir -p /app/highlights/multiangle/models \
    && cd /app/highlights/multiangle/models \
    && python -c "from ultralytics.utils.downloads import attempt_download_asset; attempt_download_asset('yolov8n.pt')" \
    && test -s /app/highlights/multiangle/models/yolov8n.pt
COPY highlights/ /app/highlights/
COPY --from=frontend /build/dist /app/highlights/app/frontend/dist
ENV HL_WORKDIR=/data PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "highlights.app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
