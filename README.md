## API Endpoints

| Endpoint                | Method      | Description                                                      |
|-------------------------|-------------|------------------------------------------------------------------|
| `/ws`                   | WebSocket   | Real-time vehicle detection (webcam/YouTube preview & detection)  |
| `/ws/process-video`     | WebSocket   | Process uploaded video/image                                      |
| `/api/video/{filename}` | GET         | Serve processed video                                            |
| `/api/youtube-proxy`    | GET         | Proxy YouTube video stream for preview/detection (CORS bypass)    |
# Vehicle Counting & Traffic Light Optimization

FastAPI + WebSocket application for vehicle counting with Faster R-CNN (ResNet18) and fuzzy logic to suggest green-light durations. Frontend is a single HTML/JS page served from `static/`.

## Features

### 1. Realtime Mode (Webcam & YouTube)
- Live preview from webcam (MediaStream API) or YouTube video (via backend proxy)
- Start/stop preview before detection for both sources
- Per-frame vehicle detection and traffic light duration calculation
- Stream cleanup on mode/source switch (webcam stops when switching to YouTube, and vice versa)

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI server

```bash
uvicorn main:app --reload
```

### 3. Open the Web UI

- Go to [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)

### 4. Modes

- **Realtime Mode**: Select webcam or YouTube, click "Preview" to start live stream, then "Start Detection". Stream cleanup is automatic when switching sources.
- **Upload Mode**: Upload video/image (max 500MB), preview, then start detection.

### 5. YouTube CORS Proxy

- Enter a YouTube URL in Realtime Mode and click "Preview". The backend `/api/youtube-proxy` endpoint will stream the video to the frontend, bypassing CORS restrictions. No browser CORS errors.

### 6. File Size Limit

- Uploads are limited to 500MB. Larger files are rejected client-side before upload.

### 7. Model

- The model is loaded from `model/best_model.pth` (PyTorch Faster R-CNN).

### 8. Configuration

- Adjust thresholds, FPS, frame skip, and ROI in the Web UI as needed.
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 and choose:
- **Realtime Mode**: Start webcam stream with live detection
- **Upload Mode**: Select video/image file, configure detection settings, and process with live preview

### Upload Mode Features
- **Supported formats**: MP4, AVI, MOV, WebM (video); JPG, PNG (image)
- **File size limit**: 500MB maximum
- **Live preview**: Stream processing frames to canvas in real-time
- **Text overlay**: Detection results (car/bike counts, traffic light duration) written to output
- **Result preview**: Play processed video before download
- **Controls**: 
  - Confidence threshold adjustment (0.0-1.0)
  - Frame skip for video processing (1-30)
  - ROI drawing for detection area restriction
  - Dynamic status with animated processing indicator
  - Cancel button with automatic page reload on completion

### Security (Production)
- Set allowed origins: `ALLOWED_ORIGINS` env var (comma-separated), e.g., `http://yourdomain.com`.
- Optional API key: set `API_KEY` env var. Then pass it via WebSocket header `X-API-Key` or query `?api_key=...`, and via HTTP header for `/api/youtube-stream`.
- Use HTTPS/WSS in production to enable camera access and secure WebSocket.

## Project Layout
- `main.py` — FastAPI backend (WebSocket + HTTP endpoints for video serving)
- `static/` — web UI (index.html + app.js with fuzzy logic calculator)
  - **Realtime tab**: Live webcam processing with canvas overlay
  - **Upload tab**: File-based processing with stream preview and text overlays
- `src/models/` — Faster R-CNN definition (ResNet18 backbone)
- `src/inference/` — image/video detection logic with frame processing
- `src/optimization/` — Fuzzy logic traffic light optimizer (scikit-fuzzy based)
- `src/utils/` — preprocessing helpers, model downloader, stats export
- `assets/` — logo, mask; `model/` — weights; `input/` — sample files

## Docker

Build the image (CPU):

```powershell
docker build -t vehicle-counting-opt .
```

Run with volumes and env (Windows PowerShell):

```powershell
# Optional: set Google Drive ID for auto-download
$env:MODEL_GDRIVE_ID = "<drive-file-id>"

docker run --rm -p 8000:8000 `
	-e MODEL_GDRIVE_ID=$env:MODEL_GDRIVE_ID `
	-e ALLOWED_ORIGINS=http://localhost:8000 `
	-v ${PWD}/model:/app/model `
	-v ${PWD}/input:/app/input `
	vehicle-counting-opt
```

Using Docker Compose:

```powershell
# MODEL_GDRIVE_ID and API_KEY are read from environment if present
docker compose up --build
```

Notes:
- The image is CPU-only. CUDA builds are not configured.
- Model weights are volume-mounted from `./model` to `/app/model`.
- Healthcheck hits `/health`; container restarts unless stopped in Compose.

## CI: GitHub Actions (Docker)

This repo includes a workflow to build and publish the Docker image to GitHub Container Registry (GHCR) on pushes to `main` and tags (`v*`, `release-*`). See [.github/workflows/docker.yml](.github/workflows/docker.yml).

Default behavior:
- Pull Requests: build only (no push)
- `main` branch: push `:latest`, `:sha`, and branch tags
- Tags: push tag-based image (e.g., `ghcr.io/<owner>/<repo>:v1.2.3`)

No extra secrets required; it uses the built-in `GITHUB_TOKEN` to push to `ghcr.io`. To pull:

```bash
docker pull ghcr.io/<owner>/<repo>:latest
```

Replace `<owner>/<repo>` with your repository path.

## Notes & Troubleshooting
- **Python**: Prefer Python 3.11 if wheels for Torch/OpenCV are missing on your platform; 3.13 may require source builds.
- **Device**: If CUDA is not available, the app automatically falls back to CPU.
- **Windows**: Temporary files for video processing use OS-portable temp paths. FFmpeg is required for video re-encoding.
- **FFmpeg**: Install via `winget install ffmpeg` (Windows) or `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)
- **WebSocket**: Maximum message size is 2MB; uploads are chunked at 1MB per chunk. Files larger than 500MB are rejected at upload.
- **Video playback**: Output videos are re-encoded to H.264 with FFmpeg for browser compatibility.
- **Realtime**: Client uses backpressure to avoid overwhelming the server; FPS and round-trip (RT) latency are shown in the UI.
- **Stream preview**: JPEG frames are streamed at quality 92 to balance quality and bandwidth.

## License
Educational and research use. For other uses, please open an issue.
