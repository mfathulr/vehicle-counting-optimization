# Vehicle Counting & Traffic Light Optimization

FastAPI + WebRTC application for vehicle counting with Faster R-CNN (ResNet18) and fuzzy logic to suggest green-light durations. Frontend is a single HTML/JS page served from `static/`.

## Highlights
- Faster R-CNN vehicle detection (cars, motorcycles)
- Realtime or upload modes (image/video, YouTube, webcam)
- Fuzzy logic optimizer for green time suggestions
- Web UI (WebRTC webcam / YouTube / upload image/video) with live metrics

## Quick Start
1) Clone & enter repo
```bash
git clone https://github.com/mfathulr/vehicle-counting-optimization.git
cd vehicle-counting-optimization
```
2) Create & activate venv (example)
```bash
python -m venv .venv
./.venv/Scripts/Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # macOS/Linux
```
3) Install deps
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
4) Get weights (Git LFS or manual)
- Git LFS: `model/best_model.pth` can be tracked via LFS. Run `git lfs install` before pulling.
- Manual: Download from [Google Drive](https://drive.google.com/drive/folders/1L419RCGY0zDCPojnsGmZsjhzgsRS1UyS?usp=sharing) and place in `model/`.

## Config
- Detection / mask / colors: [src/config.py](src/config.py)
- Fuzzy logic rules: [src/optimization/fuzzy_controller.py](src/optimization/fuzzy_controller.py)

## FastAPI (WebRTC) Run
For realtime browser camera via WebRTC, use the FastAPI app:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 and use the Realtime or Upload modes.

### Security (Production)
- Set allowed origins: `ALLOWED_ORIGINS` env var (comma-separated), e.g., `http://yourdomain.com`.
- Optional API key: set `API_KEY` env var. Then pass it via WebSocket header `X-API-Key` or query `?api_key=...`, and via HTTP header for `/api/youtube-stream`.
- Use HTTPS/WSS in production to enable camera access and secure WebSocket.

## Project Layout
- `main.py` — FastAPI backend (WebSocket + HTTP endpoints)
- `static/` — web UI (index.html + app.js)
- `src/models/` — Faster R-CNN definition
- `src/inference/` — image detection logic
- `src/optimization/` — fuzzy optimizer
- `src/utils/` — preprocessing helpers
- `assets/` — logo, mask; `model/` — weights; `input/` — samples

## Notes & Troubleshooting
- Python: Prefer Python 3.11 if wheels for Torch/OpenCV are missing on your platform; 3.13 may require source builds.
- Device: If CUDA is not available, the app automatically falls back to CPU.
- Windows: Temporary files for video processing use OS-portable temp paths.
- Realtime: Client uses backpressure to avoid overwhelming the server; FPS and round-trip (RT) latency are shown in the UI.

## License
Educational and research use. For other uses, please open an issue.
