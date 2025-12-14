## Deployment (Streamlit Cloud / Python 3.13)

- Entry point: app/streamlit_app.py
- Python: 3.13 is supported by current requirements; if the host lacks binary wheels for Torch/OpenCV, use Python 3.11.
- Requirements: see requirements.txt (torch>=2.5.0, torchvision>=0.20.0, opencv-python-headless>=4.8.0, streamlit>=1.39, yt-dlp, streamlink, numpy>=2.0, pandas>=2.2, psutil).
- Model weights: track large files via Git LFS.
- Realtime notes:
	- In cloud, app prefers OpenCV `CAP_ANY` and refreshes HLS URLs more often.
	- If direct HLS fails, a Streamlink fallback attempts to acquire a playable URL.
	- Enable "🔍 Enable realtime debug logs" in the UI to see backend details and errors.

### Quick Start

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Troubleshooting (Production)

- CUDA not available: the app automatically falls back to CPU.
- Stream fails to open: error banner persists; use "Retry Same Source" or try a different URL.
- If deployment fails on Python 3.13 due to missing wheels, set runtime to Python 3.11.

# Vehicle Counting & Traffic Light Optimization

Streamlit app for vehicle counting with Faster R-CNN (ResNet18) and fuzzy logic to suggest green-light durations.

## Highlights
- Faster R-CNN vehicle detection (cars, motorcycles)
- Realtime or upload modes (image/video, YouTube, webcam)
- Fuzzy logic optimizer for green time suggestions
- Streamlit UI with progress, metrics, and live preview

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
- Git LFS: `model/best_model.pth` is tracked via LFS. Run `git lfs install` before pulling.
- Manual: Download from [Google Drive](https://drive.google.com/drive/folders/1L419RCGY0zDCPojnsGmZsjhzgsRS1UyS?usp=sharing) and place in `model/`.

## Run
```bash
streamlit run app/streamlit_app.py
```
Open http://localhost:8501 and choose a mode:
- Upload: image/video file, then Detect/Process.
- Realtime: YouTube URL or webcam, adjustable frame skip and resolution.

## Config
- Detection / mask / colors: [src/config.py](src/config.py)
- Fuzzy logic rules: [src/optimization/fuzzy_controller.py](src/optimization/fuzzy_controller.py)

## Minimal Layout
- app/streamlit_app.py — UI entrypoint
- src/models/ — Faster R-CNN definition
- src/inference/ — image/video detectors
- src/optimization/ — fuzzy optimizer
- assets/ — logo, mask; model/ — weights; input/ — samples

## Training (optional)
See [training/README.md](training/README.md) for dataset prep, training, and evaluation scripts.

## License
Educational and research use. For other uses, please open an issue.
