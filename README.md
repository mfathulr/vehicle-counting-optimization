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
