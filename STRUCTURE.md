# Project Structure

This document describes the standardized project structure for the Vehicle Counting & Traffic Optimization system.

## Directory Layout

```
vehicle-counting-optimization/
├── app/                          # Application entry point
│   └── streamlit_app.py         # Main Streamlit web application
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                # Configuration and constants
│   │
│   ├── models/                   # Deep learning models
│   │   ├── __init__.py
│   │   └── faster_rcnn.py       # Faster R-CNN model definition
│   │
│   ├── inference/                # Detection inference
│   │   ├── __init__.py
│   │   ├── image_detector.py    # Image detection class
│   │   └── video_detector.py    # Video detection class
│   │
│   ├── optimization/             # Traffic optimization
│   │   ├── __init__.py
│   │   └── fuzzy_controller.py  # Fuzzy logic traffic light optimizer
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── preprocessing.py     # Image preprocessing utilities
│
├── training/                     # Model training & development
│   ├── __init__.py
│   ├── config.py                # Training configuration
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Model evaluation
│   ├── prepare_data.py          # Dataset preparation
│   ├── utils.py                 # Training utilities
│   └── README.md                # Training documentation
│
├── notebooks/                    # Jupyter notebooks
│   └── README.md                # Notebooks guide
│
├── data/                         # Training & validation datasets
│   ├── train/                   # Training images
│   ├── val/                     # Validation images
│   ├── annotations/             # COCO-format annotations
│   │   ├── train.json
│   │   └── val.json
│   └── README.md                # Dataset documentation
│
├── assets/                       # Static assets
│   ├── logo.png                 # Application logo
│   └── mask.png                 # Detection region mask
│
├── model/                        # Trained model weights
│   ├── best_model.pth           # Best performing model
│   ├── checkpoints/             # Training checkpoints
│   └── evaluation_results.json  # Evaluation metrics
│
├── input/                        # Input files directory
│   ├── images/                  # Sample images
│   └── videos/                  # Sample videos
│
├── logs/                         # Training logs (auto-generated)
│   └── tensorboard/             # TensorBoard logs
│
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── STRUCTURE.md                 # This file

```

## File Naming Conventions

### Python Files
- **Module files**: `lowercase_with_underscores.py`
- **Class definitions**: PascalCase (e.g., `ImageDetector`, `FasterRCNN`)
- **Functions**: lowercase_with_underscores (e.g., `detect_vehicles`, `optimize_traffic`)
- **Constants**: UPPERCASE_WITH_UNDERSCORES (e.g., `MODEL_PATH`, `NUM_CLASSES`)

### Configuration Files
- **Python config**: `config.py`
- **Dependencies**: `requirements.txt`
- **Git ignore**: `.gitignore`
- **Documentation**: `README.md`, `STRUCTURE.md`

### Application Files
- **Main app**: `streamlit_app.py` (in app/ directory)
- **Entry scripts**: descriptive names (e.g., `train_model.py`, `evaluate.py`)

## Module Organization

### src/models/
Contains neural network model definitions and architectures.

### src/inference/
Contains detection and inference logic for processing images and videos.

### src/optimization/
Contains traffic optimization algorithms (fuzzy logic, etc.).

### src/utils/
Contains utility functions for preprocessing, visualization, and helper operations.

## Best Practices

1. **Keep it modular**: Each module should have a single responsibility
2. **Use __init__.py**: Export public APIs from each package
3. **Follow PEP 8**: Standard Python style guide
4. **Document everything**: Use docstrings for all functions and classes
5. **Type hints**: Use type annotations for better code clarity
6. **No hardcoded paths**: Use config.py for all paths and constants

## Removed Legacy Files

The following legacy files have been removed for cleaner project structure:
- `detect_images.py` (replaced by src/inference/)
- `fuzzy_logic.py` (replaced by src/optimization/)
- `main.py` (replaced by app/streamlit_app.py)
- `model.py` (replaced by src/models/)
- `__pycache__/` (compiled Python files)

All functionality has been refactored into the modular src/ structure.
