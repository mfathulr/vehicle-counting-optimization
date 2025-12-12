# Vehicle Counting & Traffic Light Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-FF4B4B.svg)](https://streamlit.io/)

Professional vehicle counting system using Faster R-CNN with ResNet18 backbone and fuzzy logic for traffic light optimization.

## 📋 Features

- **Object Detection**: Modified Faster R-CNN with ResNet18 for vehicle detection (cars & motorcycles)
- **Fuzzy Logic**: Dynamic traffic light duration optimization based on vehicle counts
- **Real-time Processing**: Support for both image and video inference
- **Interactive UI**: Streamlit-based web interface with live visualization
- **Professional Architecture**: Modular, maintainable code structure

## 🏗️ Project Structure

```
vehicle-counting-optimization/
├── src/                          # Source code
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   └── faster_rcnn.py       # Faster R-CNN implementation
│   ├── inference/                # Detection logic
│   │   ├── __init__.py
│   │   ├── image_detector.py    # Image inference
│   │   └── video_detector.py    # Video inference
│   ├── optimization/             # Traffic optimization
│   │   ├── __init__.py
│   │   └── fuzzy_controller.py  # Fuzzy logic controller
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   └── preprocessing.py     # Image preprocessing & visualization
│   ├── config.py                 # Configuration settings
│   └── __init__.py
├── app/                          # Application
│   └── streamlit_app.py         # Streamlit web app (professional UI)
├── training/                     # Model training & development
│   ├── __init__.py
│   ├── config.py                # Training configuration
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Model evaluation
│   ├── prepare_data.py          # Dataset preparation
│   ├── utils.py                 # Training utilities
│   └── README.md                # Training documentation
├── notebooks/                    # Jupyter notebooks for experimentation
│   └── README.md
├── data/                         # Training & validation datasets
│   ├── train/                   # Training images
│   ├── val/                     # Validation images
│   ├── annotations/             # COCO-format annotations
│   └── README.md
├── assets/                       # Static assets
│   ├── logo.png
│   ├── mask.png
│   └── black.jpg
├── model/                        # Trained models
│   └── best_model.pth
├── input/                        # Sample inputs
├── logs/                         # Training logs (auto-generated)
│   └── tensorboard/             # TensorBoard logs
├── requirements.txt              # Python dependencies
├── .gitignore
├── README.md
└── STRUCTURE.md                 # Detailed structure guide

# Legacy files (kept for reference)
├── main.py                       # Old monolithic app
├── model.py                      # Old model code
├── fuzzy_logic.py               # Old fuzzy logic
└── detect_images.py             # Old detection code
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mfathul21/vehicle-counting-optim.git
   cd vehicle-counting-optim
   ```

2. **Create virtual environment:**
   ```bash
   # Using venv (Python 3.10+)
   python -m venv .venv
   
   # On Windows
   .\.venv\Scripts\Activate.ps1
   
   # On Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download the trained model:**
   - Download from [Google Drive](https://drive.google.com/drive/folders/1L419RCGY0zDCPojnsGmZsjhzgsRS1UyS?usp=sharing)
   - Place `best_model.pth` in the `model/` directory

### Running the Application

```bash
# Run the Streamlit app
streamlit run app/streamlit_app.py

# Or use the legacy version
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Usage

1. **Select Device**: Choose CPU or CUDA (if available)
2. **Set Threshold**: Adjust confidence threshold (default: 0.25)
3. **Choose Mode**: Select Image or Video inference
4. **Upload File**: Upload your image (.jpg, .jpeg, .png) or video (.mp4)
5. **Detect**: Click the "Detect" button to run inference
6. **View Results**: See detection counts and optimized traffic light duration

### Input Recommendations

- Use CCTV images from [Jogja CCTV](https://cctv.jogjakota.go.id), particularly from Pingit 1 Street
- Sample inputs are provided in the `input/` folder
- Note: The mask is pre-configured for specific camera angles; may need adjustment for other sources

## 🔧 Configuration

Edit `src/config.py` to customize:

- Model paths and parameters
- Detection thresholds
- Image processing settings
- Fuzzy logic membership functions
- UI appearance

## 🎓 Model Training & Fine-Tuning

The project includes a complete training infrastructure for developing and fine-tuning models. See [training/README.md](training/README.md) for detailed documentation.

### Quick Training Guide

1. **Prepare your dataset:**
   ```bash
   python training/prepare_data.py path/to/images path/to/annotations.json
   ```

2. **Configure training parameters:**
   Edit `training/config.py` to adjust hyperparameters

3. **Start training:**
   ```bash
   python training/train.py
   ```

4. **Monitor training:**
   ```bash
   tensorboard --logdir=logs/tensorboard
   ```

5. **Evaluate model:**
   ```bash
   python training/evaluate.py
   ```

### Training Features

- **TensorBoard Integration**: Real-time training visualization
- **Checkpoint Management**: Automatic best model saving
- **Early Stopping**: Prevent overfitting
- **Evaluation Metrics**: Precision, Recall, F1-score per class
- **COCO Format**: Standard annotation format
- **Data Augmentation**: Configurable transforms
- **Jupyter Notebooks**: For experimentation and analysis

See the [training directory](training/) for complete documentation and examples.

## 📊 Model Architecture

- **Backbone**: ResNet18 with extra BasicBlock layers
- **Detection Head**: Faster R-CNN with RPN
- **Anchors**: 5 sizes × 3 aspect ratios
- **Classes**: Background, Cars (mobil), Motorcycles (motor)

<figure>
    <img src="assets/arsitektur_model.jpg" alt="Faster R-CNN Architecture" width="500">
    <figcaption>Faster R-CNN with ResNet-18 backbone</figcaption>
</figure>

## 📈 Fuzzy Logic

The system uses Mamdani fuzzy inference with:

- **Inputs**: Car count (0-10), Motorcycle count (0-30)
- **Output**: Traffic light duration (0-60 seconds)
- **Membership Functions**: Low, Medium, High for each variable
- **Rules**: 15 fuzzy rules for optimal traffic control

## 🗂️ Dataset

Dataset collected from Jogja CCTV intersections:
- **Source**: [Jogja City CCTV](https://cctv.jogjakota.go.id)
- **Annotation**: Roboflow
- **Classes**: 2 (motorcycles, cars)
- **Augmentation**: Applied during training

## 👥 Contributors

- Muhammad Fathul Radhiansyah ([@mfathul21](https://github.com/mfathul21))
- Novia Putri Bahirah ([@noviaptr](https://github.com/noviaptr))
- Duana Puspitaningrum ([@DuanaPuspitaningrum](https://github.com/DuanaPuspitaningrum))
- Al Ahmad Syah Huud S. ([@alahmadss](https://github.com/alahmadss))

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- PyTorch and torchvision teams for the deep learning framework
- Streamlit for the web framework
- scikit-fuzzy for fuzzy logic implementation
- Jogja City for CCTV access

## 📧 Contact

For questions or collaboration, please open an issue or contact the contributors.

---

**Note**: This is the refactored professional version. Legacy files (`main.py`, `model.py`, etc.) are kept for backward compatibility but will be deprecated in future versions.
