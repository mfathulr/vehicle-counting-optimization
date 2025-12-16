"""Configuration file for the vehicle counting system."""

from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
MODEL_DIR = ROOT_DIR / "model"
INPUT_DIR = ROOT_DIR / "input"

# Model configuration
MODEL_PATH = MODEL_DIR / "best_model.pth"
NUM_CLASSES = 3
CLASS_NAMES = ["__background__", "mobil", "motor"]

# Google Drive model download (optional - set via env var or provide direct ID)
# Get file ID from: https://drive.google.com/drive/folders/1L419RCGY0zDCPojnsGmZsjhzgsRS1UyS
MODEL_GDRIVE_ID = None  # Set to Google Drive file ID for auto-download

# Detection colors (BGR format for OpenCV)
CLASS_COLORS = {
    "__background__": (255, 0, 0),
    "mobil": (0, 255, 0),
    "motor": (0, 0, 255),
}

# Image preprocessing
IMAGE_SIZE = 512
MASK_PATH = ASSETS_DIR / "mask.png"

# Detection parameters
# Confidence threshold for filtering detections
DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Increased from 0.25 to reduce false positives
DEFAULT_IOU_THRESHOLD = 0.5  # IoU threshold for NMS (Non-Maximum Suppression)
MIN_BOX_SIZE = 2  # Minimum box width/height in pixels

# Anchor generator configuration
ANCHOR_SIZES = ((32, 64, 128, 256, 512),)
ANCHOR_ASPECT_RATIOS = ((0.5, 1.0, 2.0),)

# ROI pooler configuration
ROI_OUTPUT_SIZE = 7
ROI_SAMPLING_RATIO = 2

# Fuzzy logic configuration
FUZZY_MOTOR_RANGE = (0, 31, 1)
FUZZY_MOBIL_RANGE = (0, 11, 1)
FUZZY_DURATION_RANGE = (0, 61, 1)

# Fuzzy membership function parameters
FUZZY_MOTOR_LOW = [0, 0, 15]
FUZZY_MOTOR_MEDIUM = [0, 15, 30]
FUZZY_MOTOR_HIGH = [15, 30, 30]

FUZZY_MOBIL_LOW = [0, 0, 5]
FUZZY_MOBIL_MEDIUM = [0, 5, 10]
FUZZY_MOBIL_HIGH = [5, 10, 10]

FUZZY_DURATION_SHORT = [0, 0, 20]
FUZZY_DURATION_MEDIUM = [10, 20, 40, 50]
FUZZY_DURATION_LONG = [40, 60, 60]

# Streamlit configuration
APP_TITLE = "🚗 Vehicle Counting System"
APP_ICON = ASSETS_DIR / "logo.png"
PAGE_TITLE = "Traffic Light Optimization"
LAYOUT = "wide"

# Video processing
VIDEO_CODEC = "mp4v"
TEMP_VIDEO_PATH = ROOT_DIR / "temp.mp4"
OUTPUT_VIDEO_PATH = ROOT_DIR / "output.mp4"
