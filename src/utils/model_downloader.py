"""Model downloader utility for automatic model weight retrieval."""

import os
import requests
from pathlib import Path
from tqdm import tqdm


def download_file_from_google_drive(file_id: str, destination: str) -> bool:
    """
    Download a file from Google Drive.

    Args:
        file_id: Google Drive file ID
        destination: Local path to save the file

    Returns:
        True if download successful, False otherwise
    """
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    try:
        print(f"Downloading model weights to {destination}...")

        response = session.get(URL, params={"id": file_id}, stream=True)

        # Handle large files with confirmation
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                params = {"id": file_id, "confirm": value}
                response = session.get(URL, params=params, stream=True)
                break

        # Get file size if available
        total_size = int(response.headers.get("content-length", 0))

        # Create parent directory if it doesn't exist
        Path(destination).parent.mkdir(parents=True, exist_ok=True)

        # Download with progress bar
        with (
            open(destination, "wb") as f,
            tqdm(
                desc="Downloading model",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✅ Model downloaded successfully to {destination}")
        return True

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        # Clean up partial download
        if os.path.exists(destination):
            os.remove(destination)
        return False


def ensure_model_exists(model_path: str, gdrive_file_id: str = None) -> bool:
    """
    Ensure model weights exist, download if missing.

    Args:
        model_path: Path to model weights file
        gdrive_file_id: Google Drive file ID (optional)

    Returns:
        True if model exists or was downloaded successfully
    """
    if os.path.exists(model_path):
        print(f"✅ Model found at {model_path}")
        return True

    if not gdrive_file_id:
        print(f"⚠️ Model not found at {model_path}")
        print("Please download manually from Google Drive or provide file ID")
        return False

    print(f"📥 Model not found, attempting automatic download...")
    return download_file_from_google_drive(gdrive_file_id, model_path)
