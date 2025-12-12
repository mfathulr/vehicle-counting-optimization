"""Image detection module for vehicle counting."""

import torch
import numpy as np
import streamlit as st
from typing import Tuple, List

from ..config import CLASS_NAMES, IMAGE_SIZE, DEFAULT_IOU_THRESHOLD
from ..utils.preprocessing import (
    preprocess_image,
    draw_detection_box,
    filter_duplicate_boxes,
)


class ImageDetector:
    """Handles vehicle detection on static images."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Initialize the image detector.

        Args:
            model: Trained Faster R-CNN model.
            device: Device to run inference on ('cpu' or 'cuda').
        """
        self.model = model
        self.device = device
        self.model.to(device).eval()

    def detect(
        self,
        image: np.ndarray,
        threshold: float = DEFAULT_IOU_THRESHOLD,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, List, List, List]:
        """
        Perform vehicle detection on an image.

        Args:
            image: Input image in BGR format.
            threshold: Confidence threshold for detections.
            show_progress: Whether to show progress bar in Streamlit.

        Returns:
            Tuple of (annotated image, boxes, scores, class names).
        """
        # Setup progress tracking
        if show_progress:
            progress_bar = st.sidebar.progress(0)
            progress_text = st.sidebar.empty()
        else:
            progress_bar = None
            progress_text = None

        # Store original image
        orig_image = image.copy()

        # Preprocess image
        processed_image, orig_height, orig_width = preprocess_image(
            image, target_size=IMAGE_SIZE, apply_mask=True
        )

        # Convert to tensor
        image_tensor = torch.tensor(processed_image, dtype=torch.float)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.device))

        # Post-process outputs
        boxes = outputs[0]["boxes"].data.cpu().numpy()
        scores = outputs[0]["scores"].data.cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()

        # Apply threshold filtering
        keep = scores >= threshold
        boxes = boxes[keep].astype(np.int32)
        scores = scores[keep]
        labels = labels[keep]
        pred_classes = [CLASS_NAMES[i] for i in labels]

        # Filter duplicate boxes
        filtered_boxes, filtered_classes, filtered_scores = filter_duplicate_boxes(
            boxes, pred_classes, scores
        )

        # Handle case with no detections
        if not filtered_boxes:
            if progress_bar:
                progress_bar.progress(1.0)
                progress_text.markdown("100%")
            return orig_image, filtered_boxes, filtered_scores, filtered_classes

        # Draw detections
        for i, (box, class_name, score) in enumerate(
            zip(filtered_boxes, filtered_classes, filtered_scores)
        ):
            draw_detection_box(
                orig_image,
                box,
                class_name,
                score,
                orig_width,
                orig_height,
                IMAGE_SIZE,
            )

            # Update progress
            if progress_bar:
                progress = (i + 1) / len(filtered_boxes)
                progress_bar.progress(progress)
                progress_text.markdown(f"{int(progress * 100)}%")

        return orig_image, filtered_boxes, filtered_scores, filtered_classes
