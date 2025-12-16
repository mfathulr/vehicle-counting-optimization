"""Utility functions for preprocessing and visualization."""

import cv2
import numpy as np
from typing import Tuple, Optional

from ..config import IMAGE_SIZE, MASK_PATH, MIN_BOX_SIZE, CLASS_COLORS, CLASS_NAMES


def load_and_apply_mask(
    image: np.ndarray, mask_path: Optional[str] = None
) -> np.ndarray:
    """
    Load mask and apply it to the image.

    Args:
        image: Input image in BGR format.
        mask_path: Path to mask image. If None, uses default from config.

    Returns:
        Masked image.
    """
    mask_path = mask_path or str(MASK_PATH)
    mask = cv2.imread(mask_path)

    if mask is None:
        # Soft fallback: return original image unmasked
        # This avoids runtime failure when mask asset is missing
        return image

    # Resize mask to match image dimensions
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Convert to same data type and apply mask
    image = image.astype(np.float32)
    mask = mask.astype(np.float32)

    return cv2.bitwise_and(image, mask)


def preprocess_image(
    image: np.ndarray, target_size: int = IMAGE_SIZE, apply_mask: bool = True
) -> Tuple[np.ndarray, int, int]:
    """
    Preprocess image for model inference.

    Args:
        image: Input image in BGR format.
        target_size: Target size for resizing.
        apply_mask: Whether to apply mask to the image.

    Returns:
        Tuple of (preprocessed image tensor, original height, original width).
    """
    orig_height, orig_width = image.shape[:2]

    # Apply mask if requested
    if apply_mask:
        image = load_and_apply_mask(image)

    # Resize
    if target_size is not None:
        image = cv2.resize(image, (target_size, target_size))

    # Convert to RGB and normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0

    # Transpose to CHW format
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)

    return image, orig_height, orig_width


def draw_detection_box(
    image: np.ndarray,
    box: np.ndarray,
    class_name: str,
    score: float,
    orig_width: int,
    orig_height: int,
    input_size: int = IMAGE_SIZE,
) -> None:
    """
    Draw bounding box with label on image (in-place).

    Args:
        image: Image to draw on (BGR format).
        box: Bounding box coordinates [xmin, ymin, xmax, ymax].
        class_name: Class label.
        score: Detection confidence score.
        orig_width: Original image width.
        orig_height: Original image height.
        input_size: Model input size for coordinate scaling.
    """
    xmin, ymin, xmax, ymax = box.tolist()

    # Scale coordinates back to original size
    xmin = int(xmin * orig_width / input_size)
    ymin = int(ymin * orig_height / input_size)
    xmax = int(xmax * orig_width / input_size)
    ymax = int(ymax * orig_height / input_size)

    # Skip very small boxes
    if (ymax - ymin <= MIN_BOX_SIZE) or (xmax - xmin <= MIN_BOX_SIZE):
        return

    # Get color for class
    color = CLASS_COLORS.get(class_name, (0, 0, 0))

    # Draw bounding box
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

    # Prepare label text
    label_text = f"{class_name} {score:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    # Draw label background
    cv2.rectangle(
        image,
        (xmin, ymin - text_height - 5),
        (xmin + text_width, ymin),
        color,
        -1,
    )

    # Draw label text
    cv2.putText(
        image,
        label_text,
        (xmin, ymin - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        lineType=cv2.LINE_AA,
    )


def filter_duplicate_boxes(
    boxes: np.ndarray, classes: list, scores: np.ndarray
) -> Tuple[list, list, list]:
    """
    Filter out duplicate boxes with the same coordinates.

    Args:
        boxes: Array of bounding boxes.
        classes: List of class names.
        scores: Array of confidence scores.

    Returns:
        Tuple of (filtered_boxes, filtered_classes, filtered_scores).
    """
    filtered_boxes = []
    filtered_classes = []
    filtered_scores = []
    seen_coordinates = set()

    for box, class_name, score in zip(boxes, classes, scores):
        coordinates = tuple(box.tolist())
        if coordinates not in seen_coordinates:
            filtered_boxes.append(box)
            filtered_classes.append(class_name)
            filtered_scores.append(score)
            seen_coordinates.add(coordinates)

    return filtered_boxes, filtered_classes, filtered_scores


def count_classes(classes: list) -> dict:
    """
    Count occurrences of each class.

    Args:
        classes: List of class names.

    Returns:
        Dictionary mapping class names to counts.
    """
    class_counts = {class_name: 0 for class_name in CLASS_NAMES[1:]}
    for class_name in classes:
        if class_name in class_counts:
            class_counts[class_name] += 1
    return class_counts
