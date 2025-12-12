"""Video detection module for vehicle counting."""

import cv2
import torch
import numpy as np
import time
from typing import Tuple, Optional

from ..config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    VIDEO_CODEC,
    MIN_BOX_SIZE,
)
from ..utils.preprocessing import preprocess_image, filter_duplicate_boxes


class VideoDetector:
    """Handles vehicle detection on video streams."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Initialize the video detector.

        Args:
            model: Trained Faster R-CNN model.
            device: Device to run inference on ('cpu' or 'cuda').
        """
        self.model = model
        self.device = device
        self.model.to(device).eval()

    def process_video(
        self,
        video_path: str,
        output_path: str,
        threshold: float = DEFAULT_IOU_THRESHOLD,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[dict, float]:
        """
        Process video file and detect vehicles frame by frame.

        Args:
            video_path: Path to input video.
            output_path: Path to save output video.
            threshold: Confidence threshold for detections.
            progress_callback: Callback function for progress updates.

        Returns:
            Tuple of (final class counts dict, average FPS).
        """
        video = cv2.VideoCapture(video_path)

        # Get video properties
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        processed_frames = 0
        total_fps = 0
        prev_time = time.time()
        class_colors_bgr = {
            "mobil": (0, 255, 0),
            "motor": (0, 0, 255),
        }
        final_counts = {"mobil": 0, "motor": 0}

        # Process each frame
        while video.isOpened():
            ret, frame = video.read()

            if not ret:
                break

            processed_frames += 1
            image = frame.copy()

            # Preprocess frame
            processed_image, _, _ = preprocess_image(
                image, target_size=IMAGE_SIZE, apply_mask=True
            )

            # Convert to tensor
            image_tensor = torch.tensor(processed_image, dtype=torch.float)
            image_tensor = torch.unsqueeze(image_tensor, 0)

            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor.to(self.device))

            # Post-process outputs
            outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

            if len(outputs[0]["boxes"]) != 0:
                boxes = outputs[0]["boxes"].data.numpy()
                scores = outputs[0]["scores"].data.numpy()
                labels = outputs[0]["labels"].cpu().numpy()

                # Apply threshold
                keep = scores >= threshold
                boxes = boxes[keep].astype(np.int32)
                scores = scores[keep]
                labels = labels[keep]
                pred_classes = [CLASS_NAMES[i] for i in labels]

                # Filter duplicates
                filtered_boxes, filtered_classes, filtered_scores = (
                    filter_duplicate_boxes(boxes, pred_classes, scores)
                )

                # Draw detections
                for box, class_name, score in zip(
                    filtered_boxes, filtered_classes, filtered_scores
                ):
                    xmin, ymin, xmax, ymax = box

                    # Scale coordinates
                    xmin = int(xmin * frame_width / IMAGE_SIZE)
                    ymin = int(ymin * frame_height / IMAGE_SIZE)
                    xmax = int(xmax * frame_width / IMAGE_SIZE)
                    ymax = int(ymax * frame_height / IMAGE_SIZE)

                    # Skip small boxes
                    if (ymax - ymin <= MIN_BOX_SIZE) or (xmax - xmin <= MIN_BOX_SIZE):
                        continue

                    color = class_colors_bgr.get(class_name, (0, 0, 0))

                    # Draw box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                    # Draw label
                    label = f"{class_name} {score:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )

                    cv2.rectangle(
                        frame,
                        (xmin, ymin - text_height - 5),
                        (xmin + text_width, ymin),
                        color,
                        -1,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        lineType=cv2.LINE_AA,
                    )

                # Update counts
                final_counts["mobil"] = filtered_classes.count("mobil")
                final_counts["motor"] = filtered_classes.count("motor")

            # Calculate FPS
            current_fps = 1 / (1e-6 + time.time() - prev_time)
            total_fps += current_fps
            prev_time = time.time()

            # Write frame
            out.write(frame)

            # Update progress
            if progress_callback:
                progress = processed_frames / total_frames
                progress_callback(progress, frame)

        # Cleanup
        video.release()
        out.release()
        cv2.destroyAllWindows()

        avg_fps = total_fps / processed_frames if processed_frames > 0 else 0
        return final_counts, avg_fps
