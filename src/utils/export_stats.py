"""Statistics export utility for detection results."""

import json
import csv
from datetime import datetime
from typing import Dict, List
from io import StringIO


def create_detection_stats(
    file_name: str,
    file_type: str,
    mobil_count: int,
    motor_count: int,
    total_vehicles: int,
    duration: float,
    frames_processed: int = None,
    total_frames: int = None,
    threshold: float = 0.5,
    frame_skip: int = 0,
) -> Dict:
    """
    Create standardized detection statistics dictionary.

    Returns:
        Dictionary with detection statistics
    """
    stats = {
        "timestamp": datetime.now().isoformat(),
        "file_name": file_name,
        "file_type": file_type,
        "detection": {
            "mobil_count": mobil_count,
            "motor_count": motor_count,
            "total_vehicles": total_vehicles,
        },
        "optimization": {
            "traffic_light_duration_seconds": duration,
        },
        "parameters": {
            "confidence_threshold": threshold,
        },
    }

    if file_type == "video" and frames_processed:
        stats["video"] = {
            "frames_processed": frames_processed,
            "total_frames": total_frames or frames_processed,
            "frame_skip": frame_skip,
        }

    return stats


def export_to_json(stats: Dict) -> str:
    """Export statistics to JSON string."""
    return json.dumps(stats, indent=2)


def export_to_csv(stats: Dict) -> str:
    """Export statistics to CSV string."""
    output = StringIO()

    # Flatten nested dict for CSV
    flat_data = {
        "timestamp": stats["timestamp"],
        "file_name": stats["file_name"],
        "file_type": stats["file_type"],
        "mobil_count": stats["detection"]["mobil_count"],
        "motor_count": stats["detection"]["motor_count"],
        "total_vehicles": stats["detection"]["total_vehicles"],
        "traffic_light_duration_seconds": stats["optimization"][
            "traffic_light_duration_seconds"
        ],
        "confidence_threshold": stats["parameters"]["confidence_threshold"],
    }

    # Add video-specific fields if present
    if "video" in stats:
        flat_data.update(
            {
                "frames_processed": stats["video"]["frames_processed"],
                "total_frames": stats["video"]["total_frames"],
                "frame_skip": stats["video"]["frame_skip"],
            }
        )

    writer = csv.DictWriter(output, fieldnames=flat_data.keys())
    writer.writeheader()
    writer.writerow(flat_data)

    return output.getvalue()
