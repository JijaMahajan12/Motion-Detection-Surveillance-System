"""Utility functions for drawing overlays and formatting timestamps."""

from datetime import datetime

import cv2
import numpy as np

from config import (
    RECORDING_INDICATOR_COLOR,
    STATUS_COLOR_MOTION,
    STATUS_COLOR_NO_MOTION,
)


def generate_timestamp(for_filename: bool = False) -> str:
    """
    Return a formatted timestamp string.

    Args:
        for_filename: If True, use filename-safe format (YYYY-MM-DD_HH-MM-SS).
                      Otherwise, use display format (YYYY-MM-DD HH:MM:SS).
    """
    now = datetime.now()
    if for_filename:
        return now.strftime("%Y-%m-%d_%H-%M-%S")
    return now.strftime("%Y-%m-%d %H:%M:%S")


def draw_timestamp(frame: np.ndarray, timestamp: str) -> None:
    """Draw the current date and time in the top-right corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)

    text_size, _ = cv2.getTextSize(timestamp, font, font_scale, thickness)
    x = frame.shape[1] - text_size[0] - 10
    y = text_size[1] + 10

    # Dark background for readability
    cv2.rectangle(
        frame,
        (x - 5, y - text_size[1] - 5),
        (x + text_size[0] + 5, y + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, timestamp, (x, y), font, font_scale, color, thickness)


def draw_status_text(frame: np.ndarray, motion_detected: bool) -> None:
    """Draw motion status in the top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    if motion_detected:
        text = "Motion Detected"
        color = STATUS_COLOR_MOTION
    else:
        text = "No Motion"
        color = STATUS_COLOR_NO_MOTION

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 10, text_size[1] + 10

    cv2.rectangle(
        frame,
        (x - 5, y - text_size[1] - 5),
        (x + text_size[0] + 5, y + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_recording_indicator(frame: np.ndarray) -> None:
    """Draw a red 'REC ●' indicator in the bottom-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text = "REC ●"
    color = RECORDING_INDICATOR_COLOR

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 10, frame.shape[0] - 10

    cv2.rectangle(
        frame,
        (x - 5, y - text_size[1] - 5),
        (x + text_size[0] + 5, y + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
