"""Motion detection using classical computer vision techniques."""

from typing import List, Tuple

import cv2
import numpy as np

from config import (
    BLUR_KERNEL,
    BOUNDING_BOX_COLOR,
    BOUNDING_BOX_THICKNESS,
    DILATION_ITERATIONS,
    MIN_CONTOUR_AREA,
    THRESHOLD_VALUE,
)


class MotionDetector:
    """
    Detects motion in video frames using frame differencing.

    Pipeline:
        1. Convert frame to grayscale
        2. Apply Gaussian blur to reduce noise
        3. Compute absolute difference with the previous frame
        4. Threshold the difference to create a binary mask
        5. Dilate the mask to fill small holes
        6. Find contours and filter by minimum area
        7. Generate bounding boxes around significant motion regions
    """

    def __init__(
        self,
        min_contour_area: int = MIN_CONTOUR_AREA,
        blur_kernel: Tuple[int, int] = BLUR_KERNEL,
        threshold_value: int = THRESHOLD_VALUE,
        dilation_iterations: int = DILATION_ITERATIONS,
    ) -> None:
        self.min_contour_area = min_contour_area
        self.blur_kernel = blur_kernel
        self.threshold_value = threshold_value
        self.dilation_iterations = dilation_iterations
        self._previous_gray: np.ndarray | None = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale and apply Gaussian blur."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        return gray

    def _find_motion_regions(
        self, frame_delta: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Threshold the frame difference, dilate, and extract bounding boxes.

        Returns:
            List of (x, y, w, h) bounding boxes and the binary motion mask.
        """
        # Step 4: Threshold — pixels above the threshold are considered motion
        _, thresh = cv2.threshold(
            frame_delta, self.threshold_value, 255, cv2.THRESH_BINARY
        )

        # Step 5: Dilation — connect nearby motion pixels and remove noise
        thresh = cv2.dilate(thresh, None, iterations=self.dilation_iterations)

        # Step 6: Contour detection on the binary mask
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bounding_boxes: List[Tuple[int, int, int, int]] = []

        for contour in contours:
            # Step 7: Ignore small contours (noise, lighting flicker)
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

        return bounding_boxes, thresh

    def _draw_bounding_boxes(
        self, frame: np.ndarray, bounding_boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Draw green rectangles around detected motion regions."""
        output = frame.copy()
        for x, y, w, h in bounding_boxes:
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                BOUNDING_BOX_COLOR,
                BOUNDING_BOX_THICKNESS,
            )
        return output

    def detect(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Process a frame and detect motion.

        Args:
            frame: BGR image from the webcam.

        Returns:
            motion_detected: True if significant motion was found.
            bounding_boxes: List of (x, y, w, h) tuples for motion regions.
            processed_frame: Frame with bounding boxes drawn (original if no motion).
        """
        gray = self._preprocess(frame)

        # First frame — store reference and report no motion
        if self._previous_gray is None:
            self._previous_gray = gray
            return False, [], frame.copy()

        # Step 3: Frame differencing — compare current frame with previous frame
        frame_delta = cv2.absdiff(self._previous_gray, gray)
        self._previous_gray = gray

        bounding_boxes, _ = self._find_motion_regions(frame_delta)
        motion_detected = len(bounding_boxes) > 0

        processed_frame = self._draw_bounding_boxes(frame, bounding_boxes)

        return motion_detected, bounding_boxes, processed_frame

    def reset(self) -> None:
        """Clear the stored reference frame (e.g., after camera reconnection)."""
        self._previous_gray = None
