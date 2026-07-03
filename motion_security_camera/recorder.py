"""Automatic video recording when motion is detected."""

import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from config import FPS, FRAME_HEIGHT, FRAME_WIDTH, NO_MOTION_TIMEOUT, RECORDINGS_DIR
from utils import generate_timestamp


class VideoRecorder:
    """
    Manages automatic video recording triggered by motion detection.

    Starts recording when motion is detected, continues while motion exists,
    and keeps recording for a configurable timeout after motion stops to
    avoid fragmented clips.
    """

    def __init__(
        self,
        output_dir: Path = RECORDINGS_DIR,
        fps: int = FPS,
        frame_size: Tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT),
        no_motion_timeout: float = NO_MOTION_TIMEOUT,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.frame_size = frame_size
        self.no_motion_timeout = no_motion_timeout

        self._writer: cv2.VideoWriter | None = None
        self._is_recording = False
        self._last_motion_time: float | None = None
        self._current_filepath: Path | None = None

        self._ensure_output_directory()

    def _ensure_output_directory(self) -> None:
        """Create the recordings directory if it does not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filepath(self) -> Path:
        """Generate a timestamped output filename."""
        filename = f"{generate_timestamp(for_filename=True)}.mp4"
        return self.output_dir / filename

    def _start_recording(self) -> None:
        """Initialize VideoWriter and begin a new recording session."""
        if self._is_recording:
            return

        filepath = self._generate_filepath()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(filepath),
            fourcc,
            self.fps,
            self.frame_size,
        )

        if not writer.isOpened():
            raise IOError(f"Failed to open video writer for: {filepath}")

        self._writer = writer
        self._current_filepath = filepath
        self._is_recording = True
        print(f"[Recorder] Started recording: {filepath.name}")

    def _stop_recording(self) -> None:
        """Release VideoWriter and finalize the current recording."""
        if not self._is_recording or self._writer is None:
            return

        self._writer.release()
        print(f"[Recorder] Stopped recording: {self._current_filepath.name}")
        self._writer = None
        self._is_recording = False
        self._current_filepath = None
        self._last_motion_time = None

    def update(self, motion_detected: bool, frame: np.ndarray) -> bool:
        """
        Update recording state based on motion and write the current frame.

        Args:
            motion_detected: Whether motion was detected in the current frame.
            frame: BGR frame to write if recording is active.

        Returns:
            True if currently recording, False otherwise.
        """
        current_time = time.time()

        if motion_detected:
            self._last_motion_time = current_time
            if not self._is_recording:
                try:
                    self._start_recording()
                except IOError as exc:
                    print(f"[Recorder] Error: {exc}")
                    return False

        if self._is_recording and self._writer is not None:
            self._writer.write(frame)

            if (
                not motion_detected
                and self._last_motion_time is not None
                and (current_time - self._last_motion_time) >= self.no_motion_timeout
            ):
                self._stop_recording()

        return self._is_recording

    @property
    def is_recording(self) -> bool:
        """Return whether a recording session is currently active."""
        return self._is_recording

    def release(self) -> None:
        """Stop any active recording and release resources."""
        if self._is_recording:
            self._stop_recording()
