"""Real-Time Motion Detection Security Camera — main entry point."""

import sys

import cv2

import config
from motion_detector import MotionDetector
from recorder import VideoRecorder
from utils import draw_recording_indicator, draw_status_text, draw_timestamp, generate_timestamp


def initialize_camera() -> cv2.VideoCapture:
    """
    Open the default webcam and configure capture properties.

    Raises:
        RuntimeError: If the webcam cannot be opened or does not return frames.
    """
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open webcam at index {config.CAMERA_INDEX}. "
            "Check that a camera is connected and not in use by another application."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    ret, _ = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Webcam opened but failed to read an initial frame.")

    return cap


def run() -> None:
    """Main application loop."""
    cap = None
    recorder = None

    try:
        cap = initialize_camera()
        detector = MotionDetector()
        recorder = VideoRecorder()

        print("Motion Security Camera started.")
        print("Press 'Q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from webcam. Exiting.")
                break

            # Motion detection pipeline
            motion_detected, _, processed_frame = detector.detect(frame)

            # Automatic recording (Phase 2)
            is_recording = recorder.update(motion_detected, processed_frame)

            # UI overlays
            draw_status_text(processed_frame, motion_detected)
            draw_timestamp(processed_frame, generate_timestamp())
            if is_recording:
                draw_recording_indicator(processed_frame)

            cv2.imshow(config.WINDOW_NAME, processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("Quit key pressed. Shutting down.")
                break

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    finally:
        if recorder is not None:
            recorder.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Resources released. Goodbye.")


if __name__ == "__main__":
    run()
