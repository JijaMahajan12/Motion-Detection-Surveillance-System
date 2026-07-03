"""Configuration parameters for the motion detection security camera."""

from pathlib import Path

# Camera settings
CAMERA_INDEX = 0
FPS = 20
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Motion detection parameters
MIN_CONTOUR_AREA = 800
BLUR_KERNEL = (21, 21)
THRESHOLD_VALUE = 25
DILATION_ITERATIONS = 2

# Recording settings
NO_MOTION_TIMEOUT = 3  # seconds to keep recording after motion stops
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"

# Display settings
WINDOW_NAME = "Motion Security Camera"
STATUS_COLOR_MOTION = (0, 255, 0)      # Green (BGR)
STATUS_COLOR_NO_MOTION = (0, 255, 255)  # Yellow (BGR)
RECORDING_INDICATOR_COLOR = (0, 0, 255)  # Red (BGR)
BOUNDING_BOX_COLOR = (0, 255, 0)       # Green (BGR)
BOUNDING_BOX_THICKNESS = 2
