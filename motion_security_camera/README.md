# Real-Time Motion Detection Security Camera

A Python application that uses your webcam to detect motion in real time with classical computer vision techniques (no deep learning). When motion is detected, the app automatically records video and saves clips to a `recordings/` folder.

## Features

- **Live motion detection** — Frame differencing with Gaussian blur, thresholding, and contour analysis
- **Visual feedback** — Green bounding boxes around moving regions, motion status, and live timestamp
- **Automatic recording** — Starts on motion, continues during activity, and stops 3 seconds after motion ends
- **Timestamped clips** — Recordings saved as `YYYY-MM-DD_HH-MM-SS.mp4`
- **Modular design** — Separate modules for detection, recording, configuration, and utilities
- **Clean exit** — Press `Q` to quit; all camera and writer resources are released safely

## Folder Structure

```
motion_security_camera/
├── main.py              # Application entry point and main loop
├── motion_detector.py   # MotionDetector class (CV pipeline)
├── recorder.py          # VideoRecorder class (auto recording)
├── utils.py             # Drawing helpers and timestamp formatting
├── config.py            # Configurable parameters
├── recordings/          # Saved video clips (created automatically)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Installation

### Prerequisites

- Python 3.11 or newer
- A working webcam

### Setup

1. Clone or download this project and navigate into the directory:

   ```bash
   cd motion_security_camera
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS / Linux
   # venv\Scripts\activate    # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

```bash
python main.py
```

- A window titled **Motion Security Camera** will open showing the live feed.
- **Motion Detected** or **No Motion** appears in the top-left corner.
- The current date and time appear in the top-right corner.
- When recording, a red **REC ●** indicator appears in the bottom-left corner.
- Press **Q** to exit the application.

Recorded videos are saved in the `recordings/` directory.

## Motion Detection Pipeline

The application uses a classical computer vision pipeline (no AI models):

1. **Capture** — Read frames from the default webcam (`CAMERA_INDEX = 0`).
2. **Grayscale conversion** — Reduce the frame to a single channel for simpler comparison.
3. **Gaussian blur** — Smooth the image to suppress sensor noise and small variations.
4. **Frame differencing** — Compute the absolute difference between the current frame and the previous frame. Changed pixels indicate potential motion.
5. **Thresholding** — Convert the difference image to a binary mask; pixels above `THRESHOLD_VALUE` are treated as motion.
6. **Dilation** — Expand white regions in the mask to connect nearby motion pixels and fill small gaps.
7. **Contour detection** — Find connected regions in the binary mask.
8. **Area filtering** — Ignore contours smaller than `MIN_CONTOUR_AREA` to reduce false positives.
9. **Bounding boxes** — Draw green rectangles around regions that pass the area filter.

All tunable parameters live in `config.py`.

## Configuration

Edit `config.py` to adjust behavior:

| Parameter            | Default   | Description                                      |
|----------------------|-----------|--------------------------------------------------|
| `CAMERA_INDEX`       | `0`       | Webcam device index                              |
| `MIN_CONTOUR_AREA`   | `800`     | Minimum contour size to count as motion          |
| `BLUR_KERNEL`        | `(21, 21)`| Gaussian blur kernel size                        |
| `THRESHOLD_VALUE`    | `25`      | Pixel difference threshold for motion mask       |
| `DILATION_ITERATIONS`| `2`       | Number of dilation passes on the motion mask     |
| `NO_MOTION_TIMEOUT`  | `3`       | Seconds to keep recording after motion stops     |
| `FPS`                | `20`      | Recording frame rate                             |
| `FRAME_WIDTH`        | `640`     | Capture width                                    |
| `FRAME_HEIGHT`       | `480`     | Capture height                                   |

## Error Handling

The application handles common failure cases:

- **Webcam unavailable** — Displays an error and exits if the camera cannot be opened or read.
- **Missing recordings folder** — Creates `recordings/` automatically on startup.
- **Video write failure** — Logs an error if `VideoWriter` cannot be initialized.
- **Safe shutdown** — Releases the camera, stops any active recording, and closes OpenCV windows on exit.

## Future Improvements

The following enhancements are planned but not implemented in this version:

- Human detection using YOLO
- Face recognition
- Email alerts on motion events
- Snapshot capture on motion detection
- Cloud storage for recordings
- Multiple camera support

## License

This project is intended as a learning and portfolio piece. Use and modify freely.
