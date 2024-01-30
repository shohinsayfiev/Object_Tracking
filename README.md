# Object Tracking with YOLOv8 and SORT

This project combines YOLOv8 for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for tracking those objects. The implementation is intended for real-time tracking in video streams.

## Introduction

The code in this repository provides an integration of YOLOv8 for object detection and SORT for object tracking. The combination allows for robust tracking of objects in video sequences. The project is implemented in Python using popular libraries such as NumPy, OpenCV, and YOLOv8.

## Requirements

To run the code, ensure you have the following dependencies installed:

- Python (>=3.6)
- NumPy (1.21.2)
- Matplotlib (3.4.3)
- scikit-image (0.18.3)
- filterpy (1.4.5)
- lapx (0.4.0)

You can install these dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

- Download the YOLOv8 model file (`yolov8m.pt`) from the official YOLO repository.
- Set up the project by installing the required dependencies.
- Run the object detection and tracking using the provided script:
`python main_tracker.py`

Adjust the configuration parameters in `main_tracker.py` if needed.

## Configuration

You can customize the behavior of the tracker by modifying the configuration parameters in `main_tracker.py`. Some of the key parameters include:

- __max_age__: Maximum number of frames to keep a track alive without associated detections.
- __min_hits__: Minimum number of associated detections before a track is initialized.
- __iou_threshold__: Minimum Intersection over Union (IOU) for a match.

## Results

The tracking results are saved in the output directory, where each sequence has its own text file containing the tracked objects for each frame.
