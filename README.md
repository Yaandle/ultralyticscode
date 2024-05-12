# Ultralytics YOLOv8 Inference

This repository contains three applications for running object detection and segmentation using the Ultralytics YOLOv8 model: image, video, and live video stream inference.

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install ultralytics`

## Usage

### Image Inference

1. Open `imagepredict.py`
2. Change `model_path` and `source`
3. Adjust `conf` and `device` if needed
4. Run: `python imagepredict.py`

### Video Inference

1. Open `videopredict.py`
2. Change `model_path` and `source`
3. Adjust `conf` and `device` if needed
4. Run: `python videopredict.py`

### Live Stream Inference

1. Open `stream.py`
2. Change `model_path`
3. Adjust `conf` and `device` if needed
4. Run: `python stream.py`

### Camera Stream Inference
* We are using Intel D145 *
1. 'pip install pyrealsense2'
2. Open `camera.py`
3. Change `model_path`
4. Adjust `conf` and `device` if needed
5. Run: `python camera.py`
   

## Notes

- Remove `device` parameter if not using GPU
- Ensure PyTorch is installed with CUDA and CuDNN support for GPU usage
