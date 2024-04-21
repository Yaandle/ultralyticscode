Ultralytics YOLOv8 Inference
This repository contains three applications for running object detection and segmentation using the Ultralytics YOLOv8 model: image, video, and live video stream inference.

Getting Started
Clone the repository
Install dependencies: pip install ultralytics
Usage
Image Inference
Open image_inference.py
Change model_path and source
Adjust conf and device if needed
Run: python image_inference.py
Video Inference
Open video_inference.py
Change model_path and source
Adjust conf and device if needed
Run: python video_inference.py
Live Stream Inference
Open stream.py
Change model_path
Adjust conf and device if needed
Run: python stream.py
Notes
Remove device parameter if not using GPU
Ensure PyTorch is installed with CUDA and CuDNN support for GPU usage
