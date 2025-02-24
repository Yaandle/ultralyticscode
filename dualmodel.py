## Use Segmentation and Pose YOLOv11 Models to predict on folder of images.


import torch
import numpy as np
from ultralytics import YOLO
import cv2
import os


detection_model = YOLO("//computer_vision/models/strawberryv11.pt")
keypoint_model = YOLO("computer_vision/models/strawberrysegmentYOLOv11.pt")


source_folder = "/computervision_datasets/predict"

if not os.path.exists(source_folder):
    raise FileNotFoundError(f"Source folder '{source_folder}' does not exist.")

image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    raise ValueError(f"No image files found in '{source_folder}'.")

output_folder = os.path.join(source_folder, "output")
os.makedirs(output_folder, exist_ok=True)

for image_file in image_files:
    image_path = os.path.join(source_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping {image_file}: Unable to load image.")
        continue

    detection_results = detection_model(image_path, save=True, save_dir=output_folder)
    keypoint_results = keypoint_model(image_path, save=True, save_dir=output_folder)
