from ultralytics import YOLO
import torch
import torchvision
import time
import os

source = ''        # Define dataset source
model = YOLO("")   # Define YOLO model path


image_files = [f for f in os.listdir(source) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(source, image_file)
    results = model(image_path, save=True, save_txt=True, show_boxes=True, conf=0.8, device=0)   # For CPU remove 'device=0'
    

    time.sleep(0.5)
