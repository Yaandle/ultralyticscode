from ultralytics import YOLO
import torch
import torchvision
import time
import os

source = 'E:\odis\odistesting\Testing\dataset\imaegimages' 
model = YOLO("E:\odis\odistesting\Testing\models\Model4600.pt")


image_files = [f for f in os.listdir(source) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(source, image_file)
    results = model(image_path, save=True, save_txt=True, conf=0.1, device=0)
    

    time.sleep(0.5)
