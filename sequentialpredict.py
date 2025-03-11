### runs objectdetection/instance segmentation, then pose/keypoint on the bounxing boxes of the 1st models results.

from ultralytics import YOLO

model = YOLO("/computer_vision/models/strawberryv11.pt")
keypoint_model = YOLO("/computer_vision/models/strawberry_keypoint.pt")
source = "/computervision_datasets/predict"

# Run inference
results = model(source, show_labels=True, show_boxes=True, show_conf=True, save=True)

for result in results:
    # Save cropped detections automatically
    result.save_crop()
  
    keypoint_results = keypoint_model(result.save_dir, show_labels=True, show_conf=True, save=True)
    
    for kp_result in keypoint_results:
        kp_result.save()
        
        if kp_result.keypoints:
            xy = kp_result.keypoints.xy  # x, y coordinates
            xyn = kp_result.keypoints.xyn  # normalized coordinates
            kpts = kp_result.keypoints.data  # keypoint data
