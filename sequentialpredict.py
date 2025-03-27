### runs objectdetection/instance segmentation, then pose/keypoint on the bounxing boxes of the 1st models results.

from ultralytics import YOLO

model = YOLO("/computer_vision/models/segmentmodel.pt")
keypoint_model = YOLO("/computer_vision/models/keypointmodel.pt")
source = "/datasets/predict"

# Run inference
results = model(source, show_labels=True, show_boxes=True, show_conf=True, save=True)

for result in results:
    keypoint_results = keypoint_model(result.save_dir, show_boxes=False, show_labels=True, show_conf=True, save=True, conf=0.4,)
    
    for kp_result in keypoint_results:
        kp_result.save()
        
        if kp_result.keypoints:
            xy = kp_result.keypoints.xy  # x, y coordinates
            xyn = kp_result.keypoints.xyn  # normalized coordinates
            kpts = kp_result.keypoints.data  # keypoint data
