from ultralytics import YOLO
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load models
model = YOLO("models/segmentyolov11.pt")
keypoint_model = YOLO("models/pose_v8.pt")
source = "input"


output_dir = "visualization_stages"
os.makedirs(output_dir, exist_ok=True)



# STAGE 1: Object Detection (Boxes Only)
print("Running Stage 1: Object Detection...")
detection_results = model.predict(
    source=source,
    save=False,  
    conf=0.4,    
    verbose=True
)


for i, result in enumerate(detection_results):
    img_filename = os.path.basename(result.path)
    base_name = os.path.splitext(img_filename)[0]
    img = cv2.imread(result.path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    box_img = img.copy()

    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(box_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    boxes_path = f"{output_dir}/{base_name}_1_boxes.jpg"
    cv2.imwrite(boxes_path, cv2.cvtColor(box_img, cv2.COLOR_RGB2BGR))
    print(f"Saved object detection visualization: {boxes_path}")
    result.orig_img = img
    result.orig_path = result.path


# STAGE 2: Instance Segmentation (Masks Only)
print("Running Stage 2: Instance Segmentation...")

for i, det_result in enumerate(detection_results):
    img_filename = os.path.basename(det_result.path)
    base_name = os.path.splitext(img_filename)[0]
    img = det_result.orig_img.copy()
    mask_img = img.copy()
    
    if hasattr(det_result, 'masks') and det_result.masks is not None:
        masks = det_result.masks.xy

        for mask in masks:
            mask = np.array(mask, dtype=np.int32)
            overlay = mask_img.copy()
            cv2.fillPoly(overlay, [mask], (0, 255, 0))  
            mask_img = cv2.addWeighted(overlay, 0.5, mask_img, 0.5, 0)

    masks_path = f"{output_dir}/{base_name}_2_masks.jpg"
    cv2.imwrite(masks_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
    print(f"Saved segmentation visualization: {masks_path}")


# STAGE 3: Keypoint Detection
print("Running Stage 3: Keypoint Detection...")

for i, det_result in enumerate(detection_results):
    original_path = det_result.orig_path
    img_filename = os.path.basename(original_path)
    base_name = os.path.splitext(img_filename)[0]
    
    keypoint_results = keypoint_model.predict(
        source=original_path,
        conf=0.4,
        save=False,
        verbose=True
    )
    
    for kp_result in keypoint_results:
        img = cv2.imread(original_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kp_img = img.copy()
        
        if hasattr(kp_result, 'keypoints') and kp_result.keypoints is not None:
            keypoints = kp_result.keypoints.xy
            
            for kpts in keypoints:
                for kp in kpts:
                    x, y = int(kp[0]), int(kp[1])
                    if x > 0 and y > 0:
                        cv2.circle(kp_img, (x, y), 5, (255, 0, 0), -1)  
    
        keypoints_path = f"{output_dir}/{base_name}_3_keypoints.jpg"
        cv2.imwrite(keypoints_path, cv2.cvtColor(kp_img, cv2.COLOR_RGB2BGR))
        print(f"Saved keypoint visualization: {keypoints_path}")


# Create a combined visualization showing the progression
print("Creating combined visualization...")
for i, det_result in enumerate(detection_results):
    img_filename = os.path.basename(det_result.path)
    base_name = os.path.splitext(img_filename)[0]
    
    # Paths to the three visualization stages
    boxes_path = f"{output_dir}/{base_name}_1_boxes.jpg"
    masks_path = f"{output_dir}/{base_name}_2_masks.jpg"
    keypoints_path = f"{output_dir}/{base_name}_3_keypoints.jpg"
    
    if all(os.path.exists(p) for p in [boxes_path, masks_path, keypoints_path]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        boxes_img = plt.imread(boxes_path)
        masks_img = plt.imread(masks_path)
        keypoints_img = plt.imread(keypoints_path)
        
        axes[0].imshow(boxes_img)
        axes[0].set_title("Stage 1: Object Detection", fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(masks_img)
        axes[1].set_title("Stage 2: Instance Segmentation", fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(keypoints_img)
        axes[2].set_title("Stage 3: Keypoint Detection", fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        combined_path = f"{output_dir}/{base_name}_combined_stages.jpg"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved combined visualization: {combined_path}")

print("Processing complete! Visualizations saved to:", output_dir)
