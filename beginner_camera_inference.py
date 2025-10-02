# ======================== Phase 1: Basic Setup ========================
# Camera Integration, GPU Setup, Model Loading
# =====================================================================

import cv2
import torch
from ultralytics import YOLO
import os

# ============ CONFIGURATION ============
SOURCE = 0  # Camera: 0,1,2 or file: 'video.mp4' or stream: 'rtsp://url'
SEG_MODEL_PATH = "models/strawberrysegment.pt"
KPT_MODEL_PATH = "models/strawberrykeypoint.pt"
CONF = 0.6
SAVE_OUTPUT = False
OUTPUT_DIR = "outputs"
USE_KEYPOINTS = True  # Switch to use keypoint model
# =======================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------- GPU Setup ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# --------- Model Loading ----------
try:
    if USE_KEYPOINTS:
        model = YOLO(KPT_MODEL_PATH).to(DEVICE)
        print("[INFO] Keypoint model loaded successfully!")
    else:
        model = YOLO(SEG_MODEL_PATH).to(DEVICE)
        print("[INFO] Segmentation model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")

# --------- Camera Integration ----------
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print(f"[ERROR] Could not open camera/source: {SOURCE}")
else:
    print(f"[INFO] Camera/source {SOURCE} opened successfully.")

# --------- Simple Inference Loop ----------
print("[INFO] Starting inference. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break
    
    # Run YOLO inference on current frame
    results = model.predict(frame, conf=CONF, verbose=False)
    annotated_frame = results[0].plot()
    
    cv2.imshow("Phase 1: YOLO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Finished Phase 1.")
