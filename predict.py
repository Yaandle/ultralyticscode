# Runs inference on VIDEO and provides bounding box and centre point co-ordinates.

from ultralytics import YOLO
import time
import os
import cv2

def process_video(video_path):
    model = YOLO("models\MF Segment 3.0.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video_path = "output_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.1, device=0)

        xyxys = []
        confidences = []
        class_ids = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)
            for idx, box in enumerate(xyxys[-1]):
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)
                print(f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1}: Object {idx + 1} - Class {class_ids[-1][idx]}, Coordinates: {box}, Center Point: ({x_center}, {y_center})")

            for box in xyxys[-1]:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_path = "videos/video.mp4"
process_video(video_path)
