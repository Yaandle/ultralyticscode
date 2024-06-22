import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import time

def map_coordinates(coords, img_shape):

    return coords

def get_depth_value(depth_frame, x, y):
    return depth_frame.get_distance(x, y)

def process_detections(detections, img_shape, depth_frame):
    output = []
    for detection in detections:
        boxes = detection.boxes.cpu().numpy()
        if len(boxes) == 0:
            continue
        for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            class_name = detection.names[int(cls)]
            mapped_bbox = map_coordinates(box, img_shape)
            x_center = int((mapped_bbox[0] + mapped_bbox[2]) / 2)
            y_center = int((mapped_bbox[1] + mapped_bbox[3]) / 2)
            depth_value = get_depth_value(depth_frame, x_center, y_center)

            detection_data = {
                "class": class_name,
                "bbox": mapped_bbox,
                "confidence": conf,
                "center_point": (x_center, y_center),
                "depth": depth_value
            }

            output.append(detection_data)
    return output

def print_detections(detections):
    if not detections:
        print("No objects detected.")
    else:
        for i, detection in enumerate(detections, start=1):
            print(f"Detection {i}:")
            print(f" Class: {detection['class']}")
            print(f" Bounding Box: {detection['bbox']}")
            print(f" Center Point: {detection['center_point']}")
            print(f" Depth: {detection['depth']:.2f} meters")
            print(f" Confidence: {detection['confidence']:.2f}")
            print()

def display_frame(frame, results):
    for result in results:
        # Ensure the frame is a numpy array and set relevant plot parameters
        frame = result.plot(conf=True, line_width=2, font_size=0.5, labels=True, boxes=True, masks=True, probs=True, img=frame)
    return frame

def process_realsense():
    model = YOLO("C:/Users/Bozzy/Desktop/MiFood/MiFood/fruits.pt")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            results = model(frame, conf=0.3, show_conf=True, show_labels=True)
            output = process_detections(results, frame.shape, depth_frame)
            print_detections(output)

            frame = display_frame(frame, results)
            cv2.imshow('RealSense Stream', cv2.resize(frame, (960, 540)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            time.sleep(0.5)  
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

process_realsense()
