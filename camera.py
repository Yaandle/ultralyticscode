import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import time

def map_coordinates(coords, img_shape):
    return coords

def get_depth_value(depth_frame, x, y):
    return depth_frame.get_distance(x, y)

def convert_depth_to_units(depth_value):
    depth_cm = depth_value * 100  
    depth_in = depth_cm / 2.54    
    return depth_cm, depth_in

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
            depth_cm, depth_in = convert_depth_to_units(depth_value)

            detection_data = {
                "class": class_name,
                "bbox": mapped_bbox,
                "confidence": conf,
                "center_point": (x_center, y_center),
                "depth_meters": depth_value,
                "depth_cm": depth_cm,
                "depth_in": depth_in
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
            print(f" Depth: {detection['depth_meters']:.2f} meters ({detection['depth_cm']:.2f} cm / {detection['depth_in']:.2f} inches)")
            print(f" Confidence: {detection['confidence']:.2f}")
            print()

def display_frame(frame, results, depth_frame):
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box, cls in zip(boxes.xyxy, boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            class_name = result.names[int(cls)]
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            depth_value = get_depth_value(depth_frame, x_center, y_center)
            depth_cm, depth_in = convert_depth_to_units(depth_value)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {depth_value:.2f}m ({depth_cm:.2f}cm / {depth_in:.2f}in)"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def process_realsense():
    model = YOLO("C:/Users/Zac/Desktop/MiFood/Applev5.pt")  # Update Model Path
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    fps = 0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            results = model(frame, conf=0.7, show_conf=False, show_labels=False, device=0)
            output = process_detections(results, frame.shape, depth_frame)
            print_detections(output)

            frame = display_frame(frame, results, depth_frame)
            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = end_time

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('RealSense Stream', cv2.resize(frame, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

process_realsense()
