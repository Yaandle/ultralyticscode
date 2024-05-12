import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np

def map_coordinates(coords, img_shape):
    return coords

def process_detections(detections, img_shape, depth_frame):
    output = []
    for detection in detections:
        boxes = detection.boxes.cpu().numpy()
        if len(boxes) == 0:
            continue
        for box, mask, conf, cls in zip(boxes.xyxy, detection.masks.cpu().numpy(), boxes.conf, boxes.cls):
            class_name = detection.names[int(cls)]
            mapped_bbox = map_coordinates(box, img_shape)
            x_center = int((mapped_bbox[0] + mapped_bbox[2]) / 2)
            y_center = int((mapped_bbox[1] + mapped_bbox[3]) / 2)
            center_point = (x_center, y_center)
            depth_value = depth_frame.get_distance(x_center, y_center)
            detection_data = {
                "class": class_name,
                "bbox": mapped_bbox,
                "mask": mask,
                "confidence": conf,
                "center_point": center_point,
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
            if detection['mask'] is not None:
                print(f" Segmentation Mask: {detection['mask'].shape}")
            print()

def process_realsense():
    model = YOLO("C:/Models/Strawberry V8.pt")                      #Update model path

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

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

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                label = result.names[int(cls)]
                confidence = conf.item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('RealSense Stream', cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

process_realsense()
