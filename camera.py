import cv2
from ultralytics import YOLO
import numpy as np
import time
import pyrealsense2 as rs

def map_coordinates(coords, img_shape):
    return [int(c) for c in coords]

def calculate_width(mask, y_center):
    if mask.ndim == 2:
        center_row = mask[y_center]
    elif mask.ndim == 3:
        center_row = mask[y_center, :, 0] 
    else:
        return None
    true_indices = np.where(center_row > 0.5)[0]  
    if len(true_indices) > 0:
        left_edge = true_indices[0]
        right_edge = true_indices[-1]
        return right_edge - left_edge
    return None

def process_detections(detections, img_shape):
    output = []
    for detection in detections:
        boxes = detection.boxes.cpu().numpy()
        masks = detection.masks.data.cpu().numpy() if detection.masks is not None else None
        for i, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
            class_name = detection.names[int(cls)]
            mapped_bbox = map_coordinates(box, img_shape)
            x_center = int((mapped_bbox[0] + mapped_bbox[2]) / 2)
            y_center = int((mapped_bbox[1] + mapped_bbox[3]) / 2)
            width_pixels = None
            if masks is not None:
                mask = masks[i]
                width_pixels = calculate_width(mask, y_center)

            detection_data = {
                "class": class_name,
                "bbox": mapped_bbox,
                "confidence": conf,
                "center_point": (x_center, y_center),
                "width_pixels": width_pixels,
                "mask": mask if masks is not None else None
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
            print(f" Confidence: {detection['confidence']:.2f}")
            if detection['width_pixels'] is not None:
                print(f" Width: {detection['width_pixels']} pixels")
            print()

class TrackedObject:
    def __init__(self, detection, object_id):
        self.id = object_id
        self.class_name = detection['class']
        self.center = detection['center_point']
        self.last_seen = time.time()

    def update(self, detection):
        self.center = detection['center_point']
        self.last_seen = time.time()

class ObjectTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.counts = {}

    def register(self, detection):
        object_id = self.next_object_id
        self.objects[object_id] = TrackedObject(detection, object_id)
        self.disappeared[object_id] = 0
        self.next_object_id += 1
        self.update_count(detection['class'], 1)

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
        else:
            object_ids = list(self.objects.keys())
            object_centers = np.array([obj.center for obj in self.objects.values()])
            detection_centers = np.array([d['center_point'] for d in detections])

            distances = np.linalg.norm(object_centers[:, np.newaxis] - detection_centers, axis=2)
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if distances[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id].update(detections[col])
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(distances.shape[0])) - used_rows
            unused_cols = set(range(distances.shape[1])) - used_cols

            if distances.shape[0] >= distances.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(detections[col])

        return self.objects

    def update_count(self, class_name, count):
        if class_name not in self.counts:
            self.counts[class_name] = count
        else:
            self.counts[class_name] += count

    def get_counts(self):
        return self.counts

def display_frame(frame, detections, tracker):
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        class_name = detection['class']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if detection['width_pixels'] is not None:
            width_label = f"Width: {detection['width_pixels']} pixels"
            cv2.putText(frame, width_label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if detection['mask'] is not None:
            mask = detection['mask'].astype(np.uint8) * 255
            mask = cv2.resize(mask, (x2 - x1, y2 - y1))
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            subframe = frame[y1:y2, x1:x2]
            if subframe.shape[:2] == colored_mask.shape[:2]:
                alpha = 0.5
                cv2.addWeighted(colored_mask, alpha, subframe, 1 - alpha, 0, subframe)
                frame[y1:y2, x1:x2] = subframe

    counts = tracker.get_counts()
    y_offset = 60
    for class_name, count in counts.items():
        cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30

    return frame

def create_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return None
    return cap

def process_webcam():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
    pipeline.start(config)

    model = YOLO("")  # Update Model Path
    
    tracker = ObjectTracker()
    fps = 0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            infrared_frame = frames.get_infrared_frame()
            
            if not depth_frame or not color_frame or not infrared_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            infrared_image = np.asanyarray(infrared_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            

            results = model(color_image, conf=0.3, show_conf=False, show_labels=False, device=0)               # Configure predict arguments
            output = process_detections(results, color_image.shape)
            print_detections(output)
            tracker.update(output)
            color_image = display_frame(color_image, output, tracker)
            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = end_time
            cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('RGB Camera', cv2.resize(color_image, (640, 480)))
            cv2.imshow('Depth Camera', cv2.resize(depth_colormap, (640, 480)))
            cv2.imshow('Infrared Camera', cv2.resize(infrared_image, (640, 480)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam()
