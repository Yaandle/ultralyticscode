import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
from datetime import datetime
import os
import csv
from typing import Dict, List
import logging

class FruitDetectionSystem:
    def __init__(self, model_path, output_dir="collected_data"):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        self.setup_directories()
        self.detection_count = 0
        self.session_start = datetime.now()
        self.csv_filename = os.path.join(output_dir, f"detection_data_{self.session_start.strftime('%Y%m%d_%H%M%S')}.csv")
        
        self.init_csv()
        
        self.stats = {
            "total_detections": 0,
            "detections_by_class": {},
            "average_confidence": [],
            "frame_count": 0
        }
        
        self.logger.info(f"Initialized FruitDetectionSystem with model: {model_path}")
        self.logger.info(f"Data will be saved to: {output_dir}")

    def setup_directories(self):
        """Create necessary directories for data storage"""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, "images"),
            os.path.join(self.output_dir, "masks"),
            os.path.join(self.output_dir, "debug")
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def init_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            "timestamp", "detection_id", "class", "confidence",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "center_x", "center_y", "area", "image_path",
            "mask_path", "environmental_conditions",
            "frame_width", "frame_height"
        ]
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def print_detection_info(self, detection_data: Dict):
        """Print formatted detection information to console"""
        print("\n" + "="*50)
        print(f"Detection ID: {detection_data['detection_id']}")
        print(f"Timestamp: {detection_data['timestamp']}")
        print(f"Class: {detection_data['class']}")
        print(f"Confidence: {detection_data['confidence']:.3f}")
        print("\nBounding Box:")
        print(f"  Top-left: ({detection_data['bbox'][0]:.1f}, {detection_data['bbox'][1]:.1f})")
        print(f"  Bottom-right: ({detection_data['bbox'][2]:.1f}, {detection_data['bbox'][3]:.1f})")
        print(f"  Center: ({detection_data['center'][0]:.1f}, {detection_data['center'][1]:.1f})")
        print(f"  Area: {detection_data['area']:.1f} pixels²")
        print(f"\nSaved Images:")
        print(f"  Detection: {os.path.basename(detection_data['image_path'])}")
        if detection_data['mask_path']:
            print(f"  Mask: {os.path.basename(detection_data['mask_path'])}")
        print("="*50 + "\n")

    def print_frame_summary(self, detections: List[Dict]):
        """Print frame summary information"""
        if detections:
            print("\nFrame Summary:")
            print(f"Number of detections: {len(detections)}")
            classes = {}
            avg_conf = 0
            for det in detections:
                classes[det['class']] = classes.get(det['class'], 0) + 1
                avg_conf += det['confidence']
            
            print("Detected classes:")
            for cls, count in classes.items():
                print(f"  - {cls}: {count}")
            print(f"Average confidence: {(avg_conf/len(detections)):.3f}")
        else:
            print("\nNo detections in this frame")

    def print_session_stats(self):
        """Print current session statistics"""
        duration = datetime.now() - self.session_start
        print("\nSession Statistics:")
        print(f"Duration: {duration}")
        print(f"Total Frames: {self.stats['frame_count']}")
        print(f"Total Detections: {self.stats['total_detections']}")
        print("\nDetections by Class:")
        for class_name, count in self.stats['detections_by_class'].items():
            print(f"  - {class_name}: {count}")
        if self.stats['average_confidence']:
            print(f"Average Confidence: {np.mean(self.stats['average_confidence']):.3f}")
        print("\n")

    def process_detection(self, frame, results):
        """Process detection results and save data"""
        timestamp = datetime.now()
        frame_height, frame_width = frame.shape[:2]
        output_data = []
        display_frame = frame.copy()
        
        self.stats['frame_count'] += 1
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            masks = result.masks.cpu().numpy() if result.masks is not None else None
        
            if len(boxes) == 0:
                continue
            for i, (box, mask) in enumerate(zip(boxes.xyxy, masks)):
                self.detection_count += 1
                detection_id = f"D{self.detection_count}"
                x1, y1, x2, y2 = [int(coord) for coord in box]
                class_name = result.names[int(boxes.cls[i])]
                confidence = float(boxes.conf[i])
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                detection_img = frame[max(0, y1-10):min(frame_height, y2+10),
                                   max(0, x1-10):min(frame_width, x2+10)]
                img_path = ""
                if detection_img.size > 0:
                    img_path = os.path.join(self.output_dir, "images", 
                                          f"{detection_id}_{class_name}.jpg")
                    cv2.imwrite(img_path, detection_img)
                mask_path = ""
                if isinstance(mask, np.ndarray):
                    mask_path = os.path.join(self.output_dir, "masks", 
                                           f"{detection_id}_{class_name}_mask.png")
                    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                self.stats["total_detections"] += 1
                self.stats["detections_by_class"][class_name] = \
                    self.stats["detections_by_class"].get(class_name, 0) + 1
                self.stats["average_confidence"].append(confidence)
                detection_data = {
                    "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    "detection_id": detection_id,
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "center": [center_x, center_y],
                    "area": area,
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "frame_width": frame_width,
                    "frame_height": frame_height
                }
                csv_row = [
                    detection_data["timestamp"],
                    detection_id,
                    class_name,
                    confidence,
                    x1, y1, x2, y2,
                    center_x, center_y,
                    area,
                    img_path,
                    mask_path,
                    "Not Available",
                    frame_width,
                    frame_height
                ]
                with open(self.csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_row)

                self.print_detection_info(detection_data)
                self.draw_detection(display_frame, box, class_name, confidence, mask)
                
                output_data.append(detection_data)
        self.print_frame_summary(output_data)
        
        if self.stats['frame_count'] % 100 == 0:
            self.print_session_stats()

        return display_frame, output_data

    def draw_detection(self, frame, box, class_name, confidence, mask):
        """Draw detection visualization on frame"""
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with class and confidence
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw mask if available
        if isinstance(mask, np.ndarray):
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask = (mask > 0.5).astype(np.uint8) * 255
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = [0, 0, 255]
            frame = cv2.addWeighted(frame, 1, colored_mask, 0.3, 0)

    def process_webcam(self):
        """Main processing loop for webcam feed"""
        cap = cv2.VideoCapture(0)
        self.logger.info("Starting webcam capture")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to capture frame from webcam")
                    break

                # Perform detection
                results = self.model(frame, conf=0.3)
                
                # Process and visualize detections
                display_frame, detections = self.process_detection(frame, results)

                # Draw statistics
                self.draw_stats(display_frame)

                # Show frame
                cv2.imshow('Fruit Detection Stream', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("User requested stop")
                    break

                time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error in webcam processing: {str(e)}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_session_summary()
            self.logger.info("Session ended")

    def draw_stats(self, frame):
        """Draw detection statistics on frame"""
        stats_text = [
            f"Total Detections: {self.stats['total_detections']}",
            f"Session Duration: {str(datetime.now() - self.session_start).split('.')[0]}",
            "Detections by Class:"
        ]
        
        y_pos = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25

        for class_name, count in self.stats["detections_by_class"].items():
            text = f"  {class_name}: {count}"
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25

    def save_session_summary(self):
        """Save session summary statistics"""
        summary = {
            "session_start": self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
            "session_end": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_frames": self.stats["frame_count"],
            "total_detections": self.stats["total_detections"],
            "detections_by_class": self.stats["detections_by_class"],
            "average_confidence": np.mean(self.stats["average_confidence"]) \
                if self.stats["average_confidence"] else 0
        }

        summary_path = os.path.join(self.output_dir, 
                                  f"session_summary_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        self.logger.info(f"Session summary saved to {summary_path}")

if __name__ == "__main__":
    detector = FruitDetectionSystem("yolo11m-seg.pt")
    detector.process_webcam()
