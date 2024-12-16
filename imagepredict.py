from ultralytics import YOLO
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, List, Optional
import os
import logging

@dataclass
class DetectionResult:
    fruit_center: Tuple[float, float]
    stem_point: Optional[Tuple[float, float]]
    grab_point: Optional[Tuple[float, float]]
    confidence: float
    bbox: Tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None

class StrawberryPickerVision:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        try:
            self.model = YOLO(model_path)
            self.confidence_threshold = confidence_threshold
            logging.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            raise

    def find_stem_point(self, mask: np.ndarray, bbox: List[float]) -> Optional[Tuple[float, float]]:
        x1, y1, x2, y2 = map(int, bbox)
        mask_region = mask[y1:y2, x1:x2]
        topmost_points = np.where(mask_region > 0)

        if len(topmost_points[0]) == 0:
            return None

        top_y = np.min(topmost_points[0])
        corresponding_x = topmost_points[1][topmost_points[0] == top_y]
        top_x = int(np.mean(corresponding_x))

        return (x1 + top_x, y1 + top_y)

    def calculate_grab_point(self, stem_point: Tuple[float, float], fruit_center: Tuple[float, float], offset: float = 50) -> Optional[Tuple[float, float]]:
        if stem_point is None:
            return None

        dx = stem_point[0] - fruit_center[0]
        dy = stem_point[1] - fruit_center[1]

        length = np.sqrt(dx**2 + dy**2)

        grab_x = stem_point[0] - (dx / length) * offset
        grab_y = stem_point[1] - (dy / length) * offset

        return (grab_x, grab_y)

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.4):
        mask = (mask > 0).astype(np.uint8)
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask > 0] = color
        return cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0)

    def process_image(self, image_path: str) -> Tuple[List[DetectionResult], np.ndarray]:
        results = self.model(image_path, conf=self.confidence_threshold)
        detections = []
        img = cv2.imread(image_path)
        vis_img = img.copy()

        for result in results:
            if hasattr(result, 'masks'):
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.data.cpu().numpy()

                    for mask, box in zip(masks, boxes):
                        x1, y1, x2, y2 = box[:4]
                        fruit_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        stem_point = self.find_stem_point(mask, [x1, y1, x2, y2])
                        grab_point = self.calculate_grab_point(stem_point, fruit_center)

                        detection = DetectionResult(
                            fruit_center=fruit_center,
                            stem_point=stem_point,
                            grab_point=grab_point,
                            confidence=box[4],
                            bbox=tuple(box[:4]),
                            mask=mask
                        )
                        detections.append(detection)
                        self._draw_detection(vis_img, detection)

                        mask = (mask > 0).astype(np.uint8)
                        vis_img[mask > 0] = img[mask > 0]

        return detections, vis_img

    def _draw_detection(self, img: np.ndarray, detection: DetectionResult):
        x1, y1, x2, y2 = map(int, detection.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cx, cy = map(int, detection.fruit_center)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(img, "Center", (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if detection.stem_point:
            sx, sy = map(int, detection.stem_point)
            cv2.circle(img, (sx, sy), 5, (0, 0, 255), -1)
            cv2.putText(img, "Stem", (sx + 5, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if detection.grab_point:
            gx, gy = map(int, detection.grab_point)
            cv2.circle(img, (gx, gy), 5, (255, 255, 0), -1)
            cv2.putText(img, "Grab", (gx + 5, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )

    picker = StrawberryPickerVision(
        model_path="",
        confidence_threshold=0.5
    )

    source = ""
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(source, image_file)
        logging.info(f"Processing: {image_file}")

        try:
            detections, vis_img = picker.process_image(image_path)

            for i, detection in enumerate(detections):
                logging.info(f"\nStrawberry {i + 1}:")
                logging.info(f"  Confidence: {detection.confidence:.2f}")
                logging.info(f"  Fruit center: ({int(detection.fruit_center[0])}, {int(detection.fruit_center[1])})")

                if detection.stem_point:
                    logging.info(f"  Stem point: ({int(detection.stem_point[0])}, {int(detection.stem_point[1])})")

                if detection.grab_point:
                    logging.info(f"  Grab point: ({int(detection.grab_point[0])}, {int(detection.grab_point[1])})")

            output_path = os.path.join(output_dir, f"detected_{image_file}")
            cv2.imwrite(output_path, vis_img)
            logging.info(f"Visualization saved: {output_path}")

        except Exception as e:
            logging.error(f"Error processing {image_file}: {str(e)}")

if __name__ == "__main__":
    main()
