# Streams RGB and every 10 secs takes a frame and runs inference and prints coordinates to table in terminal.



import time
import numpy as np
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO

model = YOLO("/strawberrysegment.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


align_to = rs.stream.color
align = rs.align(align_to)
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"[INFO] Depth scale: {depth_scale} meters/unit")

def get_3d_coords(depth_frame, x, y, intrinsics):
    """Convert pixel to 3D world coordinate."""
    depth = depth_frame.get_distance(x, y)
    if depth == 0:
        return None
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    return point  # in meters

last_detection_time = 0
detection_frame = None

try:
    while True:
        
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    
        cv2.imshow("Live RGB", color_image)


        current_time = time.time()
        if current_time - last_detection_time >= 10:
            last_detection_time = current_time
            results = model(color_image)[0]
            detection_frame = color_image.copy()

            print("\nDetected Objects:")
            print(f"{'Class':<15} {'X (m)':<10} {'Y (m)':<10} {'Z (m)':<10}")
            print("-" * 45)

            for r in results.boxes:
                cls_id = int(r.cls[0])
                class_name = model.names[cls_id]
                bbox = r.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                coords = get_3d_coords(depth_frame, cx, cy, depth_intrinsics)
                if coords:
                    X, Y, Z = map(lambda x: round(x, 3), coords)
                    print(f"{class_name:<15} {X:<10} {Y:<10} {Z:<10}")

                    # Annotate frame
                    cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{class_name} ({X:.2f}, {Y:.2f}, {Z:.2f}m)"
                    cv2.putText(detection_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        if detection_frame is not None:
            cv2.imshow("Last Detection", detection_frame)
        key = cv2.waitKey(1)
        if key == 27:  
            break

finally:
    print("[INFO] Stopping pipeline...")
    pipeline.stop()
    cv2.destroyAllWindows()
