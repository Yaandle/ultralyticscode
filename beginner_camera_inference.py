import pyrealsense2 as rs
import torch
from ultralytics import YOLO

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(cfg)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        return aligned.get_color_frame(), aligned.get_depth_frame()

def setup_model(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return YOLO(path).to(device)

def run_inference(model, source, out="outputs"):
    return model.predict(
        source=source,
        save=True,
        conf=0.4,
        project=out,
        device=0 if torch.cuda.is_available() else "cpu"
    )

class StrawberryDetection:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seg = YOLO("models/strawberrysegment.pt").to(device)
        self.kpt = YOLO("models/strawberrykeypoint.pt").to(device)

    def run(self, image):
        seg = self.seg.predict(image, conf=0.4, iou=0.5, verbose=False)
        kpt = self.kpt.predict(image, conf=0.4, verbose=False)
        return seg, kpt

if __name__ == "__main__":
    det = setup_model("models/strawberry_detection.pt")
    run_inference(det, "sample.jpg")
    pipeline = StrawberryDetection()
    seg, kpt = pipeline.run("sample.jpg")
    print(seg, kpt)
