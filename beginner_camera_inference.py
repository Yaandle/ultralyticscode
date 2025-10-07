import torch
from ultralytics import YOLO

# Check if CUDA-capable GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if device == 'cuda':
    # Display GPU information
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Enable PyTorch optimizations for faster inference
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable mixed precision for faster processing with less memory
    torch.set_float32_matmul_precision('medium')

# Load the YOLO model and move it to GPU
model = YOLO("yolov8n.pt")
model.to(device)

# Verify model is on the correct device
print(f"Model is running on: {next(model.model.parameters()).device}")

# Configure inference settings for optimal GPU performance
model.predict(
    source=0,           # Use webcam
    device=device,      # Run on GPU if available
    half=True,          # Use FP16 precision for faster inference
    show=True,          # Display results
    conf=0.4,           # Confidence threshold
    verbose=False       # Reduce console output
)
