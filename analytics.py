from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

# Configuration
SOURCE_DIR = "input"                               #Update Input Path
MODEL_PATH = "models/model.pt"                     #Update Model Path
OUTPUT_DIR = "runs/detect"                         #Update Output Path
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_images():
    model = YOLO(MODEL_PATH)
    
    image_files = [
        f for f in os.listdir(SOURCE_DIR) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    class_counts = {}
    
    for image_file in image_files:
        image_path = os.path.join(SOURCE_DIR, image_file)
        results = model(
            image_path, 
            save=True, 
            project=OUTPUT_DIR,
            name='results', 
            show_labels=True, 
            show_conf=False, 
            conf=0.6
        )
        
        for result in results:
            if result.boxes is not None:
                classes = result.boxes.cls.cpu().numpy()
                for cls in classes:
                    class_name = result.names[int(cls)]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

    if class_counts:
        plt.figure(figsize=(10, 6))
        plt.pie(
            list(class_counts.values()), 
            labels=list(class_counts.keys()), 
            autopct='%1.1f%%'
        )
        plt.title('Fruit Detection Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'detection_analytics.png'))
        plt.close()
        
        # Print detection summary
        print("Detection Summary:")
        for cls, count in class_counts.items():
            print(f"{cls}: {count}")

if __name__ == "__main__":
    process_images()
