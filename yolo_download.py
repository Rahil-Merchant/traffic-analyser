from ultralytics import YOLO
import shutil
import os

model = YOLO('yolov8l.pt')

target_dir = r'models\weights'
os.makedirs(target_dir, exist_ok=True)

default_model_path = model.ckpt_path
shutil.copy(default_model_path, os.path.join(target_dir, 'yolov8l.pt'))

print(f"Model copied to: {os.path.join(target_dir, 'yolov8l.pt')}")
