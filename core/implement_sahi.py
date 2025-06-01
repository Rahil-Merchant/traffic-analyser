# import cv2
# from sahi.model import UltralyticsDetectionModel
# from sahi.predict import get_sliced_prediction
# from models.model_loader import load_model
# video = cv2.VideoCapture(f"C:/Users/EMAAN/Documents/YOLO/DAV Videos/part1.mp4")


# # Load your custom YOLOv8 model
# model_path = "C:/Users/EMAAN/Documents/YOLO/project/models/weights/silent_contract_9082_960px.pt"  # Replace with your actual path
# detection_model = UltralyticsDetectionModel(
#     model_path=model_path,
#     confidence_threshold=0.3,
#     device="cuda:0"  # or "cpu"
# )

# while True:
#     ret,cap = video.read()

import cv2
from sahi.model import UltralyticsYoloDetectionModel
from sahi.predict import get_sliced_prediction
import os

# ========== Configuration ==========
VIDEO_PATH = f"C:/Users/EMAAN/Documents/YOLO/DAV Videos/part1.mp4"
OUTPUT_PATH = "C:/Users/EMAAN/Documents/YOLO/project/detected_frames/sahi_output_video.mp4"
MODEL_PATH = "C:/Users/EMAAN/Documents/YOLO/project/models/weights/silent_contract_9082_960px.pt"
DEVICE = "cuda:0"  # use "cpu" if not using GPU
SLICE_SIZE = 512
OVERLAP_RATIO = 0.2
CONFIDENCE = 0.3
SHOW_LABELS = True
# ===================================

# Load SAHI-compatible YOLOv8 model
detection_model = UltralyticsYoloDetectionModel(
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE,
    device=DEVICE
)

# Read input video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Processing {frame_count} frames...")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = get_sliced_prediction(
        image=frame,
        detection_model=detection_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO
    )

    # Draw boxes
    visual_frame = result.image_with_boxes  # numpy image with boxes
    if visual_frame is not None:
        out.write(visual_frame)

    frame_idx += 1
    if frame_idx % 10 == 0:
        print(f"[INFO] Processed {frame_idx}/{frame_count} frames")

cap.release()
out.release()
print(f"[DONE] Output saved to: {OUTPUT_PATH}")
