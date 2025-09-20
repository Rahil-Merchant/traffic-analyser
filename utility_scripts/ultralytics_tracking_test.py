import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\EMAAN\Documents\YOLO\project\models\weights\few_anchowy.engine"

model = YOLO(MODEL_PATH)
video_path = r"../../5 minute test - 4 Way Intersection.mp4"
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])

BATCH_SIZE = 64

while cap.isOpened():
    frames = []
    original_frames = []

    # Collect up to 64 frames
    for _ in range(BATCH_SIZE):
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
        original_frames.append(frame.copy())

    if not frames:
        break

    # Run batch inference
    results = model.track(frames, persist=True, batch=BATCH_SIZE)

    # Loop through results for each frame
    for idx, result in enumerate(results):
        annotated_frame = result.plot()
        boxes = result.boxes.xywh.cpu() if result.boxes is not None else []
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []

        # Draw tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        # Show frame
        cv2.imshow("YOLO11 Tracking", cv2.resize(annotated_frame,(0,0),fx=0.4,fy=0.4))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
