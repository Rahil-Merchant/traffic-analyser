import os
import glob
from roboflow import Roboflow
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize Roboflow client
rf = Roboflow(api_key="KebUB2aZYK0JETrujiWA")

# Paths
IMG_DIR = r"C:\Users\EMAAN\Downloads\Night Time Screenshots - Annotated\images"
ANN_DIR = r"C:\Users\EMAAN\Downloads\Night Time Screenshots - Annotated\labels"
IMG_EXT = ".png"
ANN_EXT = ".txt"

LABEL_MAP = {
    0: "Bus",
    1: "Cyclist",
    2: "Light Vehicle",
    3: "Long Truck",
    4: "Medium Truck",
    5: "Minibus Taxi",
    6: "Motorcycle",
    7: "Pedestrian",
    8: "Short Truck"
}

# Get project
project = rf.workspace().project("full-res-labelled")

# Collect valid (image, annotation) pairs
pairs = []
image_paths = glob.glob(os.path.join(IMG_DIR, f"*{IMG_EXT}"))

for image_path in image_paths:
    base = os.path.splitext(os.path.basename(image_path))[0]
    ann_path = os.path.join(ANN_DIR, base + ANN_EXT)
    if os.path.exists(ann_path):
        pairs.append((image_path, ann_path))
    else:
        print(f"Skipping {image_path} — annotation not found")

def upload_one(image_path, ann_path):
    return project.single_upload(
        image_path=image_path,
        annotation_path=ann_path,
        annotation_labelmap=LABEL_MAP,
        tag_names=["night"],
        batch_name="260107_night"
    )

# Parallel upload (32 workers)
MAX_WORKERS = 32

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(upload_one, img, ann): img
        for img, ann in pairs
    }

    for future in as_completed(futures):
        img = futures[future]
        try:
            result = future.result()
            print(f"Uploaded: {img} → {result}")
        except Exception as e:
            print(f"FAILED: {img} → {e}")
