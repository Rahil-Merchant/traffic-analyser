import os
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\models\weights\fixed_cupola_6120_640px_bs64_dynamic_ultralytics.pt"
MODEL_INPUT_SIZE = 640
NMS = False
EXPORT_FP16 = True
EXPORT_INT8 = False
dynamic = True
bs = 64
output_dir = os.path.dirname(MODEL_PATH)
engine_filename = os.path.splitext(os.path.basename(MODEL_PATH))[0] + f"_batch1_sz{MODEL_INPUT_SIZE}_fp16.engine"
OUTPUT_ENGINE_PATH = os.path.join(output_dir, engine_filename)

model = YOLO(MODEL_PATH)
model.export(
            format='engine',
            imgsz=MODEL_INPUT_SIZE,
            # nms=NMS,
            half=EXPORT_FP16,
            dynamic=dynamic,
            batch=bs
        )