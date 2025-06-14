# Filename: export_to_tensorrt.py

import os
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\models\weights\silent_contract_9082.pt"
MODEL_INPUT_SIZE = 640
NMS = False
EXPORT_FP16 = True
EXPORT_INT8 = False
output_dir = os.path.dirname(MODEL_PATH)
engine_filename = os.path.splitext(os.path.basename(MODEL_PATH))[0] + f"_batch1_sz{MODEL_INPUT_SIZE}_fp16.engine"
OUTPUT_ENGINE_PATH = os.path.join(output_dir, engine_filename)

CALIBRATION_DATA_YAML = r"C:/Users/EMAAN/Documents/YOLO/20250411 Dataset/dataset_split/data.yaml"

def main():
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n--- Starting TensorRT Export ---")
    print(f"Input Size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
    print(f"Exporting FP16: {EXPORT_FP16}")
    print(f"Exporting INT8: {EXPORT_INT8}")
    if EXPORT_INT8:
        print(f"Calibration Data (for INT8): {CALIBRATION_DATA_YAML}")
    print(f"Output Engine Path: {OUTPUT_ENGINE_PATH}")

    try:
        model.export(
            format='engine',
            imgsz=MODEL_INPUT_SIZE,
            nms=NMS,
            half=EXPORT_FP16,
            int8=EXPORT_INT8,
            data=CALIBRATION_DATA_YAML if EXPORT_INT8 else None, # Calibration data for INT8
            dynamic=True,
            batch=64
        )
        print("\n--- Export Successful ---")
        print(f"Model stored at: {OUTPUT_ENGINE_PATH}")

    except Exception as e:
        print(f"\n--- Export Failed ---")
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()