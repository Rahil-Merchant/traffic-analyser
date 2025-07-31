from ultralytics import YOLO

MODEL_PATH = "models/weights/20072025/few_anchowy_linux_fp32.pt"
MODEL_INPUT_SIZE = 640
EXPORT_FP16 = False
EXPORT_INT8 = {'export':False,'yaml_path':r'C:\Users\swest\Desktop\traffic-capturing\training\datasets\relabelled12072025\data.yaml'}
dynamic = True
bs = 64

model = YOLO(MODEL_PATH)
model.export(
            format='engine',
            imgsz=MODEL_INPUT_SIZE,
            half=EXPORT_FP16,
            int8=EXPORT_INT8['export'],
            data=EXPORT_INT8['yaml_path'],
            dynamic=dynamic,
            simplify=True,
            batch=bs
        ) 