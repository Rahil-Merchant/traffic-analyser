from ultralytics import YOLO

model = YOLO(r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\models\weights\silent_contract_9082_original_bs2048.pt")

model.export(format="onnx",imgsz=960,half=True,dynamic=True,device=0,simplify=True,batch=512)