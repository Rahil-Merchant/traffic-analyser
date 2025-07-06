# project/models/model_loader.py

import torch
from ultralytics import YOLO
import os
from config import MODEL_INPUT_SIZE
from loguru import logger
     
def load_model(model_path, device, conf_threshold, iou_threshold):
    """Loads the model, automatically selecting the runtime based on file extension"""
    logger.debug(f"Loading model from: {model_path} onto device: {device}")

    if not os.path.exists(model_path):
        print(f"FATAL ERROR: Model file not found at '{model_path}'")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        if model_path.endswith('.pt'):
            # --- Load Standard PyTorch (.pt) Model ---
            logger.debug("Detected .pt file, using standard PyTorch loading.")
            model = YOLO(model_path)
            model.to(device)
            model.conf = conf_threshold
            model.iou = iou_threshold
            model.verbose = False
            logger.debug(f".pt Model loaded. conf={model.conf}, iou={model.iou}")

            # Warm-up for PyTorch models on CUDA
            if device == 'cuda':
                try:
                    logger.debug("Warming up .pt model...")
                    inp_size = getattr(model, 'imgsz', MODEL_INPUT_SIZE)
                    if isinstance(inp_size, list):
                        inp_size = inp_size[0]
                    if not isinstance(inp_size, int):
                        inp_size = MODEL_INPUT_SIZE
                    dummy_input = torch.zeros(1, 3, inp_size, inp_size).to(device)
                    model(dummy_input, verbose=False)
                    torch.cuda.synchronize()
                    logger.debug("Model warm-up complete.")
                except Exception as e:
                    logger.debug(f"Warning: Model warm-up failed: {e}")
            return model

        elif model_path.endswith('.engine'):
            # --- Load TensorRT (.engine) Model ---
            logger.debug("Detected .engine file, using TensorRT runtime via YOLO wrapper.")
            model = YOLO(model_path, task='detect')
            model.verbose = False
            logger.debug(f"TensorRT engine loaded successfully from: {model_path}")
            return model
        
        else:
            # --- Unsupported Model Type ---
            error_msg = f"Unsupported model file extension: '{os.path.splitext(model_path)[1]}'. Please use .pt, .engine, or .onnx."
            print(f"FATAL ERROR: {error_msg}")
            raise ValueError(error_msg)

    except Exception as e:
        print(f"FATAL ERROR loading model '{model_path}': {e}")
        import traceback
        traceback.print_exc()
        raise

def get_device():
    """Determines the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available: Using CPU")
    return device