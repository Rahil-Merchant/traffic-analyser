# project/models/model_loader.py
import torch
from ultralytics import YOLO
from core.utils import debug_print
import os
from config import MODEL_INPUT_SIZE

def load_model(model_path, device, conf_threshold, iou_threshold):
    """
    Loads the YOLO model, automatically selecting the runtime based on
    the model file extension (.pt or .engine).
    Uses conf/iou thresholds only for .pt models.
    Device placement for .engine models happens implicitly or at inference time.
    """
    debug_print(f"Loading model from: {model_path} onto device: {device}")

    if not os.path.exists(model_path):
        print(f"FATAL ERROR: Model file not found at '{model_path}'")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        if model_path.endswith('.pt'):
            # --- Load Standard PyTorch (.pt) Model ---
            debug_print("Detected .pt file, using standard PyTorch loading.")
            model = YOLO(model_path)
            model.to(device) # Move .pt model to the target device
            # Set thresholds for PyTorch model using values from config
            model.conf = conf_threshold
            model.iou = iou_threshold
            model.verbose = False
            debug_print(f".pt Model loaded. conf={model.conf}, iou={model.iou}")

            # Warm-up for PyTorch models on CUDA
            if device == 'cuda':
                try:
                    debug_print("Warming up .pt model...")
                    # Determine input size from model if possible, fallback otherwise
                    inp_size = getattr(model, 'imgsz', MODEL_INPUT_SIZE)
                    if isinstance(inp_size, list): inp_size = inp_size[0]
                    if not isinstance(inp_size, int): inp_size = MODEL_INPUT_SIZE # Fallback if imgsz attribute is unusual
                    dummy_input = torch.zeros(1, 3, inp_size, inp_size).to(device)
                    model(dummy_input, verbose=False) # Perform a dummy inference
                    torch.cuda.synchronize()
                    debug_print("Model warm-up complete.")
                except Exception as e:
                    debug_print(f"Warning: Model warm-up failed: {e}")
            return model

        elif model_path.endswith('.engine'):
            # --- Load TensorRT (.engine) Model ---
            debug_print("Detected .engine file, using TensorRT runtime via YOLO wrapper.")
            model = YOLO(model_path, task='detect')

            # Engine uses baked-in settings or defaults.
            model.verbose = False
            debug_print(f"TensorRT engine loaded successfully from: {model_path}")
            return model

        else:
            # --- Unsupported Model Type ---
            error_msg = f"Unsupported model file extension: '{os.path.splitext(model_path)[1]}'. Please use .pt or .engine."
            print(f"FATAL ERROR: {error_msg}")
            raise ValueError(error_msg)

    except Exception as e:
        print(f"FATAL ERROR loading model '{model_path}': {e}")
        import traceback
        traceback.print_exc()


def get_device():
    """Determines the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available: Using CPU")
        # Note: TensorRT engines generally require a CUDA-enabled GPU.
    return device

def load_reid_model(reid_config, device):
    """Loads the Re-Identification model."""
    model_path = reid_config.get('model_path')
    model_type = reid_config.get('model_type', 'pytorch') # Default to PyTorch

    print(f"Loading Re-ID model from: {model_path} onto device: {device}")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Re-ID model file not found at '{model_path}'")

    try:
        if model_type == 'pytorch': # Example for a PyTorch model
            # This is highly dependent on how your Re-ID model is saved and structured.
            # Option A: Model is a torch.nn.Module saved with torch.save(model.state_dict(), PATH)
            # You'd need the model class definition here.
            # from project.reid.models import YourReIDNet # Assuming you have this
            # reid_model = YourReIDNet(feature_dim=reid_config.get('feature_dim', 512))
            # reid_model.load_state_dict(torch.load(model_path, map_location=device))
            
            # Option B: Model saved entirely torch.save(model, PATH)
            reid_model = torch.load(model_path, map_location=device)
            
            reid_model.to(device)
            reid_model.eval() # Set to evaluation mode
            print("PyTorch Re-ID model loaded successfully.")
            return reid_model
        # Add elif for other model types (ONNX, TensorRT for Re-ID if you have them)
        else:
            raise ValueError(f"Unsupported Re-ID model type: {model_type}")
    except Exception as e:
        print(f"FATAL ERROR loading Re-ID model '{model_path}': {e}")
        import traceback
        traceback.print_exc()
        raise