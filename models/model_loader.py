## # project/models/model_loader.py

import torch
from ultralytics import YOLO
from core.utils import debug_print
import os
from config import MODEL_INPUT_SIZE, MIXED_PRECISION

# ONNX Runtime imports
try:
    import onnxruntime as ort
    import numpy as np
    ONNX_AVAILABLE = True
    debug_print("ONNX Runtime available for inference")
except ImportError:
    ONNX_AVAILABLE = False
    debug_print("ONNX Runtime not available")

class ONNXWrapper:
    """Wrapper class to make ONNX models compatible with YOLO interface"""
    
    def __init__(self, model_path, device, conf_threshold, iou_threshold):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime-gpu")
        
        self.model_path = model_path
        self.device = device
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.verbose = False
        
        # Set up ONNX Runtime providers based on device
        providers = []
        if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            debug_print("ONNX: Using CUDA provider")
        else:
            providers = ['CPUExecutionProvider']
            debug_print("ONNX: Using CPU provider")
        
        # Create session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load ONNX session
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get model input shape info
        input_shape = self.session.get_inputs()[0].shape
        debug_print(f"ONNX model input shape: {input_shape}")
        debug_print(f"ONNX model output names: {self.output_names}")
        
        # Verify dynamic batch support
        if input_shape[0] == 'batch' or str(input_shape[0]).startswith('unk'):
            debug_print("ONNX: Dynamic batch size confirmed")
        else:
            debug_print(f"ONNX: Fixed batch size detected: {input_shape[0]}")
        
        # Store model dimensions
        self.input_height = MODEL_INPUT_SIZE
        self.input_width = MODEL_INPUT_SIZE
    
    def __call__(self, input_tensor, verbose=False, imgsz=None, augment=False, **kwargs):
        """Main inference method to match YOLO interface"""
        if input_tensor is None:
            return []
        
        # Convert torch tensor to numpy
        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.cpu().numpy()
        else:
            input_np = input_tensor
        
        # Handle mixed precision - use float16 if enabled and model expects it
        target_dtype = np.float16 if MIXED_PRECISION else np.float32
        if input_np.dtype != target_dtype:
            input_np = input_np.astype(target_dtype)
        
        # Run ONNX inference
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_np})
            
            # Convert outputs to format expected by your pipeline
            results = []
            batch_size = input_np.shape[0]
            
            # Process each item in the batch
            for i in range(batch_size):
                # Extract single image results from batch
                single_output = outputs[0][i:i+1]  # Keep batch dimension
                
                # Create result object that matches YOLO format
                result = ONNXResult(single_output, (self.input_height, self.input_width), self.conf)
                results.append(result)
            
            return results
            
        except Exception as e:
            debug_print(f"ONNX inference error: {e}")
            return []
    
    def to(self, device):
        """Compatibility method for device placement"""
        # ONNX device is set during initialization
        return self

class ONNXResult:
    """Result wrapper to match YOLO result format"""
    
    def __init__(self, output_data, orig_shape, conf_threshold):
        self.orig_shape = orig_shape
        self.boxes = ONNXBoxes(output_data, conf_threshold)
    
class ONNXBoxes:
    """Boxes wrapper to match YOLO boxes format"""
    
    def __init__(self, output_data, conf_threshold):
        # output_data shape should be (1, 84, 8400) for YOLOv8
        # Convert to format: [x1, y1, x2, y2, conf, class_id]
        
        if output_data is None or len(output_data.shape) != 3:
            self.data = torch.empty(0, 6)
            return
        
        # Transpose to (1, 8400, 84) for easier processing
        output_transposed = np.transpose(output_data, (0, 2, 1))  # (1, 8400, 84)
        
        # Extract boxes and confidence scores
        boxes_data = output_transposed[0]  # Remove batch dimension: (8400, 84)
        
        # First 4 columns are box coordinates (cx, cy, w, h)
        box_coords = boxes_data[:, :4]  # (8400, 4)
        
        # Remaining columns are class confidences
        class_scores = boxes_data[:, 4:]  # (8400, 80) for 80 classes
        
        # Get max confidence and class index for each detection
        max_conf = np.max(class_scores, axis=1)  # (8400,)
        class_ids = np.argmax(class_scores, axis=1)  # (8400,)
        
        # Filter by confidence threshold from config
        conf_mask = max_conf > conf_threshold
        
        if not np.any(conf_mask):
            self.data = torch.empty(0, 6)
            return
        
        # Apply confidence mask
        filtered_boxes = box_coords[conf_mask]
        filtered_conf = max_conf[conf_mask]
        filtered_classes = class_ids[conf_mask]
        
        # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
        x1 = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
        y1 = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
        x2 = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
        y2 = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
        
        # Stack all data together
        detections = np.column_stack([x1, y1, x2, y2, filtered_conf, filtered_classes])
        
        # Convert to torch tensor
        self.data = torch.from_numpy(detections.astype(np.float32))
    
    def __len__(self):
        """Return number of detections"""
        return len(self.data)
        
def load_model(model_path, device, conf_threshold, iou_threshold):
    """Loads the model, automatically selecting the runtime based on file extension"""
    debug_print(f"Loading model from: {model_path} onto device: {device}")

    if not os.path.exists(model_path):
        print(f"FATAL ERROR: Model file not found at '{model_path}'")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        if model_path.endswith('.pt'):
            # --- Load Standard PyTorch (.pt) Model ---
            debug_print("Detected .pt file, using standard PyTorch loading.")
            model = YOLO(model_path)
            model.to(device)
            model.conf = conf_threshold
            model.iou = iou_threshold
            model.verbose = False
            debug_print(f".pt Model loaded. conf={model.conf}, iou={model.iou}")

            # Warm-up for PyTorch models on CUDA
            if device == 'cuda':
                try:
                    debug_print("Warming up .pt model...")
                    inp_size = getattr(model, 'imgsz', MODEL_INPUT_SIZE)
                    if isinstance(inp_size, list):
                        inp_size = inp_size[0]
                    if not isinstance(inp_size, int):
                        inp_size = MODEL_INPUT_SIZE
                    dummy_input = torch.zeros(1, 3, inp_size, inp_size).to(device)
                    model(dummy_input, verbose=False)
                    torch.cuda.synchronize()
                    debug_print("Model warm-up complete.")
                except Exception as e:
                    debug_print(f"Warning: Model warm-up failed: {e}")
            return model

        elif model_path.endswith('.engine'):
            # --- Load TensorRT (.engine) Model ---
            debug_print("Detected .engine file, using TensorRT runtime via YOLO wrapper.")
            model = YOLO(model_path, task='detect')
            model.verbose = False
            debug_print(f"TensorRT engine loaded successfully from: {model_path}")
            return model
        
        elif model_path.endswith('.onnx'):
            # --- Load ONNX Model ---
            debug_print("Detected .onnx file, using ONNX Runtime.")
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime-gpu")
            
            model = ONNXWrapper(model_path, device, conf_threshold, iou_threshold)
            debug_print(f"ONNX model loaded successfully from: {model_path}")
            debug_print(f"ONNX Model settings: conf={model.conf}, iou={model.iou}")
            
            # Warm-up for ONNX models
            if device == 'cuda':
                try:
                    debug_print("Warming up ONNX model...")
                    # Use mixed precision if enabled
                    dtype = torch.float16 if MIXED_PRECISION else torch.float32
                    dummy_input = torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, dtype=dtype)
                    model(dummy_input, verbose=False)
                    debug_print("ONNX model warm-up complete.")
                except Exception as e:
                    debug_print(f"Warning: ONNX model warm-up failed: {e}")
            
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

def load_reid_model(reid_config, device):
    """Loads the Re-Identification model."""
    model_path = reid_config.get('model_path')
    model_type = reid_config.get('model_type', 'pytorch')

    print(f"Loading Re-ID model from: {model_path} onto device: {device}")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Re-ID model file not found at '{model_path}'")

    try:
        if model_type == 'pytorch':
            reid_model = torch.load(model_path, map_location=device)
            reid_model.to(device)
            reid_model.eval()
            print("PyTorch Re-ID model loaded successfully.")
            return reid_model
        else:
            raise ValueError(f"Unsupported Re-ID model type: {model_type}")
    except Exception as e:
        print(f"FATAL ERROR loading Re-ID model '{model_path}': {e}")
        import traceback
        traceback.print_exc()
        raise