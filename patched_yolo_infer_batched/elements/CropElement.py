import numpy as np
import cv2 # Keep for resize operations

class CropElement:
    def __init__(
        self,
        source_image: np.ndarray,
        source_image_resized: np.ndarray,
        crop: np.ndarray,
        number_of_crop: int, # Number within its source image
        x_start: int,
        y_start: int,
        source_image_unique_id: any # Can be int, str, etc.
    ) -> None:
        self.source_image = source_image # Original image, never modified by this class after init
        self.source_image_resized = source_image_resized # Image from which self.crop was taken
        self.crop = crop # The actual crop image data
        self.number_of_crop = number_of_crop
        self.x_start = x_start # x_start of self.crop in self.source_image_resized
        self.y_start = y_start # y_start of self.crop in self.source_image_resized
        self.source_image_unique_id = source_image_unique_id

        # YOLO output results (local to self.crop dimensions):
        self.detected_conf: list[float] = [] 
        self.detected_cls: list[int] = []
        self.detected_xyxy: list[list[int]] = [] 
        self.detected_masks: np.ndarray = None # For YOLO-Seg: (N, H_pred, W_pred)
        self.polygons: list[np.ndarray] = None # For YOLO-Seg memory_optimize: List of (M,2) arrays
        
        # Detections projected to a larger canvas, used by CombineDetections.
        # Scale depends on whether resize_results() has been called.
        # Initialized by calculate_real_values() to be on source_image_resized scale.
        # Optionally overwritten by resize_results() to be on source_image scale.
        self.detected_xyxy_real: list[list[int]] = []
        self.detected_masks_real: list[np.ndarray] = [] # List of (H_canvas, W_canvas) binary masks
        self.detected_polygons_real: list[np.ndarray] = [] # List of (M,2) polygon coordinate arrays

    def calculate_inference(self, model, imgsz=640, conf=0.35, iou=0.7, segment=False, classes_list=None, memory_optimize=False, extra_args=None):
        """
        Performs inference on self.crop.
        Used if MakeCropsDetectThem is in 'process_image' mode.
        """
        extra_args = {} if extra_args is None else extra_args
        if self.crop is None or self.crop.size == 0 or self.crop.shape[0] == 0 or self.crop.shape[1] == 0:
            self.detected_xyxy = []
            self.detected_cls = []
            self.detected_conf = []
            self.polygons = []
            self.detected_masks = np.array([]) if not memory_optimize else None
            return

        predictions = model.predict(self.crop, imgsz=imgsz, conf=conf, iou=iou, classes=classes_list, verbose=False, **extra_args)
        
        if not predictions or not hasattr(predictions[0], 'boxes') or predictions[0].boxes is None:
            self.detected_xyxy = []
            self.detected_cls = []
            self.detected_conf = []
            self.polygons = []
            self.detected_masks = np.array([]) if not memory_optimize else None
            return

        pred = predictions[0]
        self.detected_xyxy = pred.boxes.xyxy.cpu().int().tolist()
        self.detected_cls = pred.boxes.cls.cpu().int().tolist()
        self.detected_conf = pred.boxes.conf.cpu().numpy().tolist()

        if segment and len(self.detected_cls) != 0 and hasattr(pred, 'masks') and pred.masks is not None:
            if memory_optimize:
                self.polygons = [mask.astype(np.uint16) for mask in pred.masks.xy]
            else:
                self.detected_masks = pred.masks.data.cpu().numpy().astype(np.uint8)
        else:
            self.polygons = []
            self.detected_masks = np.array([]) if not memory_optimize else None


    def calculate_real_values(self):
        """
        Projects crop-local detections to the self.source_image_resized coordinate system.
        Populates self.detected_xyxy_real, self.detected_masks_real, self.detected_polygons_real.
        """
        x_start_global = self.x_start
        y_start_global = self.y_start

        self.detected_xyxy_real = []
        if self.detected_xyxy:
            for bbox in self.detected_xyxy:
                x_min, y_min, x_max, y_max = bbox
                self.detected_xyxy_real.append([
                    x_min + x_start_global, y_min + y_start_global,
                    x_max + x_start_global, y_max + y_start_global
                ])
        
        self.detected_masks_real = []
        if self.detected_masks is not None and len(self.detected_masks) > 0:
            for mask_crop_resolution in self.detected_masks:
                if self.crop.shape[0] > 0 and self.crop.shape[1] > 0 and \
                   mask_crop_resolution is not None and mask_crop_resolution.shape[0] > 0 and mask_crop_resolution.shape[1] > 0:
                    
                    full_mask_on_resized = np.zeros((self.source_image_resized.shape[0], self.source_image_resized.shape[1]), dtype=np.uint8)
                    mask_resized_to_crop_dims = cv2.resize(
                        mask_crop_resolution, 
                        (self.crop.shape[1], self.crop.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    y_end_local = self.crop.shape[0]
                    x_end_local = self.crop.shape[1]
                    
                    y_start_abs = y_start_global
                    y_end_abs = min(y_start_global + y_end_local, self.source_image_resized.shape[0])
                    x_start_abs = x_start_global
                    x_end_abs = min(x_start_global + x_end_local, self.source_image_resized.shape[1])

                    # Ensure slices are valid
                    if y_end_abs > y_start_abs and x_end_abs > x_start_abs:
                         full_mask_on_resized[y_start_abs:y_end_abs, x_start_abs:x_end_abs] = \
                            mask_resized_to_crop_dims[:(y_end_abs-y_start_abs), :(x_end_abs-x_start_abs)]
                    self.detected_masks_real.append(full_mask_on_resized)

        self.detected_polygons_real = []
        if self.polygons is not None and len(self.polygons) > 0:
            for poly_crop_local in self.polygons:
                if poly_crop_local is not None and len(poly_crop_local) > 0:
                    poly_on_resized = poly_crop_local.astype(np.int32).copy()
                    poly_on_resized[:, 0] += x_start_global
                    poly_on_resized[:, 1] += y_start_global
                    self.detected_polygons_real.append(poly_on_resized.astype(np.uint16))


    def resize_results(self):
        """
        Resizes detections from self.source_image_resized scale to self.source_image (original) scale.
        Overwrites self.detected_xyxy_real, self.detected_masks_real, self.detected_polygons_real.
        """
        has_xyxy = self.detected_xyxy_real and len(self.detected_xyxy_real) > 0
        has_masks = self.detected_masks_real and len(self.detected_masks_real) > 0
        has_polygons = self.detected_polygons_real and len(self.detected_polygons_real) > 0

        if not (has_xyxy or has_masks or has_polygons): return
        if self.source_image.shape == self.source_image_resized.shape: return 

        scale_x = self.source_image.shape[1] / self.source_image_resized.shape[1]
        scale_y = self.source_image.shape[0] / self.source_image_resized.shape[0]

        if has_xyxy:
            resized_xyxy_list = []
            for bbox in self.detected_xyxy_real:
                x_min, y_min, x_max, y_max = bbox
                resized_xyxy_list.append([
                    int(x_min * scale_x), int(y_min * scale_y),
                    int(x_max * scale_x), int(y_max * scale_y)
                ])
            self.detected_xyxy_real = resized_xyxy_list

        if has_masks:
            resized_masks_list = []
            for mask_on_resized in self.detected_masks_real:
                if mask_on_resized.shape[0] > 0 and mask_on_resized.shape[1] > 0:
                    mask_on_orig = cv2.resize(mask_on_resized, 
                                              (self.source_image.shape[1], self.source_image.shape[0]), 
                                              interpolation=cv2.INTER_NEAREST)
                    resized_masks_list.append(mask_on_orig)
            self.detected_masks_real = resized_masks_list

        if has_polygons:
            resized_polygons_list = []
            for poly_on_resized in self.detected_polygons_real:
                if poly_on_resized is not None and len(poly_on_resized) > 0:
                    poly_on_orig = poly_on_resized.astype(np.float32).copy()
                    poly_on_orig[:, 0] *= scale_x
                    poly_on_orig[:, 1] *= scale_y
                    resized_polygons_list.append(poly_on_orig.astype(np.uint16))
            self.detected_polygons_real = resized_polygons_list