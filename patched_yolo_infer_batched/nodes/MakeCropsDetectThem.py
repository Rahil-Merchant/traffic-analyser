from collections import Counter
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from ..elements.CropElement import CropElement


class MakeCropsDetectThem:
    """
    Class implementing cropping and (optionally) passing crops through a neural network
    for detection/segmentation. Can also be used purely as a crop generator.
    """
    def __init__(
        self,
        image: np.ndarray = None,
        model_path="yolov8n.pt",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        classes_list=None,
        segment=False,
        shape_x=700,
        shape_y=600,
        overlap_x=25,
        overlap_y=25,
        show_crops=False,
        show_processing_status=False,
        resize_initial_size=True,
        model=None,
        memory_optimize=True,
        inference_extra_args=None,
        batch_inference=False, # Internal batching for the instance's image
        progress_callback=None,
        mode: str = "process_image" # "process_image" or "generate_crops_only"
    ) -> None:

        self.show_process_status = show_processing_status
        self._progress_bars = {}
        
        if progress_callback is not None and show_processing_status:
            self.progress_callback = progress_callback
        elif show_processing_status:
            self.progress_callback = self._tqdm_callback
        else:
            self.progress_callback = None
            
        if model is None:
            self.model = YOLO(model_path)
        else:
            self.model = model

        self.image_ref_for_instance = image
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes_list = classes_list
        self.segment = segment
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.show_crops = show_crops
        self.resize_initial_size = resize_initial_size # This flag is stored in CropElements via factory
        self.memory_optimize = memory_optimize
        self.class_names_dict = self.model.names
        self.inference_extra_args = inference_extra_args if inference_extra_args else {}
        self.internal_batch_inference_flag = batch_inference
        self.mode = mode

        self.crops_elements: list[CropElement] = []
        self.crop_image_arrays: list[np.ndarray] = []

        if self.mode == "process_image":
            if self.image_ref_for_instance is None:
                raise ValueError("Image must be provided when mode is 'process_image'")
            
            _source_id = id(self.image_ref_for_instance)

            result = self.get_crops_xy(
                self.image_ref_for_instance,
                shape_x=self.shape_x, shape_y=self.shape_y,
                overlap_x=self.overlap_x, overlap_y=self.overlap_y,
                show=self.show_crops,
                force_return_image_arrays=True,
                source_image_unique_id_for_elements=_source_id
            )
            self.crops_elements, self.crop_image_arrays = result

            if self.internal_batch_inference_flag:
                self._detect_objects_batch_internal()
            else:
                self._detect_objects_sequential_internal()

        elif self.mode == "generate_crops_only":
            if self.image_ref_for_instance is not None:
                _source_id = id(self.image_ref_for_instance)
                result = self.get_crops_xy(
                    self.image_ref_for_instance,
                    shape_x=self.shape_x, shape_y=self.shape_y,
                    overlap_x=self.overlap_x, overlap_y=self.overlap_y,
                    show=self.show_crops,
                    force_return_image_arrays=True,
                    source_image_unique_id_for_elements=_source_id
                )
                self.crops_elements, self.crop_image_arrays = result
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _tqdm_callback(self, task, current, total):
        if total <=0: return
        if task not in self._progress_bars:
            self._progress_bars[task] = tqdm(total=total, desc=task, unit='items', leave=False) # leave=False for better nesting
        self._progress_bars[task].n = current
        self._progress_bars[task].refresh()
        if current >= total:
            self._progress_bars[task].close()
            if task in self._progress_bars:
                 del self._progress_bars[task]

    def get_crops_xy(
        self, image_full: np.ndarray, shape_x: int, shape_y: int,
        overlap_x: int = 25, overlap_y: int = 25, show: bool = False,
        force_return_image_arrays: bool = False,
        source_image_unique_id_for_elements: any = None
    ) -> tuple[list[CropElement], list[np.ndarray]]:
        
        if image_full is None: return [], []

        cross_koef_x = 1 - (overlap_x / 100) if overlap_x < 100 else 0.01
        cross_koef_y = 1 - (overlap_y / 100) if overlap_y < 100 else 0.01

        crop_elements_list = []
        crop_image_arrays_list = []

        orig_h, orig_w = image_full.shape[:2]
        eff_shape_y = min(shape_y, orig_h)
        eff_shape_x = min(shape_x, orig_w)

        if eff_shape_y <= 0 or eff_shape_x <= 0 : return [],[]

        stride_y = eff_shape_y * cross_koef_y if eff_shape_y * cross_koef_y > 0 else 1.0
        stride_x = eff_shape_x * cross_koef_x if eff_shape_x * cross_koef_x > 0 else 1.0

        y_steps = int((orig_h - eff_shape_y) / stride_y) + 1 if orig_h > eff_shape_y else 1
        x_steps = int((orig_w - eff_shape_x) / stride_x) + 1 if orig_w > eff_shape_x else 1
        
        y_new_dim = round((y_steps - 1) * stride_y + eff_shape_y)
        x_new_dim = round((x_steps - 1) * stride_x + eff_shape_x)
        
        y_new_dim = max(orig_h, y_new_dim) if orig_h <= eff_shape_y else y_new_dim
        x_new_dim = max(orig_w, x_new_dim) if orig_w <= eff_shape_x else x_new_dim
        y_new_dim = max(1, int(y_new_dim))
        x_new_dim = max(1, int(x_new_dim))

        image_initial_copy = image_full.copy()
        if orig_h == y_new_dim and orig_w == x_new_dim:
            image_resized_for_cropping = image_full.copy() # No resize needed
        else:
            try:
                image_resized_for_cropping = cv2.resize(image_full, (x_new_dim, y_new_dim))
            except cv2.error:
                image_resized_for_cropping = image_full.copy()


        if show: plt.figure(figsize=[max(1,x_steps) * 0.9, max(1,y_steps) * 0.9])

        count = 0
        total_steps = y_steps * x_steps
        for i in range(y_steps):
            for j in range(x_steps):
                y_start = int(i * stride_y)
                x_start = int(j * stride_x)

                current_crop_h = min(eff_shape_y, image_resized_for_cropping.shape[0] - y_start)
                current_crop_w = min(eff_shape_x, image_resized_for_cropping.shape[1] - x_start)

                if current_crop_w <=0 or current_crop_h <=0: continue

                im_temp = image_resized_for_cropping[y_start : y_start + current_crop_h, x_start : x_start + current_crop_w]
                if im_temp.size == 0: continue

                if show:
                    plt.subplot(y_steps, x_steps, i * x_steps + j + 1)
                    plt.imshow(cv2.cvtColor(im_temp.copy(), cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                count += 1
                
                if self.progress_callback: self.progress_callback(f"Getting crops (ImgID: {source_image_unique_id_for_elements})", count, total_steps)

                crop_elements_list.append(CropElement(
                    source_image=image_initial_copy, # Store original, unresized
                    source_image_resized=image_resized_for_cropping.copy(), # Store the version crops were cut from
                    crop=im_temp.copy(), # Store a copy of the patch itself
                    number_of_crop=count,
                    x_start=x_start, y_start=y_start,
                    source_image_unique_id=source_image_unique_id_for_elements
                ))
                if force_return_image_arrays or self.internal_batch_inference_flag :
                     crop_image_arrays_list.append(im_temp)

        if show:
            plt.show()

        return crop_elements_list, crop_image_arrays_list

    def generate_crops_for_external_processing(
        self,
        image_to_crop: np.ndarray,
        source_id: any,
        shape_x_override=None,
        shape_y_override=None,
        overlap_x_override=None,
        overlap_y_override=None,
        show_crops_override=None
    ) -> tuple[list[CropElement], list[np.ndarray]]:
        
        current_shape_x = shape_x_override if shape_x_override is not None else self.shape_x
        current_shape_y = shape_y_override if shape_y_override is not None else self.shape_y
        current_overlap_x = overlap_x_override if overlap_x_override is not None else self.overlap_x
        current_overlap_y = overlap_y_override if overlap_y_override is not None else self.overlap_y
        current_show_crops = show_crops_override if show_crops_override is not None else self.show_crops
        
        original_progress_callback = self.progress_callback
        is_internal_tqdm = (self.progress_callback == self._tqdm_callback)
        if is_internal_tqdm and self.show_process_status: 
             self.progress_callback = None 

        elements, arrays = self.get_crops_xy(
            image_to_crop,
            shape_x=current_shape_x,
            shape_y=current_shape_y,
            overlap_x=current_overlap_x,
            overlap_y=current_overlap_y,
            show=current_show_crops,
            force_return_image_arrays=True,
            source_image_unique_id_for_elements=source_id
        )
        
        if is_internal_tqdm and self.show_process_status:
            self.progress_callback = original_progress_callback
        
        return elements, arrays

    def _detect_objects_sequential_internal(self):
        total_crops = len(self.crops_elements)
        if total_crops == 0: return

        for index, crop_element in enumerate(self.crops_elements):
            crop_element.calculate_inference(
                self.model, self.imgsz, self.conf, self.iou,
                self.segment, self.classes_list, self.memory_optimize, self.inference_extra_args
            )
            crop_element.calculate_real_values()
            if self.resize_initial_size: crop_element.resize_results()
            if self.progress_callback: self.progress_callback("Detecting objects (sequential)", (index + 1), total_crops)

    def _detect_objects_batch_internal(self):
        if not self.crop_image_arrays: return

        if self.progress_callback: self.progress_callback("Detecting objects (batch_internal)", 0, 1)
        
        predictions = MakeCropsDetectThem._calculate_batch_inference_static(
            self.crop_image_arrays, self.model, self.imgsz, self.conf, self.iou,
            self.classes_list, self.inference_extra_args
        )

        for pred_result, crop_element in zip(predictions, self.crops_elements):
            MakeCropsDetectThem.populate_crop_element_with_prediction(
                crop_element, pred_result, self.segment, self.memory_optimize
            )
            crop_element.calculate_real_values()
            if self.resize_initial_size: crop_element.resize_results()
        
        if self.progress_callback: self.progress_callback("Detecting objects (batch_internal)", 1, 1)

    @staticmethod
    def _calculate_batch_inference_static(
        batch_crop_arrays: list[np.ndarray], model: YOLO, imgsz: int, conf: float, iou: float,
        classes_list: list = None, extra_args: dict = None
    ) -> list: 
        if not batch_crop_arrays: return []
        
        valid_batch_arrays = []
        original_indices = [] # To map results back if some arrays are invalid
        for idx, arr in enumerate(batch_crop_arrays):
            if arr is not None and arr.size > 0 and arr.shape[0]>0 and arr.shape[1]>0 and arr.ndim == 3 and arr.shape[2] == 3:
                valid_batch_arrays.append(arr)
                original_indices.append(idx)
            # else:
                # print(f"Warning: Invalid array found in batch for inference. Shape: {arr.shape if arr is not None else 'None'}. Skipping.")

        if not valid_batch_arrays: 
            # Return a list of Nones matching the original batch_crop_arrays length
            return [None] * len(batch_crop_arrays)

        extra_args = {} if extra_args is None else extra_args
        try:
            raw_predictions = model.predict(
                valid_batch_arrays, imgsz=imgsz, conf=conf, iou=iou,
                classes=classes_list, verbose=False, **extra_args
            )
            # Reconstruct the full list of predictions, inserting Nones for invalid inputs
            full_predictions_list = [None] * len(batch_crop_arrays)
            for i, raw_pred_idx in enumerate(original_indices):
                full_predictions_list[raw_pred_idx] = raw_predictions[i]
            return full_predictions_list
            
        except Exception as e:
            # print(f"Error during YOLO predict: {e}")
            # print(f"Problematic batch (first 5 valid shapes): {[arr.shape for arr in valid_batch_arrays[:5]]}")
            # For robustness, return a list of Nones if prediction fails catastrophically
            return [None] * len(batch_crop_arrays)


    @staticmethod
    def populate_crop_element_with_prediction(
        crop_element: CropElement, prediction_result, 
        segment_mode: bool, memory_optimize_mode: bool
    ):
        if prediction_result is None or not hasattr(prediction_result, 'boxes') or prediction_result.boxes is None:
            crop_element.detected_xyxy = []
            crop_element.detected_cls = []
            crop_element.detected_conf = []
            crop_element.polygons = []
            crop_element.detected_masks = np.array([]) if not memory_optimize_mode else None
            return

        crop_element.detected_xyxy = prediction_result.boxes.xyxy.cpu().int().tolist()
        crop_element.detected_cls = prediction_result.boxes.cls.cpu().int().tolist()
        crop_element.detected_conf = prediction_result.boxes.conf.cpu().numpy().tolist()

        if segment_mode and len(crop_element.detected_cls) != 0 and hasattr(prediction_result, 'masks') and prediction_result.masks is not None:
            if memory_optimize_mode:
                crop_element.polygons = [mask.astype(np.uint16) for mask in prediction_result.masks.xy]
            else:
                crop_element.detected_masks = prediction_result.masks.data.cpu().numpy().astype(np.uint8)
        else:
            crop_element.polygons = []
            crop_element.detected_masks = np.array([]) if not memory_optimize_mode else None

    def __str__(self):
        image_shape_str = str(self.image_ref_for_instance.shape) if self.image_ref_for_instance is not None else 'N/A (factory mode)'
        return (
            f"MakeCropsDetectThem instance: {len(self.crops_elements)} crops generated for current instance image. "
            f"Mode: {self.mode}. "
            f"Instance image shape: {image_shape_str}."
        )

    def patches_info(self):
        if self.mode != "process_image" or not self.crops_elements :
            return "Patches info is available only when mode='process_image' and detections are run for the instance's image."
        
        output_str = str(self) # General info
        output_str += "\nDetailed information about the detections for each patch (for instance's image):"
        has_any_detections = False
        for i, patch in enumerate(self.crops_elements):
            if patch.detected_cls and len(patch.detected_cls) > 0:
                has_any_detections = True
                detected_cls_names_list = [
                    self.class_names_dict.get(int(value), f"ID_{int(value)}") for value in patch.detected_cls
                ] 
                class_counts = Counter(detected_cls_names_list)
                class_info = ", ".join([f"{count} {cls}" for cls, count in class_counts.items()])
                output_str += f"\nOn patch № {i} (ImgID {patch.source_image_unique_id}), {class_info} were detected"
            else:
                output_str += f"\nOn patch № {i} (ImgID {patch.source_image_unique_id}), nothing was detected"
        
        if not self.crops_elements:
             output_str += "\nNo patches were generated for this instance."
        elif not has_any_detections:
             output_str += "\nNo detections found in any patch for this instance."
        return output_str

    def __del__(self):
        for pbar_name in list(self._progress_bars.keys()):
            if self._progress_bars[pbar_name].n < self._progress_bars[pbar_name].total:
                 self._progress_bars[pbar_name].update(self._progress_bars[pbar_name].total - self._progress_bars[pbar_name].n)
            self._progress_bars[pbar_name].close()
        self._progress_bars.clear()