from typing import Union, List, Any
import torch
import numpy as np

# Forward declaration for CropElement and MakeCropsDetectThem
# These are for type hints and will be resolved if the actual classes are imported/defined.
if 'CropElement' not in globals():
    class CropElement: pass 
if 'MakeCropsDetectThem' not in globals():
    class MakeCropsDetectThem: pass


class MCDTLike: # Wrapper for CombineDetections
    """
    A simple container to provide CombineDetections with the necessary attributes
    when not using a full MakeCropsDetectThem instance directly.
    """
    def __init__(self, crops_list: List['CropElement'], 
                 class_names: dict, 
                 segment_flag: bool, 
                 memory_optimize_flag: bool, 
                 resize_initial_size_flag: bool, # Indicates scale of coords in CropElement.detected_*_real
                 image_ref: np.ndarray): # The image corresponding to the coordinate scale
        self.crops = crops_list # List of CropElement objects for a single original image
        self.class_names_dict = class_names
        self.segment = segment_flag
        self.memory_optimize = memory_optimize_flag
        self.resize_initial_size = resize_initial_size_flag 
        self.image = image_ref # Reference image for NMS context (original or resized_for_cropping)


class CombineDetections:
    """
    Class implementing combining masks/boxes from multiple crops + NMS (Non-Maximum Suppression)
    for a single original image's context.
    """
    def __init__(
        self,
        element_crops: Union['MakeCropsDetectThem', List['MakeCropsDetectThem'], MCDTLike],
        nms_threshold=0.3,
        match_metric='IOS',
        intelligent_sorter=True,
        sorter_bins=5,
        class_agnostic_nms=True
    ) -> None:

        self.nms_threshold = nms_threshold
        self.match_metric = match_metric
        self.intelligent_sorter = intelligent_sorter
        self.sorter_bins = sorter_bins
        self.class_agnostic_nms = class_agnostic_nms

        if not hasattr(element_crops, 'crops') or \
           not hasattr(element_crops, 'class_names_dict') or \
           not hasattr(element_crops, 'segment') or \
           not hasattr(element_crops, 'memory_optimize') or \
           not hasattr(element_crops, 'image'):
            raise TypeError(f"Input 'element_crops' (type: {type(element_crops)}) is not compatible. Missing required attributes.")

        self.class_names = element_crops.class_names_dict
        self.crops_for_nms = element_crops.crops 
        self.image_for_nms_context = element_crops.image 
        self.segment = element_crops.segment
        self.memory_optimize = element_crops.memory_optimize
        
        self.filtered_confidences: list[float] = []
        self.filtered_boxes: list[list[int]] = []
        self.filtered_classes_id: list[int] = []
        self.filtered_classes_names: list[str] = []
        self.filtered_masks: list[np.ndarray] = []
        self.filtered_polygons: list[np.ndarray] = []
        self.filtered_indices: list[int] = []
        self.detected_conf_list_full: list[float] = []
        self.detected_xyxy_list_full: list[list[int]] = []
        self.detected_masks_list_full: list[np.ndarray] = []
        self.detected_cls_id_list_full: list[int] = []
        self.detected_polygons_list_full: list[np.ndarray] = []
        self.detected_cls_names_list_full: list[str] = []


        if not self.crops_for_nms: return

        (
            self.detected_conf_list_full, self.detected_xyxy_list_full,
            self.detected_masks_list_full, self.detected_cls_id_list_full,
            self.detected_polygons_list_full
        ) = self.combinate_detections(crops=self.crops_for_nms)

        if not self.detected_cls_id_list_full: return
        
        self.detected_cls_names_list_full = [
            self.class_names.get(int(value), f"ID_{int(value)}") for value in self.detected_cls_id_list_full
        ]

        conf_tensor = torch.tensor(self.detected_conf_list_full, dtype=torch.float32) if self.detected_conf_list_full else torch.empty(0, dtype=torch.float32)
        boxes_tensor = torch.tensor(self.detected_xyxy_list_full, dtype=torch.float32) if self.detected_xyxy_list_full else torch.empty(0, 4, dtype=torch.float32)
        cls_id_tensor = torch.tensor(self.detected_cls_id_list_full, dtype=torch.long) if self.detected_cls_id_list_full else torch.empty(0, dtype=torch.long)

        if boxes_tensor.numel() == 0: # No boxes to process
            return

        if self.class_agnostic_nms:
            self.filtered_indices = self.nms(
                conf_tensor, boxes_tensor, self.match_metric, self.nms_threshold,
                self.detected_masks_list_full, self.intelligent_sorter
            )
        else:
            self.filtered_indices = self.not_agnostic_nms(
                cls_id_tensor, conf_tensor, boxes_tensor, self.match_metric, self.nms_threshold,
                self.detected_masks_list_full, self.intelligent_sorter
            )

        self.filtered_confidences = [self.detected_conf_list_full[i] for i in self.filtered_indices]
        self.filtered_boxes = [self.detected_xyxy_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_id = [self.detected_cls_id_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_names = [self.detected_cls_names_list_full[i] for i in self.filtered_indices]

        if self.segment and not self.memory_optimize and self.detected_masks_list_full:
            self.filtered_masks = [self.detected_masks_list_full[i] for i in self.filtered_indices if i < len(self.detected_masks_list_full)]
        
        if self.segment and self.memory_optimize and self.detected_polygons_list_full:
            self.filtered_polygons = [self.detected_polygons_list_full[i] for i in self.filtered_indices if i < len(self.detected_polygons_list_full)]


    def combinate_detections(self, crops: List['CropElement']):
        d_conf, d_xyxy, d_masks, d_cls, d_poly = [], [], [], [], []
        for ce in crops:
            if ce.detected_conf: d_conf.extend(ce.detected_conf)
            if ce.detected_xyxy_real: d_xyxy.extend(ce.detected_xyxy_real)
            if ce.detected_masks_real: d_masks.extend(ce.detected_masks_real)
            if ce.detected_cls: d_cls.extend(ce.detected_cls)
            if ce.detected_polygons_real: d_poly.extend(ce.detected_polygons_real)
        return d_conf, d_xyxy, d_masks, d_cls, d_poly

    @staticmethod
    def average_to_bound(confidences: Union[float, list, np.ndarray], N: int = 10) -> Union[float, list]:
        is_scalar = isinstance(confidences, (float, np.floating)) # np.floating for numpy scalars
        if is_scalar:
            conf_array = np.array([confidences], dtype=float) # Ensure it's float for np.digitize
        elif isinstance(confidences, list):
            conf_array = np.array(confidences, dtype=float)
        elif isinstance(confidences, np.ndarray):
            conf_array = confidences.astype(float)
        else:
            raise TypeError("Confidences must be float, list of floats, or numpy array of floats.")
            
        step = 1.0 / N
        # Ensure bounds cover [0, 1] inclusive for np.digitize (it bins x_i such that bins[i-1] <= x_i < bins[i])
        # To make it inclusive of 1 at the top bin, add a small epsilon or ensure last bound is > 1
        bounds = np.arange(0, 1.0 + step, step) 
        if bounds[-1] < 1.0: # Ensure last bound is at least 1.0
             bounds = np.append(bounds, 1.0 + step/2)


        # np.digitize returns indices from 1 to len(bins)
        # We want indices from 0 to N-1 (for N bins, where bin k is [bounds[k], bounds[k+1]))
        indices = np.digitize(conf_array, bounds, right=False) -1
        indices = np.clip(indices, 0, N-1) # Clip to valid bin indices for bounds[indices]
        
        # We want the left boundary of the bin
        averaged_confidences = np.round(bounds[indices], 2)
        return averaged_confidences[0] if is_scalar else averaged_confidences.tolist()

    @staticmethod
    def intersect_over_union(mask1: np.ndarray, masks_list: List[np.ndarray]) -> torch.Tensor:
        iou_scores = []
        if mask1 is None or mask1.size == 0:
            return torch.tensor([0.0] * len(masks_list), dtype=torch.float32)
            
        for mask2 in masks_list:
            if mask2 is None or mask2.size == 0:
                iou_scores.append(0.0)
                continue
            try:
                # Ensure masks are boolean for logical operations
                mask1_bool = mask1.astype(bool)
                mask2_bool = mask2.astype(bool)
                intersection = np.logical_and(mask1_bool, mask2_bool).sum()
                union = np.logical_or(mask1_bool, mask2_bool).sum()
                iou = float(intersection) / float(union) if union != 0 else 0.0
            except Exception: 
                iou = 0.0
            iou_scores.append(iou)
        return torch.tensor(iou_scores, dtype=torch.float32)

    @staticmethod
    def intersect_over_smaller(mask1: np.ndarray, masks_list: List[np.ndarray]) -> torch.Tensor:
        ios_scores = []
        if mask1 is None or mask1.size == 0:
             return torch.tensor([0.0] * len(masks_list), dtype=torch.float32)

        for mask2 in masks_list:
            if mask2 is None or mask2.size == 0:
                ios_scores.append(0.0)
                continue
            try:
                mask1_bool = mask1.astype(bool)
                mask2_bool = mask2.astype(bool)
                intersection = np.logical_and(mask1_bool, mask2_bool).sum()
                area1 = mask1_bool.sum()
                area2 = mask2_bool.sum()
                smaller_area = min(area1, area2)
                ios = float(intersection) / float(smaller_area) if smaller_area != 0 else 0.0
            except Exception:
                 ios = 0.0
            ios_scores.append(ios)
        return torch.tensor(ios_scores, dtype=torch.float32)

    def nms(
        self,
        confidences: torch.Tensor,
        boxes: torch.Tensor,
        match_metric: str,
        nms_threshold: float,
        masks: List[np.ndarray] = None,
        intelligent_sorter: bool = False,
        cls_indexes_for_mapping: torch.Tensor = None 
    ) -> List[int]:
        if masks is None: masks = []
        if len(boxes) == 0: return []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        if intelligent_sorter:
            order = sorted(
                range(len(confidences)),
                key=lambda k: (
                    self.average_to_bound(confidences[k].item(), self.sorter_bins),
                    areas[k].item(),
                ) 
            ) 
        else:
            order = confidences.argsort().tolist()

        keep = []
        while order:
            i = order.pop() 
            keep.append(i)
            if not order: break

            remaining_indices_tensor = torch.tensor(order, dtype=torch.long)
            if remaining_indices_tensor.numel() == 0: break # No more items to compare
            
            xx1 = torch.maximum(x1[i], x1[remaining_indices_tensor])
            yy1 = torch.maximum(y1[i], y1[remaining_indices_tensor])
            xx2 = torch.minimum(x2[i], x2[remaining_indices_tensor])
            yy2 = torch.minimum(y2[i], y2[remaining_indices_tensor])

            w = torch.clamp(xx2 - xx1, min=0.0)
            h = torch.clamp(yy2 - yy1, min=0.0)
            inter = w * h

            rem_areas = areas[remaining_indices_tensor]
            current_area = areas[i]

            if match_metric == "IOU":
                union = (rem_areas - inter) + current_area
                metric_values = inter / (union + 1e-6)
            elif match_metric == "IOS":
                smaller = torch.minimum(rem_areas, current_area)
                metric_values = inter / (smaller + 1e-6)
            else:
                raise ValueError("Unknown matching metric")

            indices_to_remove_from_order = set()
            suppress_by_box_metric_mask = metric_values >= nms_threshold
            
            for idx_in_remaining, suppress_flag in enumerate(suppress_by_box_metric_mask):
                if suppress_flag:
                    indices_to_remove_from_order.add(order[idx_in_remaining])

            if masks and len(masks) > 0 and i < len(masks) and masks[i] is not None:
                mask_candidates_indices_in_order = []
                masks_to_compare_list = []
                metric_values_indices_for_mask_check = []


                for k_idx, original_order_idx in enumerate(order):
                    if original_order_idx not in indices_to_remove_from_order and metric_values[k_idx] > 0.001 : 
                         if original_order_idx < len(masks) and masks[original_order_idx] is not None:
                            mask_candidates_indices_in_order.append(original_order_idx)
                            masks_to_compare_list.append(masks[original_order_idx])
                            metric_values_indices_for_mask_check.append(k_idx) # Keep track of corresponding metric_value
                
                if masks_to_compare_list:
                    if match_metric == "IOU":
                        mask_metric_scores = self.intersect_over_union(masks[i], masks_to_compare_list)
                    else: # IOS
                        mask_metric_scores = self.intersect_over_smaller(masks[i], masks_to_compare_list)
                    
                    suppress_by_mask_metric_mask = mask_metric_scores >= nms_threshold
                    for idx_in_candidates, suppress_flag in enumerate(suppress_by_mask_metric_mask):
                        if suppress_flag:
                            # If mask IoU is high, AND its original bbox IoU was also somewhat high, then suppress.
                            # This prevents suppressing distant objects that happen to have high mask IoU due to occlusion.
                            # Check metric_values[metric_values_indices_for_mask_check[idx_in_candidates]]
                            original_bbox_metric_for_this_mask = metric_values[metric_values_indices_for_mask_check[idx_in_candidates]]
                            if original_bbox_metric_for_this_mask > 0.01: # Small threshold for box overlap
                                indices_to_remove_from_order.add(mask_candidates_indices_in_order[idx_in_candidates])
            
            order = [idx for idx in order if idx not in indices_to_remove_from_order]

        if cls_indexes_for_mapping is not None:
            return [cls_indexes_for_mapping[k].item() for k in keep]
        return keep

    def not_agnostic_nms(
        self, cls_ids_tensor: torch.Tensor, conf_tensor: torch.Tensor, boxes_tensor: torch.Tensor,
        match_metric: str, nms_threshold: float, masks_list: List[np.ndarray], intelligent_sorter: bool
    ) -> List[int]:
        all_kept_indices = []
        unique_classes = torch.unique(cls_ids_tensor)

        for cls_id_val in unique_classes:
            cls_mask = (cls_ids_tensor == cls_id_val)
            original_indices_for_this_class = torch.where(cls_mask)[0]

            if len(original_indices_for_this_class) == 0: continue

            conf_for_cls = conf_tensor[cls_mask]
            boxes_for_cls = boxes_tensor[cls_mask]
            
            masks_for_cls = []
            if masks_list and len(masks_list) > 0:
                 # Ensure indices are valid before trying to access masks_list
                 masks_for_cls = [masks_list[i.item()] for i in original_indices_for_this_class if i.item() < len(masks_list) and masks_list[i.item()] is not None]


            kept_local_indices = self.nms(
                conf_for_cls, boxes_for_cls, match_metric, nms_threshold,
                masks_for_cls, intelligent_sorter,
                cls_indexes_for_mapping=None
            )
            
            for local_idx in kept_local_indices:
                all_kept_indices.append(original_indices_for_this_class[local_idx].item())
        
        return all_kept_indices

    def __str__(self) -> str:
        output_parts = ["CombineDetections Results:"]
        attrs_map = {
            "Filtered Confidences": self.filtered_confidences,
            "Filtered Boxes": self.filtered_boxes,
            "Filtered Class IDs": self.filtered_classes_id,
            "Filtered Class Names": self.filtered_classes_names,
            "Filtered Masks": self.filtered_masks,
            "Filtered Polygons": self.filtered_polygons,
        }
        has_content = False
        for name, attr_list in attrs_map.items():
            if attr_list:
                has_content = True
                count = len(attr_list)
                preview = ""
                if name == "Filtered Masks" and count > 0 and isinstance(attr_list[0], np.ndarray):
                    preview = f"list of {count} binary masks, first shape: {attr_list[0].shape if attr_list[0] is not None else 'None'}"
                elif name == "Filtered Polygons" and count > 0 and isinstance(attr_list[0], np.ndarray):
                    preview = f"list of {count} polygons, first example points: {attr_list[0].shape[0] if attr_list[0] is not None else 'N/A'}"
                elif count > 3:
                    preview = f"{str(attr_list[:3])[:-1]} ...]" # Show first 3
                else:
                    preview = str(attr_list)
                output_parts.append(f"  {name}: {preview} (Total: {count})")
        
        if not has_content:
            return "CombineDetections Results: Nothing detected or NMS filtered everything."
        return "\n".join(output_parts)