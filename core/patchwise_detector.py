# project/core/patchwise_detector.py

import torch
import cv2
from torchvision.ops import nms
from ultralytics.engine.results import Boxes

class InferenceResult:
    """A compatibility class to mimic ultralytics.YOLO.Results object."""
    def __init__(self, boxes_tensor):
        self.boxes = self._create_boxes_object(boxes_tensor)

    def _create_boxes_object(self, tensor):
        mock_boxes = Boxes(tensor, orig_shape=None)
        return mock_boxes

def _optimized_single_frame_inference(model, frame_bgr, config):
    """Optimized patch inference for a single frame."""
    img_h, img_w = frame_bgr.shape[:2]
    crop_h, crop_w = config['crop_height'], config['crop_width']
    stride_x = int(crop_w * (1 - config['overlap_x'] / 100))
    stride_y = int(crop_h * (1 - config['overlap_y'] / 100))

    crop_coordinates = []
    for y in range(0, img_h, stride_y):
        for x in range(0, img_w, stride_x):
            x1, y1 = x, y
            x2, y2 = min(x + crop_w, img_w), min(y + crop_h, img_h)
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                crop_coordinates.append((x1, y1, x2, y2))

    if not crop_coordinates:
        return InferenceResult(torch.empty(0, 6, device=model.device))

    crop_images_list = [frame_bgr[y1:y2, x1:x2] for x1, y1, x2, y2 in crop_coordinates]

    results_list = model.predict(
        crop_images_list,
        imgsz=config['model_input_size'],
        conf=config['base_model_conf'],
        iou=config['postprocess_iou_thresh'],
        verbose=False
    )

    all_boxes, all_scores, all_classes = [], [], []
    for i, results in enumerate(results_list):
        if results.boxes is None or len(results.boxes.data) == 0:
            continue
        
        detections = results.boxes.data.clone()
        offset_x, offset_y = crop_coordinates[i][0], crop_coordinates[i][1]
        detections[:, :4] += torch.tensor([offset_x, offset_y, offset_x, offset_y], device=detections.device)
        
        all_boxes.append(detections[:, :4])
        all_scores.append(detections[:, 4])
        all_classes.append(detections[:, 5])

    if not all_boxes:
        return InferenceResult(torch.empty(0, 6, device=model.device))

    boxes_tensor = torch.cat(all_boxes, dim=0)
    scores_tensor = torch.cat(all_scores, dim=0)
    classes_tensor = torch.cat(all_classes, dim=0)

    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=config['nms_thresh'])

    final_boxes = boxes_tensor[keep_indices]
    final_scores = scores_tensor[keep_indices]
    final_classes = classes_tensor[keep_indices]

    final_detections_tensor = torch.cat([
        final_boxes,
        final_scores.unsqueeze(1),
        final_classes.unsqueeze(1)
    ], dim=1)

    return InferenceResult(final_detections_tensor)

def _optimized_batch_frame_inference(model, frames_bgr, config):
    """High-throughput patch inference on a BATCH of frames, with chunking for TRT compatibility."""
    all_crop_images = []
    all_crop_coords = []
    patches_per_frame = []

    for frame_bgr in frames_bgr:
        img_h, img_w = frame_bgr.shape[:2]
        crop_h, crop_w = config['crop_height'], config['crop_width']
        stride_x = int(crop_w * (1 - config['overlap_x'] / 100))
        stride_y = int(crop_h * (1 - config['overlap_y'] / 100))
        
        num_patches_this_frame = 0
        for y in range(0, img_h, stride_y):
            for x in range(0, img_w, stride_x):
                x1, y1 = x, y
                x2, y2 = min(x + crop_w, img_w), min(y + crop_h, img_h)
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    all_crop_images.append(frame_bgr[y1:y2, x1:x2])
                    all_crop_coords.append((x1, y1))
                    num_patches_this_frame += 1
        patches_per_frame.append(num_patches_this_frame)

    if not all_crop_images:
        return [InferenceResult(torch.empty(0, 6, device=model.device))] * len(frames_bgr)

    results_list = []
    max_batch_size = 64
    for i in range(0, len(all_crop_images), max_batch_size):
        chunk = all_crop_images[i : i + max_batch_size]
        chunk_results = model.predict(
            chunk,
            imgsz=config['model_input_size'],
            conf=config['base_model_conf'],
            iou=config['postprocess_iou_thresh'],
            verbose=False
        )
        results_list.extend(chunk_results)

    final_results_for_batch = []
    current_patch_idx = 0
    for num_patches in patches_per_frame:
        frame_patch_results = results_list[current_patch_idx : current_patch_idx + num_patches]
        frame_patch_coords = all_crop_coords[current_patch_idx : current_patch_idx + num_patches]
        
        all_boxes, all_scores, all_classes = [], [], []
        for i, patch_result in enumerate(frame_patch_results):
            if patch_result.boxes is None or len(patch_result.boxes.data) == 0:
                continue

            detections = patch_result.boxes.data.clone()
            offset_x, offset_y = frame_patch_coords[i]
            detections[:, :4] += torch.tensor([offset_x, offset_y, offset_x, offset_y], device=detections.device)
            
            all_boxes.append(detections[:, :4])
            all_scores.append(detections[:, 4])
            all_classes.append(detections[:, 5])

        if not all_boxes:
            final_results_for_batch.append(InferenceResult(torch.empty(0, 6, device=model.device)))
            current_patch_idx += num_patches
            continue

        boxes_tensor = torch.cat(all_boxes, dim=0)
        scores_tensor = torch.cat(all_scores, dim=0)
        classes_tensor = torch.cat(all_classes, dim=0)

        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=config['nms_thresh'])
        
        final_boxes = boxes_tensor[keep_indices]
        final_scores = scores_tensor[keep_indices]
        final_classes = classes_tensor[keep_indices]

        final_detections_tensor = torch.cat([
            final_boxes,
            final_scores.unsqueeze(1),
            final_classes.unsqueeze(1)
        ], dim=1)
        
        final_results_for_batch.append(InferenceResult(final_detections_tensor))
        current_patch_idx += num_patches

    return final_results_for_batch

class PatchwiseDetector:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def process_frame(self, frame_bgr):
        """Processes a single frame using patch-based inference."""
        return _optimized_single_frame_inference(self.model, frame_bgr, self.config)
        
    def process_batch(self, frames_bgr):
        """Processes a batch of frames using high-throughput patch-based inference."""
        return _optimized_batch_frame_inference(self.model, frames_bgr, self.config)