# project/core/patchwise_detector.py

import torch
import numpy as np
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections
from ultralytics.engine.results import Boxes

class InferenceResult:
    """A compatibility class to mimic ultralytics.YOLO.Results object."""
    def __init__(self, boxes_tensor):
        self.boxes = self._create_boxes_object(boxes_tensor)

    def _create_boxes_object(self, tensor):
        # The post-processor primarily needs the .data attribute of the boxes object.
        # We create a mock object that has this attribute.
        mock_boxes = Boxes(tensor, orig_shape=None)
        return mock_boxes

class PatchwiseDetector:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def _convert_to_result_format(self, nms_detections):
        boxes, confidences, classes = nms_detections
        if not boxes:
            return InferenceResult(torch.empty(0, 6, device=self.model.device))

        # Combine into a single tensor of shape [N, 6] -> [x1, y1, x2, y2, conf, cls]
        detections_tensor = torch.tensor(np.column_stack([
            boxes,
            confidences,
            classes
        ]), device=self.model.device, dtype=torch.float32)

        return InferenceResult(detections_tensor)

    @torch.no_grad()
    def process_frame(self, frame_bgr):
        element_crops = MakeCropsDetectThem(
            image=frame_bgr,
            model=self.model,
            imgsz=self.config['model_input_size'],
            conf=self.config['base_model_conf'],
            iou=self.config['postprocess_iou_thresh'],
            shape_x=self.config['crop_width'],
            shape_y=self.config['crop_height'],
            overlap_x=self.config['overlap_x'],
            overlap_y=self.config['overlap_y'],
            batch_inference=True
        )

        combined_result = CombineDetections(
            element_crops=element_crops,
            nms_threshold=self.config['nms_thresh'],
            match_metric=self.config['match_metric']
        )

        nms_detections = (
            combined_result.filtered_boxes,
            combined_result.filtered_confidences,
            combined_result.filtered_classes_id
        )

        return self._convert_to_result_format(nms_detections)