# project/tracking/bytetrack_wrapper.py
import numpy as np
# from types import SimpleNamespace # Not needed if args passed directly
from tracking.byte_track import BYTETracker
from core.utils import debug_print

class ByteTrackWrapper:
    def __init__(self, bytetrack_args_namespace, frame_rate, display_resolution_wh):
        """
        Initializes the ByteTrackWrapper.
        Args:
            bytetrack_args_namespace: A namespace object with ByteTrack parameters
                                      (track_high_thresh, track_low_thresh, etc.)
            frame_rate: Frame rate of the video (for ByteTrack's max_time_lost).
            display_resolution_wh: Tuple (width, height) of the display/output video.
        """
        self.args = bytetrack_args_namespace
        self.byte_tracker_instance = BYTETracker(self.args, frame_rate=frame_rate)
        self.display_w, self.display_h = display_resolution_wh # (e.g., 640, 480)

    def _scale_bboxes_from_model_to_display(self, bboxes_model_xyxy, model_input_shape_hw):
        """
        Scales bounding boxes from model input space to display resolution space.
        Args:
            bboxes_model_xyxy (np.array): Nx4 array of (x1,y1,x2,y2) in model input space.
            model_input_shape_hw (tuple): (height, width) of the image fed to the model.
        Returns:
            np.array: Nx4 array of (x1,y1,x2,y2) in display resolution space.
        """
        if not isinstance(bboxes_model_xyxy, np.ndarray) or bboxes_model_xyxy.shape[0] == 0:
            return np.empty((0, 4), dtype=float) # Return float for consistency before int conversion

        model_h, model_w = model_input_shape_hw # e.g., (416, 416)
        
        # Ensure model dimensions are valid
        if model_w <= 0 or model_h <= 0:
            print(f"Warning: Invalid model input dimensions for scaling: W={model_w}, H={model_h}")
            return np.empty((0,4), dtype=float)

        scaled_boxes = bboxes_model_xyxy.astype(float).copy() # Work with float for scaling

        # Scale factor
        scale_x = self.display_w / model_w
        scale_y = self.display_h / model_h

        scaled_boxes[:, 0] = bboxes_model_xyxy[:, 0] * scale_x  # x1
        scaled_boxes[:, 1] = bboxes_model_xyxy[:, 1] * scale_y  # y1
        scaled_boxes[:, 2] = bboxes_model_xyxy[:, 2] * scale_x  # x2
        scaled_boxes[:, 3] = bboxes_model_xyxy[:, 3] * scale_y  # y2
        
        # Clip to display boundaries
        scaled_boxes[:, 0] = np.clip(scaled_boxes[:, 0], 0, self.display_w -1)
        scaled_boxes[:, 1] = np.clip(scaled_boxes[:, 1], 0, self.display_h -1)
        scaled_boxes[:, 2] = np.clip(scaled_boxes[:, 2], 0, self.display_w -1)
        scaled_boxes[:, 3] = np.clip(scaled_boxes[:, 3], 0, self.display_h -1)
        
        return scaled_boxes.astype(int) # Return as int for pixel coordinates


    def process_frame_detections(self, yolo_results_single_frame):
        """
        Processes detections for a single frame using ByteTrack and scales results.
        Args:
            yolo_results_single_frame: An Ultralytics Results object for one frame.
                                       Assumes boxes are in model input space.
        Returns:
            list_of_tracks: A list of dictionaries, each representing an active track.
                            [{'id': int,
                              'box_xyxy_display': (x1,y1,x2,y2), # Scaled to display
                              'class_id': int,
                              'conf': float,
                              'class_name': str}, ...]
        """
        # yolo_results_single_frame is an ultralytics.engine.results.Results object.
        # It has attributes like:
        # - .boxes.xyxy (Tensor, model input space)
        # - .boxes.conf (Tensor)
        # - .boxes.cls (Tensor)
        # - .orig_shape (tuple, H,W of image fed to model, e.g., (416,416))
        # - .names (dict, class_id to class_name map)

        # ByteTrack's update method expects an object that itself has .xywh, .conf, .cls attributes
        # The `yolo_results_single_frame` object is ALREADY in this compatible format for ByteTrack.
        
        online_targets_model_space = self.byte_tracker_instance.update(yolo_results_single_frame)
        # online_targets_model_space is a NumPy array:
        # [[x1, y1, x2, y2, track_id, score, cls_id, det_idx_in_input_results], ...]
        # The x1,y1,x2,y2 are in the *model input coordinate system*.

        processed_tracks_for_app = []
        if online_targets_model_space.shape[0] > 0:
            bboxes_model_xyxy = online_targets_model_space[:, :4] # In model input space
            track_ids = online_targets_model_space[:, 4].astype(int)
            confs = online_targets_model_space[:, 5]
            class_ids = online_targets_model_space[:, 6].astype(int)

            model_input_sh_hw = yolo_results_single_frame.orig_shape # (H, W) of image fed to model
            
            bboxes_display_xyxy = self._scale_bboxes_from_model_to_display(
                bboxes_model_xyxy, model_input_sh_hw
            )

            for i in range(len(track_ids)):
                class_name = yolo_results_single_frame.names.get(class_ids[i], f"CLS_{class_ids[i]}")
                processed_tracks_for_app.append({
                    'id': track_ids[i],
                    'box_xyxy_display': tuple(bboxes_display_xyxy[i]), # (x1,y1,x2,y2) in display pixels
                    'class_id': class_ids[i],
                    'conf': confs[i],
                    'class_name': class_name
                })
        debug_print(f"ByteTrackWrapper: Input Dets (YOLO): {len(yolo_results_single_frame.boxes) if yolo_results_single_frame.boxes else 0}, "
                    f"ByteTrack Raw Output Targets: {online_targets_model_space.shape[0]}, "
                    f"Processed (Scaled) Tracks: {len(processed_tracks_for_app)}")
        return processed_tracks_for_app

    def reset_tracker(self):
        """Resets the internal ByteTrack instance."""
        self.byte_tracker_instance.reset()