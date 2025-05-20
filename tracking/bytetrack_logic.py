# project/tracking/bytetrack_logic.py
import numpy as np
from scipy.optimize import linear_sum_assignment # For Hungarian algorithm
from tracking.kalman_filter_wrapper import KalmanFilterXYAH
from config import DEBUG_MODE_TRACKING
import sys
from datetime import datetime

if DEBUG_MODE_TRACKING: # Only redirect if debugging tracking
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"tracking/logs/bytetrack_debug_log_{timestamp_str}.txt"
    print(f"DEBUG: Redirecting ByteTrack debug prints to: {log_filename}")
    original_stdout = sys.stdout 
    log_file = open(log_filename, "w", encoding='utf-8')
    sys.stdout = log_file

# --- Bounding Box Utility Functions ---
def tlwh_to_xyah(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2.0
    if ret[3] > 0: ret[2] /= ret[3] # aspect_ratio = width / height
    else: ret[2] = 0 # Avoid division by zero if height is zero
    return ret

def xyah_to_tlwh(xyah):
    ret = np.asarray(xyah).copy()
    ret[2] *= ret[3] # width = aspect_ratio * height
    ret[:2] -= ret[2:] / 2.0
    return ret

def tlwh_to_xyxy(tlwh):
    x1, y1, w, h = tlwh
    return np.array([x1, y1, x1 + w, y1 + h], dtype=tlwh.dtype)

def xyxy_to_tlwh(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=xyxy.dtype)

def iou_batch(bboxes1_tlwh, bboxes2_tlwh):
    if bboxes1_tlwh.ndim == 1: bboxes1_tlwh = bboxes1_tlwh[None, :] # Ensure 2D
    if bboxes2_tlwh.ndim == 1: bboxes2_tlwh = bboxes2_tlwh[None, :] # Ensure 2D

    if bboxes1_tlwh.size == 0 or bboxes2_tlwh.size == 0:
        return np.empty((bboxes1_tlwh.shape[0], bboxes2_tlwh.shape[0]))
    
    bboxes1_areas = (bboxes1_tlwh[:, 2]) * (bboxes1_tlwh[:, 3])
    bboxes2_areas = (bboxes2_tlwh[:, 2]) * (bboxes2_tlwh[:, 3])
    
    inter_tl_x = np.maximum(bboxes1_tlwh[:, None, 0], bboxes2_tlwh[None, :, 0])
    inter_tl_y = np.maximum(bboxes1_tlwh[:, None, 1], bboxes2_tlwh[None, :, 1])
    inter_br_x = np.minimum(bboxes1_tlwh[:, None, 0] + bboxes1_tlwh[:, None, 2], 
                            bboxes2_tlwh[None, :, 0] + bboxes2_tlwh[None, :, 2])
    inter_br_y = np.minimum(bboxes1_tlwh[:, None, 1] + bboxes1_tlwh[:, None, 3], 
                            bboxes2_tlwh[None, :, 1] + bboxes2_tlwh[None, :, 3])

    inter_widths = np.maximum(0, inter_br_x - inter_tl_x)
    inter_heights = np.maximum(0, inter_br_y - inter_tl_y)
    inter_area = inter_widths * inter_heights

    union_area = bboxes1_areas[:, None] + bboxes2_areas[None, :] - inter_area
    iou_matrix = inter_area / (union_area + 1e-7) 
    return iou_matrix

class STrack:
    _count = 0  # Class variable for unique track IDs

    def __init__(self, tlwh, score, class_id, class_name, frame_id, kalman_filter_dt=1.0):
        self.track_id = STrack._get_next_id()
        self._tlwh = np.asarray(tlwh, dtype=float) # Internal [top, left, width, height]
        self.score = score
        self.class_id = class_id
        self.class_name_initial = class_name # First detected class name
        self.class_name_consistent = class_name # Smoothed class name

        self.kalman_filter = KalmanFilterXYAH(dt=kalman_filter_dt)
        self.mean, self.covariance = None, None # KF will manage these internally

        self.state = "new"  # "new", "tracked", "lost", "removed"
        self.is_activated = False # Becomes True after first successful update based on high-score detection
        
        self.tracklet_len = 0 # Number of frames this track has been actively tracked
        self.frames_since_update = 0 # Age: frames since last matched update
        
        self.frame_id = frame_id # Frame ID of last update
        self.start_frame = frame_id # Frame ID when track was initiated

        self.type_history_counts = {class_name: 1}

    @staticmethod
    def _get_next_id():
        STrack._count += 1
        return STrack._count
    
    def _update_consistent_type(self, current_class_name):
        self.type_history_counts[current_class_name] = self.type_history_counts.get(current_class_name, 0) + 1
        self.class_name_consistent = max(self.type_history_counts, key=self.type_history_counts.get)

    def predict(self):
        prev_mean_str = "None"
        if self.mean is not None:
            prev_mean_str = np.array2string(self.mean[:4], formatter={'float_kind':lambda x: "%.2f" % x})
            self.kalman_filter.kf.x = self.mean.reshape(-1,1)
            self.kalman_filter.kf.P = self.covariance
        
        self.kalman_filter.predict() # This updates kf.x and kf.P (and kf.x_prior)
        
        # Store the predicted state (this is kf.x after kf.predict())
        self.mean, self.covariance = self.kalman_filter.kf.x.flatten(), self.kalman_filter.kf.P
        self.frames_since_update += 1

        if DEBUG_MODE_TRACKING and self.track_id < 5: # Limit printing to first few tracks for less spam
             print(f"  STrack ID {self.track_id} PREDICT: Prev Mean(xyah): {prev_mean_str} -> New Mean(xyah): {np.array2string(self.mean[:4], formatter={'float_kind':lambda x: '%.2f' % x})}, Age: {self.frames_since_update}")

    # Add print in update methods
    def _log_update(self, type, detection_track_tlwh):
        if DEBUG_MODE_TRACKING and self.track_id < 5:
            print(f"  STrack ID {self.track_id} {type}: Det TLWH: {np.array2string(detection_track_tlwh, formatter={'float_kind':lambda x: '%.2f' % x})}, New KF Mean(xyah): {np.array2string(self.mean[:4], formatter={'float_kind':lambda x: '%.2f' % x})}")

    def activate(self, frame_id, detection_track):
        self.kalman_filter.initiate(tlwh_to_xyah(detection_track.tlwh))
        self.mean, self.covariance = self.kalman_filter.kf.x.flatten(), self.kalman_filter.kf.P
        self._tlwh = detection_track._tlwh
        self.score = detection_track.score
        self._update_consistent_type(detection_track.class_name_initial)
        self.state = "tracked"; self.is_activated = True; self.frame_id = frame_id
        self.tracklet_len = 1; self.frames_since_update = 0
        self._log_update("ACTIVATE", detection_track.tlwh)

    def re_activate(self, frame_id, detection_track):
        self.kalman_filter.update(tlwh_to_xyah(detection_track.tlwh))
        self.mean, self.covariance = self.kalman_filter.kf.x.flatten(), self.kalman_filter.kf.P
        self._tlwh = detection_track._tlwh; self.score = detection_track.score
        self._update_consistent_type(detection_track.class_name_initial)
        self.state = "tracked"; self.is_activated = True; self.frame_id = frame_id
        self.tracklet_len += 1; self.frames_since_update = 0
        self._log_update("RE-ACTIVATE", detection_track.tlwh)

    def update_match(self, frame_id, detection_track):
        self.kalman_filter.update(tlwh_to_xyah(detection_track.tlwh))
        self.mean, self.covariance = self.kalman_filter.kf.x.flatten(), self.kalman_filter.kf.P
        self._tlwh = detection_track._tlwh; self.score = detection_track.score
        self._update_consistent_type(detection_track.class_name_initial)
        self.state = "tracked"; self.is_activated = True; self.frame_id = frame_id
        self.tracklet_len += 1; self.frames_since_update = 0
        self._log_update("UPDATE_MATCH", detection_track.tlwh)

    def mark_lost(self):
        self.state = "lost"

    def mark_removed(self):
        self.state = "removed"

    @property
    def tlwh(self): # Public accessor for current bbox
        # If activated and not 'new', use KF's smoothed state
        if self.is_activated and self.state != "new" and self.mean is not None:
            return xyah_to_tlwh(self.mean[:4]) # Use [x,y,a,h] from KF state
        return self._tlwh # Fallback to last known detection _tlwh

    @property
    def xyxy(self):
        return tlwh_to_xyxy(self.tlwh)

class ByteTracker:
    def __init__(self, track_thresh=0.5, track_high_thresh=0.6, new_track_thresh=0.7,
                 match_thresh_iou=0.8, track_buffer_age_frames=30, kalman_filter_dt=1.0):
        self.track_thresh = track_thresh
        self.track_high_thresh = track_high_thresh
        self.new_track_thresh = new_track_thresh # Must be >= track_high_thresh
        self.match_thresh_iou = match_thresh_iou
        self.track_buffer_age_frames = track_buffer_age_frames
        self.kalman_filter_dt = kalman_filter_dt

        self.current_frame_id_internal = 0 # Renamed to avoid clash if frame_id is passed
        STrack._count = 0 
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.debug_frame_count = 0 # For limiting debug prints
        self.max_debug_frames = 0  # Print details for first N frames ByteTracker processes
    
    def update(self, detections_info_list, current_video_frame_id_for_debug=-1): # Added external frame ID for debug 
        self.current_frame_id_internal += 1 # ByteTracker's internal frame counter
        self.debug_frame_count +=1
        do_debug_prints = DEBUG_MODE_TRACKING and (self.debug_frame_count <= self.max_debug_frames)

        if do_debug_prints:
            print(f"\n--- ByteTracker Update - External Frame: {current_video_frame_id_for_debug}, Internal BT Frame: {self.current_frame_id_internal} ---")
            print(f"Input Detections ({len(detections_info_list)}):")
            for i, det in enumerate(detections_info_list[:5]): # Print first 5 dets
                print(f"  Det {i}: XYXY={det['bbox_xyxy']}, Score={det['score']:.2f}, Class={det['class_name']}")
            print(f"Current Track Stats: Tracked={len(self.tracked_stracks)}, Lost={len(self.lost_stracks)}")
            print(f"ByteTrack Params: track_thresh={self.track_thresh}, high_thresh={self.track_high_thresh}, new_thresh={self.new_track_thresh}, iou_match_thresh={self.match_thresh_iou}, buffer_age={self.track_buffer_age_frames}, kf_dt={self.kalman_filter_dt}")


        active_tracks_pool = []
        unconfirmed_new_tracks_pool = []

        for track in self.tracked_stracks:
            if not track.is_activated: unconfirmed_new_tracks_pool.append(track)
            else: active_tracks_pool.append(track)

        if do_debug_prints:
            print(f"Predicting for {len(active_tracks_pool)} active tracks and {len(self.lost_stracks)} lost tracks.")
        for track in active_tracks_pool: track.predict()
        for track in self.lost_stracks: track.predict()
        
        current_detections_as_stracks = []
        if detections_info_list:
            for det_info in detections_info_list:
                if det_info['score'] >= self.track_thresh:
                    tlwh = xyxy_to_tlwh(np.array(det_info['bbox_xyxy']))
                    current_detections_as_stracks.append(STrack(
                        tlwh, det_info['score'], det_info['class_id'],
                        det_info['class_name'], self.current_frame_id_internal, self.kalman_filter_dt # Pass current_frame_id_internal
                    ))
        high_conf_dets = [d for d in current_detections_as_stracks if d.score >= self.track_high_thresh]
        low_conf_dets = [d for d in current_detections_as_stracks if self.track_thresh <= d.score < self.track_high_thresh]

        if do_debug_prints:
            print(f"Total Dets qualifying (score >= {self.track_thresh}): {len(current_detections_as_stracks)}")
            print(f"High-conf Dets (score >= {self.track_high_thresh}): {len(high_conf_dets)}")
            print(f"Low-conf Dets ({self.track_thresh} <= score < {self.track_high_thresh}): {len(low_conf_dets)}")

        activated_current_frame = []
        refound_current_frame = []
        newly_lost_this_step = []

        # --- First association (High-conf with active_tracks_pool + unconfirmed_new_tracks_pool) ---
        tracks_for_high_match = active_tracks_pool + unconfirmed_new_tracks_pool
        if do_debug_prints: print(f"\n--- Assoc 1 (High-Conf Dets vs Active/Unconfirmed Tracks) ---")
        matches1, u_dets1_indices, u_tracks1_indices = self._associate_iou(
            high_conf_dets, tracks_for_high_match, self.match_thresh_iou, "Assoc1", do_debug_prints
        )
        
        for i_det, i_track in matches1:
            track = tracks_for_high_match[i_track]
            det = high_conf_dets[i_det]
            if track.state == "new": track.activate(self.current_frame_id_internal, det) 
            else: track.update_match(self.current_frame_id_internal, det)
            activated_current_frame.append(track)
        u_high_conf_dets_after_match1 = [high_conf_dets[i] for i in u_dets1_indices]
        for i_track in u_tracks1_indices:
            track = tracks_for_high_match[i_track]
            if track.is_activated: track.mark_lost(); newly_lost_this_step.append(track)
        
        u_high_conf_dets_after_match1 = [high_conf_dets[i] for i in u_dets1_indices] # Correct: u_dets1_indices are for high_conf_dets
        
        for i_track in u_tracks1_indices: # i_track is index into tracks_for_high_match
            track = tracks_for_high_match[i_track]
            if track.is_activated: 
                track.mark_lost()
                newly_lost_this_step.append(track)

        # --- Second association (Low-conf with tracks that just became lost from active_tracks_pool) ---
        candidate_lost_for_low_match = [t for t in newly_lost_this_step if t in active_tracks_pool]
        if do_debug_prints: print(f"\n--- Assoc 2 (Low-Conf Dets vs Recently Lost Active Tracks) ---")
        matches2, u_low_dets_indices, u_lost_tracks2_indices = self._associate_iou(
            low_conf_dets, candidate_lost_for_low_match, 0.5, "Assoc2", do_debug_prints # Lower IoU for low-score
        )

        for i_det, i_track in matches2: # i_det is index into low_conf_dets, i_track is index into candidate_lost_for_low_match
            track = candidate_lost_for_low_match[i_track]
            det = low_conf_dets[i_det]   # <<<<<<< CORRECTED HERE: Use i_det directly
            track.re_activate(self.current_frame_id_internal, det)
            refound_current_frame.append(track)
            if track in newly_lost_this_step: newly_lost_this_step.remove(track)
        
        # --- Third association (Remaining high-conf dets with self.lost_stracks (long-term lost)) ---
        if do_debug_prints: print(f"\n--- Assoc 3 (Unmatched High-Conf Dets vs Long-Lost Tracks) ---")
        matches3, u_high_conf_dets3_indices, u_long_lost3_indices = self._associate_iou(
            u_high_conf_dets_after_match1, self.lost_stracks, self.match_thresh_iou, "Assoc3", do_debug_prints
        )
        for i_det, i_track in matches3:
            track = self.lost_stracks[i_track]
            det = u_high_conf_dets_after_match1[i_det]
            track.re_activate(self.current_frame_id_internal, det)
            refound_current_frame.append(track)
        final_long_lost_kept = []
        for i_track in u_long_lost3_indices:
            track = self.lost_stracks[i_track]
            if track.frames_since_update <= self.track_buffer_age_frames: final_long_lost_kept.append(track)
            else: track.mark_removed(); self.removed_stracks.append(track)
        self.lost_stracks = final_long_lost_kept

        # --- Initialize new tracks ---
        new_track_candidate_dets = [u_high_conf_dets_after_match1[i] for i in u_high_conf_dets3_indices]
        if do_debug_prints: print(f"\n--- New Track Initialization (Candidates: {len(new_track_candidate_dets)}) ---")
        for det_track in new_track_candidate_dets:
            if det_track.score >= self.new_track_thresh:
                det_track.activate(self.current_frame_id_internal, det_track) # Pass itself as the 'detection_track'
                activated_current_frame.append(det_track)
                if do_debug_prints: print(f"  NEW Track Initialized: ID {det_track.track_id} from Det TLWH {np.array2string(det_track.tlwh, formatter={'float_kind':lambda x: '%.2f' % x})}")

        # --- Update global track lists ---
        current_tracked_stracks = []
        for t in (activated_current_frame + refound_current_frame):
            if t.state == "tracked": current_tracked_stracks.append(t)
        self.tracked_stracks = current_tracked_stracks
        for t in newly_lost_this_step:
            if t.state == "lost" and t not in self.lost_stracks and t not in self.tracked_stracks:
                self.lost_stracks.append(t)
        for track in unconfirmed_new_tracks_pool:
            if track.state == "new" and track not in self.tracked_stracks: 
                if track.frames_since_update <= self.track_buffer_age_frames :
                     track.mark_lost(); 
                     if track not in self.lost_stracks: self.lost_stracks.append(track)
                else: 
                     track.mark_removed(); 
                     if track not in self.removed_stracks: self.removed_stracks.append(track)
        unique_lost_ids = set()
        filtered_lost_stracks = []
        for t in self.lost_stracks: # Iterate over a copy or manage indices carefully if removing
            if t.track_id not in unique_lost_ids and t.state == "lost":
                filtered_lost_stracks.append(t)
                unique_lost_ids.add(t.track_id)
        self.lost_stracks = filtered_lost_stracks
        
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        if do_debug_prints:
            print(f"End of Update: Final Tracked={len(self.tracked_stracks)} (Outputting {len(output_stracks)} activated), Lost={len(self.lost_stracks)}, Removed={len(self.removed_stracks)}")
            for i, trk in enumerate(output_stracks[:3]):
                print(f"  Output Track {i}: ID={trk.track_id}, TLWH={np.array2string(trk.tlwh, formatter={'float_kind':lambda x: '%.2f' % x})}, Score={trk.score:.2f}, State={trk.state}, Activated={trk.is_activated}, Age={trk.frames_since_update}, Len={trk.tracklet_len}")
        
            # --- Pause Condition ---
            if DEBUG_MODE_TRACKING and self.max_debug_frames > 0 and self.current_frame_id_internal == self.max_debug_frames:
                print(f"\nDEBUG: Reached max_debug_frames ({self.max_debug_frames}).")
                input("Press Enter to continue processing (or Ctrl+C to stop)...")
                # self.pause_after_debug = False 
            # --- End Pause ---
        
        return output_stracks
    
    def _associate_iou(self, detections_as_stracks, tracks_to_match, iou_threshold, assoc_name="Assoc", do_debug_prints=False):
        # `do_debug_prints` is passed from ByteTracker.update based on its internal frame count limit

        if do_debug_prints:
            print(f"    >>>>>> {assoc_name} - Input <<<<<<")
            print(f"    Num Candidate Detections (STracks): {len(detections_as_stracks)}")
            for i, det_s in enumerate(detections_as_stracks[:3]): # Print first 3 candidate detections
                print(f"      DetCandidate {i}: TLWH={np.array2string(det_s.tlwh, formatter={'float_kind':lambda x: '%.2f' % x})}, Score={det_s.score:.2f}")
            
            print(f"    Num Existing Tracks to Match Against: {len(tracks_to_match)}")
            for i, trk_m in enumerate(tracks_to_match[:5]): # Print first 5 existing tracks
                # The trk_m.tlwh here will be based on its KF predicted state (because STrack.predict() was called before this)
                print(f"      ExistingTrack {i}: ID={trk_m.track_id}, State='{trk_m.state}', Activated={trk_m.is_activated}, Predicted TLWH={np.array2string(trk_m.tlwh, formatter={'float_kind':lambda x: '%.2f' % x})}, Age={trk_m.frames_since_update}")

        if not detections_as_stracks or not tracks_to_match:
            # ... (return empty matches as before) ...
            return np.empty((0,2), dtype=int), list(range(len(detections_as_stracks))), list(range(len(tracks_to_match)))
        
        # Prepare TLWH bboxes for iou_batch
        det_bboxes_tlwh_list = [d.tlwh for d in detections_as_stracks]
        trk_bboxes_tlwh_list = []
        for t_idx, t in enumerate(tracks_to_match):
            # This STrack.tlwh uses the KF's predicted state (self.mean[:4]) if available
            # because STrack.predict() has already been called on all tracks_to_match
            current_track_tlwh = t.tlwh 
            trk_bboxes_tlwh_list.append(current_track_tlwh)
            if do_debug_prints and t.track_id <= 4 : # Focus on early tracks
                 print(f"    PRE-IOU Check for Track ID {t.track_id} (State: {t.state}, Activated: {t.is_activated}, Age: {t.frames_since_update}):")
                 print(f"      KF Mean (xyah) used for prediction: {np.array2string(t.mean[:4], formatter={'float_kind':lambda x: '%.2f' % x}) if t.mean is not None else 'None'}")
                 print(f"      Resulting TLWH for matching: {np.array2string(current_track_tlwh, formatter={'float_kind':lambda x: '%.2f' % x})}")
                 if t.mean is not None and len(t.mean) == 8:
                     print(f"      Estimated Velocities (vx,vy,va,vh): {np.array2string(t.mean[4:], formatter={'float_kind':lambda x: '%.2f' % x})}")

        # Ensure lists are not empty before converting to numpy array
        if not det_bboxes_tlwh_list or not trk_bboxes_tlwh_list:
             return np.empty((0,2), dtype=int), list(range(len(detections_as_stracks))), list(range(len(tracks_to_match)))

        det_bboxes_tlwh = np.array(det_bboxes_tlwh_list)
        trk_bboxes_tlwh = np.array(trk_bboxes_tlwh_list)
            
        iou_matrix = iou_batch(det_bboxes_tlwh, trk_bboxes_tlwh) # Calculate IoU

        if do_debug_prints and iou_matrix.size > 0:
            print(f"    >>>>>> {assoc_name} - IoU Matrix (Dets x Tracks, shape {iou_matrix.shape}) <<<<<<")
            # To make it more readable, let's print relevant parts, especially for early tracks
            # Print header: Track IDs
            track_ids_header = " ".join([f"TrkID-{t.track_id:02d}" for t in tracks_to_match[:10]]) # Show up to 10 track IDs
            print(f"    DetIdx | {track_ids_header}")
            print(f"    -------|--------------------------------------------------")
            for i_det, det_s in enumerate(detections_as_stracks[:5]): # For first 5 detections
                row_str = f"    Det {i_det:02d}  | "
                # Print IoU of this detection with first few tracks
                for i_trk, trk_m in enumerate(tracks_to_match[:10]):
                    row_str += f"{iou_matrix[i_det, i_trk]:.3f}   "
                print(row_str)
                # If you want to focus on a specific problematic STrack ID, e.g., ID 1:
                for i_trk, trk_m in enumerate(tracks_to_match):
                    if trk_m.track_id == 1 or trk_m.track_id == 2: # Check for early problematic IDs
                        print(f"        IoU(Det {i_det} with TrackID {trk_m.track_id}): {iou_matrix[i_det, i_trk]:.3f} (Det TLWH: {np.array2string(det_s.tlwh, formatter={'float_kind':lambda x: '%.2f' % x})}, Trk Pred TLWH: {np.array2string(trk_m.tlwh, formatter={'float_kind':lambda x: '%.2f' % x})})")
            print(f"    IoU Threshold for this association: {iou_threshold}")
        
        cost_matrix = 1 - iou_matrix
        if cost_matrix.size == 0:
             return np.empty((0,2), dtype=int), list(range(len(detections_as_stracks))), list(range(len(tracks_to_match)))
        
        cost_matrix[np.isnan(cost_matrix) | np.isinf(cost_matrix)] = 1.0 
            
        det_indices_assigned, track_indices_assigned = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_detections = list(range(len(detections_as_stracks)))
        unmatched_tracks = list(range(len(tracks_to_match)))
        
        for d_idx, t_idx in zip(det_indices_assigned, track_indices_assigned):
            if iou_matrix[d_idx, t_idx] >= iou_threshold:
                matches.append((d_idx, t_idx))
                if d_idx in unmatched_detections: unmatched_detections.remove(d_idx)
                if t_idx in unmatched_tracks: unmatched_tracks.remove(t_idx)
        
        if do_debug_prints:
            print(f"    >>>>>> {assoc_name} - Results <<<<<<")
            print(f"    Num Matches Made: {len(matches)}")
            for i, (d_idx, t_idx) in enumerate(matches[:5]): # Print first 5 matches
                det_s = detections_as_stracks[d_idx]
                trk_m = tracks_to_match[t_idx]
                print(f"      Match {i}: Det Idx {d_idx} (Score {det_s.score:.2f}) <-> Track Idx {t_idx} (ID {trk_m.track_id}, State '{trk_m.state}') with IoU {iou_matrix[d_idx, t_idx]:.3f}")
            # print(f"    Unmatched Detection Indices (from input list): {unmatched_detections[:5]}") # Can be long
            # print(f"    Unmatched Track Indices (from input list): {unmatched_tracks[:5]}")
        return np.array(matches, dtype=int), unmatched_detections, unmatched_tracks