# byte_track.py (Your local copy)

import numpy as np
import torch 

from tracker_utils import xywh2ltwh # For STrack init if converting from xywh (center) from detection
from tracker_basetrack import BaseTrack, TrackState
import tracker_matching as matching 
from tracker_utils import GPUKalmanFilterXYAH # Use the GPU Kalman Filter


class GPUSTrack(BaseTrack):
    def __init__(self, initial_tlwh_np: np.ndarray, score_scalar: float, cls_scalar: int, 
                 device: torch.device, kalman_filter_instance: GPUKalmanFilterXYAH):
        super().__init__()
        self.device = device
        self.kalman_filter = kalman_filter_instance
        self._tlwh_gpu = torch.from_numpy(initial_tlwh_np).float().to(self.device) # [x1,y1,w,h]
        self.mean_gpu, self.covariance_gpu = None, None 
        self.is_activated = False 
        self.score = float(score_scalar) 
        self.tracklet_len = 0
        self.cls = int(cls_scalar)     
        self.angle = None 

    @staticmethod
    def tlwh_to_xyah_gpu(tlwh_gpu: torch.Tensor) -> torch.Tensor:
        if tlwh_gpu.shape[0] == 0:
            return torch.empty((0,4), device=tlwh_gpu.device, dtype=tlwh_gpu.dtype)
        ret_gpu = tlwh_gpu.clone()
        ret_gpu[..., 0] = tlwh_gpu[..., 0] + tlwh_gpu[..., 2] / 2.0
        ret_gpu[..., 1] = tlwh_gpu[..., 1] + tlwh_gpu[..., 3] / 2.0
        valid_h_mask = tlwh_gpu[..., 3] > 1e-6
        if tlwh_gpu.ndim == 1 :
            if valid_h_mask: ret_gpu[2] = tlwh_gpu[2] / tlwh_gpu[3]
            else: ret_gpu[2] = torch.tensor(1.0, device=tlwh_gpu.device) 
        elif tlwh_gpu.ndim == 2:
            ret_gpu[..., 2] = torch.ones_like(ret_gpu[..., 2]) # Default aspect ratio to 1
            if valid_h_mask.any(): # Check if any True values exist before indexing
                 ret_gpu[valid_h_mask, 2] = tlwh_gpu[valid_h_mask, 2] / tlwh_gpu[valid_h_mask, 3]
        return ret_gpu

    def predict(self):
        if self.mean_gpu is None or self.kalman_filter is None: return
        mean_state_gpu = self.mean_gpu.clone()
        if self.state != TrackState.Tracked: mean_state_gpu[7] = 0.0 
        self.mean_gpu, self.covariance_gpu = self.kalman_filter.predict(mean_state_gpu, self.covariance_gpu)

    @staticmethod
    def multi_predict(stracks: list):
        if not stracks: return
        kf_instance = stracks[0].kalman_filter 
        means_list, covs_list, valid_indices = [], [], []
        for i, st in enumerate(stracks):
            if st.mean_gpu is not None and st.covariance_gpu is not None:
                mean_copy = st.mean_gpu.clone()
                if st.state != TrackState.Tracked: mean_copy[7] = 0.0
                means_list.append(mean_copy)
                covs_list.append(st.covariance_gpu); valid_indices.append(i)
        if not means_list: return
        means_batch_gpu = torch.stack(means_list); covs_batch_gpu = torch.stack(covs_list)
        predicted_means_gpu, predicted_covs_gpu = kf_instance.multi_predict(means_batch_gpu, covs_batch_gpu)
        for i, original_idx in enumerate(valid_indices):
            stracks[original_idx].mean_gpu = predicted_means_gpu[i]
            stracks[original_idx].covariance_gpu = predicted_covs_gpu[i]

    def activate(self, frame_id: int):
        if self.kalman_filter is None: print(f"ERROR: GPUSTrack activate: KF not set."); return
        self.track_id = self.next_id()
        measurement_xyah_gpu = self.tlwh_to_xyah_gpu(self._tlwh_gpu)
        self.mean_gpu, self.covariance_gpu = self.kalman_filter.initiate(measurement_xyah_gpu)
        self.is_activated = False 
        if self.mean_gpu is None or torch.isnan(self.mean_gpu).any() or torch.isinf(self.mean_gpu).any():
            print(f"  WARNING: GPUSTrack {self.track_id} - mean_gpu NaN/Inf AFTER KF INITIATE. Not activating.")
            self.state = TrackState.Removed; return 
        self.tracklet_len = 0; self.state = TrackState.Tracked
        self.is_activated = True; self.frame_id = frame_id; self.start_frame = frame_id
        # print(f"    GPUSTrack {self.track_id} ACTIVATED. Frame: {frame_id}. Score: {self.score:.2f}")

    def re_activate(self, new_det_gpustrack, frame_id: int, new_id: bool = False):
        if self.kalman_filter is None or self.mean_gpu is None: 
            # Try to activate if mean_gpu was none (e.g. if it failed init before but now gets a chance)
            if self.kalman_filter: self.activate(frame_id); return # Attempt re-activation as new if state was bad
            print(f"Warning: GPUSTrack {self.track_id} re_activate on uninit track."); return

        measurement_xyah_gpu = self.tlwh_to_xyah_gpu(new_det_gpustrack._tlwh_gpu)
        updated_mean, updated_cov = self.kalman_filter.update(self.mean_gpu, self.covariance_gpu, measurement_xyah_gpu)
        if torch.isnan(updated_mean).any() or torch.isinf(updated_mean).any():
            print(f"  WARNING: GPUSTrack {self.track_id} - KF update in re_activate NaN/Inf. State not updated.")
        else: self.mean_gpu, self.covariance_gpu = updated_mean, updated_cov
        self.tracklet_len = 0; self.state = TrackState.Tracked; self.is_activated = True
        self.frame_id = frame_id
        if new_id: self.track_id = self.next_id()
        self.score = new_det_gpustrack.score; self.cls = new_det_gpustrack.cls

    def update(self, new_det_gpustrack, frame_id: int):
        if self.kalman_filter is None or self.mean_gpu is None:
            print(f"Warning: GPUSTrack {self.track_id} update on uninit track."); self.state=TrackState.Lost; self.is_activated=False; return
        self.frame_id = frame_id; self.tracklet_len += 1
        measurement_xyah_gpu = self.tlwh_to_xyah_gpu(new_det_gpustrack._tlwh_gpu)
        updated_mean, updated_cov = self.kalman_filter.update(self.mean_gpu, self.covariance_gpu, measurement_xyah_gpu)
        if torch.isnan(updated_mean).any() or torch.isinf(updated_mean).any():
            print(f"  WARNING: GPUSTrack {self.track_id} - KF update in 'update' NaN/Inf. State not updated.")
        else: self.mean_gpu, self.covariance_gpu = updated_mean, updated_cov
        self.state = TrackState.Tracked; self.is_activated = True
        self.score = new_det_gpustrack.score; self.cls = new_det_gpustrack.cls

    @property
    def tlwh_gpu(self) -> torch.Tensor:
        if self.mean_gpu is None: return self._tlwh_gpu.clone() if hasattr(self, '_tlwh_gpu') and self._tlwh_gpu is not None else torch.zeros(4, device=self.device)
        mean_xyah = self.mean_gpu[:4]
        if torch.isnan(mean_xyah).any() or torch.isinf(mean_xyah).any():
            print(f"Warning: Track {self.track_id} tlwh_gpu - mean has NaN/Inf. Returning initial _tlwh_gpu.")
            return self._tlwh_gpu.clone() if hasattr(self, '_tlwh_gpu') and self._tlwh_gpu is not None else torch.zeros(4, device=self.device)
        h = mean_xyah[3]; aspect_ratio = mean_xyah[2]
        if h < 1e-6 : w = torch.tensor(0.0, device=self.device)
        elif torch.isinf(aspect_ratio) or torch.isnan(aspect_ratio) or aspect_ratio < 1e-6 : w = torch.tensor(0.0, device=self.device)
        else: w = aspect_ratio * h
        x1 = mean_xyah[0] - w / 2.0; y1 = mean_xyah[1] - h / 2.0
        return torch.stack([x1, y1, w, h])

    @property
    def xyxy_gpu(self) -> torch.Tensor:
        tlwh_val = self.tlwh_gpu
        xyxy_val = tlwh_val.clone(); xyxy_val[2] = tlwh_val[0] + tlwh_val[2]; xyxy_val[3] = tlwh_val[1] + tlwh_val[3]
        return xyxy_val

    @property
    def result_cpu_numpy(self):
        if not self.is_activated or self.mean_gpu is None: return [0.0,0.0,0.0,0.0, self.track_id, 0.0, -1]
        xyxy_list_cpu = self.xyxy_gpu.cpu().tolist()
        return xyxy_list_cpu + [self.track_id, self.score, self.cls]

    @property
    def xywha(self): # Kept from user file, but note angle is always None in GPUSTrack
        if self.angle is None: print("Warning: GPUSTrack `angle` attr is None, returning `xywh` from GPU state.")
        tlwh_val_cpu = self.tlwh_gpu.cpu().numpy()
        xywh_val_cpu = np.copy(tlwh_val_cpu); xywh_val_cpu[:2] += xywh_val_cpu[2:] / 2.0
        return xywh_val_cpu # If angle was GPU tensor: np.concatenate([xywh_val_cpu, self.angle.cpu().numpy()[None]])
        
    def __repr__(self): return f"GPU_OT_{self.track_id}_({self.start_frame}-{self.end_frame}) (Act:{self.is_activated} St:{self.state})"


class BYTETracker:
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = [] # List[GPUSTrack]
        self.lost_stracks = []    # List[GPUSTrack]
        self.removed_stracks = [] # List[GPUSTrack]
        self.frame_id = 0
        self.args = args 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"BYTETracker Initializing on device: {self.device} with args: "
              f"track_high_thresh={getattr(self.args, 'track_high_thresh', 'N/A')}, "
              f"new_track_thresh={getattr(self.args, 'new_track_thresh', 'N/A')}")
        self.kalman_filter = GPUKalmanFilterXYAH(device=self.device)
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        GPUSTrack.reset_id()

    def update(self, results_boxes_gpu, img=None, feats=None):
        self.frame_id += 1
        cur_activated_stracks, cur_refind_stracks, cur_lost_stracks, cur_removed_stracks = [], [], [], []

        if results_boxes_gpu.shape[0] == 0: # No detections from YOLO
            # Manage existing tracks
            tracks_to_pop_from_lost = []
            for i, track in enumerate(self.lost_stracks):
                track.predict() # Important to advance its internal frame count idea for max_time_lost
                if self.frame_id - track.end_frame > self.max_time_lost: # track.end_frame is last matched frame
                    track.mark_removed(); cur_removed_stracks.append(track); tracks_to_pop_from_lost.append(i)
            for i in sorted(tracks_to_pop_from_lost, reverse=True): self.lost_stracks.pop(i)
            
            tracks_to_move_to_lost_from_tracked = []
            for track in self.tracked_stracks: # Any currently tracked become lost if no detections to match
                track.predict() # Predict to update state
                track.mark_lost(); cur_lost_stracks.append(track); tracks_to_move_to_lost_from_tracked.append(track)
            for t in tracks_to_move_to_lost_from_tracked: self.tracked_stracks.remove(t) # Now empty

            self.lost_stracks.extend(cur_lost_stracks) # Consolidate
            self.removed_stracks.extend(cur_removed_stracks)
            # No new STracks to create, no matching to perform.
            output_tracks = [x for x in self.tracked_stracks if x.is_activated] # Will be empty
            return np.asarray([x.result_cpu_numpy for x in output_tracks], dtype=np.float32) if output_tracks else np.empty((0, 7), dtype=np.float32)

        scores_gpu = results_boxes_gpu.conf
        bboxes_orig_gpu = (results_boxes_gpu.xywhr if hasattr(results_boxes_gpu, "xywhr") else results_boxes_gpu.xywh).clone()
        cls_gpu = results_boxes_gpu.cls.clone()
        
        is_high_confidence = scores_gpu >= self.args.track_high_thresh
        is_above_low_threshold = scores_gpu > self.args.track_low_thresh
        is_below_high_threshold = scores_gpu < self.args.track_high_thresh
        inds_second_tier_bool_gpu = is_above_low_threshold & is_below_high_threshold

        try:
            dets_xywh_high_conf_gpu = bboxes_orig_gpu[is_high_confidence]
            scores_high_conf_gpu = scores_gpu[is_high_confidence]
            cls_high_conf_gpu = cls_gpu[is_high_confidence]
            dets_xywh_second_tier_gpu = bboxes_orig_gpu[inds_second_tier_bool_gpu]
            scores_second_tier_gpu = scores_gpu[inds_second_tier_bool_gpu]
            cls_second_tier_gpu = cls_gpu[inds_second_tier_bool_gpu]
        except IndexError: # Fallback if indexing still fails (should be rare with torch)
            print(f"CRITICAL PYTORCH INDEXING ERROR in BYTETracker.update (GPU). Falling back.") # Less verbose
            num_cols_bbox = bboxes_orig_gpu.shape[1]
            empty_args = {'device': self.device, 'dtype': bboxes_orig_gpu.dtype}
            s_empty_args = {'device': self.device, 'dtype': scores_gpu.dtype}; c_empty_args = {'device': self.device, 'dtype': cls_gpu.dtype}
            dets_xywh_high_conf_gpu = torch.empty((0, num_cols_bbox), **empty_args); scores_high_conf_gpu = torch.empty((0,), **s_empty_args); cls_high_conf_gpu = torch.empty((0,), **c_empty_args)
            dets_xywh_second_tier_gpu = torch.empty((0, num_cols_bbox), **empty_args); scores_second_tier_gpu = torch.empty((0,), **s_empty_args); cls_second_tier_gpu = torch.empty((0,), **c_empty_args)
        
        detections_high_gpu_stracks = self.init_gpu_track(dets_xywh_high_conf_gpu, scores_high_conf_gpu, cls_high_conf_gpu)
        detections_second_tier_gpu_stracks = self.init_gpu_track(dets_xywh_second_tier_gpu, scores_second_tier_gpu, cls_second_tier_gpu)
        
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        current_tracked_only_stracks = [t for t in self.tracked_stracks if t.is_activated]
        strack_pool = self.joint_stracks(current_tracked_only_stracks, self.lost_stracks)
        GPUSTrack.multi_predict(strack_pool)
        
        if hasattr(self, "gmc") and img is not None and dets_xywh_high_conf_gpu.shape[0] > 0 :
            print("Warning: GMC processing skipped in GPU ported tracker (not fully ported).") # GMC not ported

        dists_high = self.get_dists_gpu(strack_pool, detections_high_gpu_stracks)
        matches_high, u_track_pool1, u_detection_high = matching.linear_assignment(dists_high, thresh=self.args.match_thresh)

        for itracked, idet in matches_high:
            track = strack_pool[itracked]; det = detections_high_gpu_stracks[idet]
            if track.state == TrackState.Tracked: track.update(det, self.frame_id)
            else: track.re_activate(det, self.frame_id, new_id=False)
            if track.is_activated: cur_activated_stracks.append(track)

        tracks_for_second_match = [strack_pool[i] for i in u_track_pool1 if strack_pool[i].state == TrackState.Tracked]
        if tracks_for_second_match and detections_second_tier_gpu_stracks:
            dists_low = self.get_dists_gpu(tracks_for_second_match, detections_second_tier_gpu_stracks)
            matches_low, u_track_pool2, u_detection_second = matching.linear_assignment(dists_low, thresh=0.5)
            for itracked, idet in matches_low:
                track = tracks_for_second_match[itracked]; det = detections_second_tier_gpu_stracks[idet]
                track.update(det, self.frame_id)
                if track.is_activated: cur_activated_stracks.append(track)
            for it in u_track_pool2: track = tracks_for_second_match[it]; track.mark_lost(); cur_lost_stracks.append(track)
        else:
            for track in tracks_for_second_match: track.mark_lost(); cur_lost_stracks.append(track)
        
        unmatched_high_after_first_match = [detections_high_gpu_stracks[i] for i in u_detection_high]
        if unconfirmed and unmatched_high_after_first_match:
            dists_unconfirmed = self.get_dists_gpu(unconfirmed, unmatched_high_after_first_match)
            matches_unconfirmed, u_unconfirmed, u_detection_high_after_unconfirmed = matching.linear_assignment(dists_unconfirmed, thresh=0.7)
            for itracked, idet in matches_unconfirmed:
                unconfirmed[itracked].update(unmatched_high_after_first_match[idet], self.frame_id)
                if unconfirmed[itracked].is_activated: cur_activated_stracks.append(unconfirmed[itracked])
            for it in u_unconfirmed: track = unconfirmed[it]; track.mark_removed(); cur_removed_stracks.append(track)
        else:
            for track in unconfirmed: track.mark_removed(); cur_removed_stracks.append(track)
            u_detection_high_after_unconfirmed = list(range(len(unmatched_high_after_first_match)))
            
        for inew_idx in u_detection_high_after_unconfirmed:
            track_candidate = unmatched_high_after_first_match[inew_idx]
            new_track_threshold_val = getattr(self.args, 'new_track_thresh', self.args.track_high_thresh + 0.1)
            if track_candidate.score < new_track_threshold_val: continue
            track_candidate.activate(self.frame_id)
            if track_candidate.is_activated: cur_activated_stracks.append(track_candidate)
            else: cur_removed_stracks.append(track_candidate) 
            
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed(); cur_removed_stracks.append(track)
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked and t not in cur_lost_stracks and t not in cur_removed_stracks]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, cur_activated_stracks)
        # Refind_stracks was populated if a lost track was matched with high-conf.
        # The original code had refind_stracks from high-conf matching. This version adds to cur_activated_stracks directly.
        # For simplicity, refind_stracks_cur_frame isn't heavily used here but kept if needed by other logic.

        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks) 
        self.lost_stracks.extend(cur_lost_stracks)
        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost and t not in cur_removed_stracks]
        self.removed_stracks.extend(cur_removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        if len(self.removed_stracks) > 1000: self.removed_stracks = self.removed_stracks[len(self.removed_stracks) - 999:]

        output_tracks = [x for x in self.tracked_stracks if x.is_activated]
        # Optional: Periodic debug for final output tracks
        # if self.frame_id % 60 == 0 and output_tracks: print(f"  Frame {self.frame_id} Outputting {len(output_tracks)} tracks.")

        return np.asarray([x.result_cpu_numpy for x in output_tracks], dtype=np.float32) if output_tracks else np.empty((0, 7), dtype=np.float32)

    def init_gpu_track(self, dets_xywh_gpu, scores_gpu, cls_gpu):
        tracks = []
        if dets_xywh_gpu.shape[0] > 0:
            tlwh_dets_gpu = dets_xywh_gpu[..., :4].clone()
            tlwh_dets_gpu[..., 0] = dets_xywh_gpu[..., 0] - dets_xywh_gpu[..., 2] / 2.0
            tlwh_dets_gpu[..., 1] = dets_xywh_gpu[..., 1] - dets_xywh_gpu[..., 3] / 2.0
            for i in range(dets_xywh_gpu.shape[0]):
                initial_tlwh_np_for_strack = tlwh_dets_gpu[i].cpu().numpy()
                score_val = scores_gpu[i].item() 
                cls_val = cls_gpu[i].item()     
                tracks.append(GPUSTrack(initial_tlwh_np_for_strack, score_val, cls_val, self.device, self.kalman_filter))
        return tracks
    
    def get_dists_gpu(self, tracks1: list, tracks2: list) -> np.ndarray:
        cost_matrix_cpu = matching.iou_distance(tracks1, tracks2) 
        if hasattr(self.args, 'fuse_score') and self.args.fuse_score: # Check if fuse_score exists
            cost_matrix_cpu = matching.fuse_score(cost_matrix_cpu, tracks2)
        return cost_matrix_cpu

    def get_kalmanfilter(self): return GPUKalmanFilterXYAH(device=self.device)
    def multi_predict(self, tracks): GPUSTrack.multi_predict(tracks)
    @staticmethod
    def reset_id(): GPUSTrack.reset_id()
    def reset(self):
        self.tracked_stracks, self.lost_stracks, self.removed_stracks = [], [], []
        self.frame_id = 0; self.kalman_filter = GPUKalmanFilterXYAH(device=self.device); self.reset_id()
    
    @staticmethod
    def joint_stracks(tlista, tlistb):
        exists = {}; res = []
        for t in tlista: exists[t.track_id] = 1; res.append(t)
        for t in tlistb:
            if not exists.get(t.track_id, 0): exists[t.track_id] = 1; res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        if not stracksa or not stracksb: return list(stracksa), list(stracksb)
        pdist = matching.iou_distance(stracksa, stracksb) 
        if pdist.size == 0: return list(stracksa), list(stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq: dupb.append(q)
            else: dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb