# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np

from tracking.tracker_utils import xywh2ltwh
from tracking.tracker_basetrack import BaseTrack, TrackState
import tracking.tracker_matching as matching
from tracking.tracker_utils import KalmanFilterXYAH

TARGET_DEBUG_TRACK_ID = 136

class STrack(BaseTrack):
    """
    Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (Any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.

    Methods:
        predict(): Predict the next state of the object using Kalman filter.
        multi_predict(stracks): Predict the next states for multiple tracks.
        multi_gmc(stracks, H): Update multiple track states using a homography matrix.
        activate(kalman_filter, frame_id): Activate a new tracklet.
        re_activate(new_track, frame_id, new_id): Reactivate a previously lost tracklet.
        update(new_track, frame_id): Update the state of a matched track.
        convert_coords(tlwh): Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah(tlwh): Convert tlwh bounding box to xyah format.

    Examples:
        Initialize and activate a new track
        >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh, score, cls):
        """
        Initialize a new STrack instance.

        Args:
            xywh (List[float]): Bounding box coordinates and dimensions in the format (x, y, w, h, [a], idx), where
                (x, y) is the center, (w, h) are width and height, [a] is optional aspect ratio, and idx is the id.
            score (float): Confidence score of the detection.
            cls (Any): Class label for the detected object.

        Examples:
            >>> xywh = [100.0, 150.0, 50.0, 75.0, 1]
            >>> score = 0.9
            >>> cls = "person"
            >>> track = STrack(xywh, score, cls)
        """
        super().__init__()
        # xywh+idx or xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

    def predict(self):
        """Predicts the next state (mean and covariance) of the object using the Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix for multiple tracks."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        global TARGET_DEBUG_TRACK_ID
        if self.track_id == TARGET_DEBUG_TRACK_ID or (new_id and self.next_id_val() -1 == TARGET_DEBUG_TRACK_ID): # Check if it's about to become the target
            print(f"DEBUG_BYTE: STrack ID {self.track_id} RE-ACTIVATE called. Frame: {frame_id}. New Score: {new_track.score:.2f}. State before: {self.state}")
        
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx
        if self.track_id == TARGET_DEBUG_TRACK_ID:
             print(f"DEBUG_BYTE: STrack ID {self.track_id} RE-ACTIVATE complete. State after: {self.state}. New ID assigned: {new_id}")
    def update(self, new_track, frame_id):
        global TARGET_DEBUG_TRACK_ID
        if self.track_id == TARGET_DEBUG_TRACK_ID:
            print(f"DEBUG_BYTE: STrack ID {self.track_id} UPDATE called. Frame: {frame_id}. New Score: {new_track.score:.2f}. State before: {self.state}")
        
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx
        if self.track_id == TARGET_DEBUG_TRACK_ID:
            print(f"DEBUG_BYTE: STrack ID {self.track_id} UPDATE complete. State after: {self.state}")

    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Returns the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self):
        """Converts bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self):
        """Returns the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self):
        """Returns position in (center x, center y, width, height, angle) format, warning if angle is missing."""
        if self.angle is None:
            print("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self):
        """Returns the current tracking results: x1,y1,x2,y2, track_id, score, class_id, original_detection_idx."""
        # Always use xyxy for the primary bounding box output for consistency
        coords_xyxy = self.xyxy.tolist() # This is 4 elements
        return coords_xyxy + [self.track_id, self.score, self.cls, self.idx] # Total 4 + 1 + 1 + 1 + 1 = 8 elements

    def __repr__(self):
        """Returns a string representation of the STrack object including start frame, end frame, and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """
    BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.

    This class encapsulates the functionality for initializing, updating, and managing the tracks for detected objects in a
    video sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman filtering for
    predicting the new object locations, and performs data association.

    Attributes:
        tracked_stracks (List[STrack]): List of successfully activated tracks.
        lost_stracks (List[STrack]): List of lost tracks.
        removed_stracks (List[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (Namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (KalmanFilterXYAH): Kalman Filter object.

    Methods:
        update(results, img=None): Updates object tracker with new detections.
        get_kalmanfilter(): Returns a Kalman filter object for tracking bounding boxes.
        init_track(dets, scores, cls, img=None): Initialize object tracking with detections.
        get_dists(tracks, detections): Calculates the distance between tracks and detections.
        multi_predict(tracks): Predicts the location of tracks.
        reset_id(): Resets the ID counter of STrack.
        joint_stracks(tlista, tlistb): Combines two lists of stracks.
        sub_stracks(tlista, tlistb): Filters out the stracks present in the second list from the first list.
        remove_duplicate_stracks(stracksa, stracksb): Removes duplicate stracks based on IoU.

    Examples:
        Initialize BYTETracker and update with detection results
        >>> tracker = BYTETracker(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """

    def __init__(self, args, frame_rate=30):
        """
        Initialize a BYTETracker instance for object tracking.

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video sequence.

        Examples:
            Initialize BYTETracker with command-line arguments and a frame rate of 30
            >>> args = Namespace(track_buffer=30)
            >>> tracker = BYTETracker(args, frame_rate=30)
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(self, results, img=None, feats=None): # 'results' here is the Ultralytics Results object
        """Updates the tracker with new detections and returns the current list of tracked objects."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # --- CORRECTED SECTION START ---
        if results.boxes is None or len(results.boxes) == 0:
            # No detections in this frame, handle accordingly
            # This logic will effectively make all existing tracks 'lost' for this frame
            # or keep them as they are if already lost.
            # The rest of the function will execute with empty detection lists.
            scores = np.array([])
            bboxes = np.array([]).reshape(0, 5) # Assuming xywh + idx format later
            cls = np.array([])
        else:
            scores = results.boxes.conf.cpu().numpy()  # Access .boxes.conf
            # ByteTrack expects xywh, Ultralytics Results has .boxes.xywh
            # The original ByteTrack code also adds an index to the bboxes
            # which is used in STrack's __init__ if the input is [x,y,w,h,idx]
            # For STrack(xywh, score, cls), xywh is expected to be [x,y,w,h,idx] for init_track
            # or [x,y,w,h,a,idx] if rotated boxes (xywhr)
            
            # Check if rotated boxes are present, else use standard xywh
            if hasattr(results.boxes, "xywhr") and results.boxes.xywhr is not None and results.boxes.xywhr.shape[0] > 0 :
                # Assuming results.boxes.xywhr is [N, 5] (x,y,w,h,angle)
                raw_bboxes_xywha = results.boxes.xywhr.cpu().numpy()
                # Add index for STrack initialization
                bboxes = np.concatenate([raw_bboxes_xywha, np.arange(len(raw_bboxes_xywha)).reshape(-1, 1)], axis=-1) # Shape (N, 6) -> x,y,w,h,a,idx
            elif hasattr(results.boxes, "xywh") and results.boxes.xywh is not None and results.boxes.xywh.shape[0] > 0:
                raw_bboxes_xywh = results.boxes.xywh.cpu().numpy()
                # Add index for STrack initialization
                bboxes = np.concatenate([raw_bboxes_xywh, np.arange(len(raw_bboxes_xywh)).reshape(-1, 1)], axis=-1) # Shape (N, 5) -> x,y,w,h,idx
            else: # Fallback if no recognized box format, or empty boxes
                bboxes = np.array([]).reshape(0,5) # Ensure it's (0,5) or (0,6) if you need placeholder for angle
            
            cls = results.boxes.cls.cpu().numpy()    # Access .boxes.cls
        # --- CORRECTED SECTION END ---


        # The rest of the update method remains the same, as it uses these `scores`, `bboxes`, `cls`
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh # This should be scores >= self.args.track_high_thresh

        # Correct logic for high and second tier detections
        # inds_high was previously defined for scores *below* high_thresh, which is wrong for primary dets
        # remain_inds IS for high confidence detections
        # inds_second is for low confidence detections that are not high confidence

        # Corrected logic for selecting detections:
        inds_second = inds_low & (~remain_inds) # Low confidence detections that are NOT high confidence
                                                 # The original ByteTrack had `inds_high` which was scores < self.args.track_high_thresh
                                                 # So `inds_low & inds_high` meant scores between low_thresh and high_thresh.
                                                 # The `~remain_inds` (not high confidence) achieves the same here when combined with `inds_low`.


        dets_second = bboxes[inds_second]    # Detections for second association (low score)
        dets = bboxes[remain_inds]           # Detections for first association (high score)
        
        scores_keep = scores[remain_inds]    # Scores for high score detections
        scores_second = scores[inds_second]  # Scores for low score detections
        
        cls_keep = cls[remain_inds]          # Classes for high score detections
        cls_second = cls[inds_second]        # Classes for low score detections

        # Ensure img is not None if feats is None and GMC is potentially used by STrack (though GMC is optional)
        # For init_track, img or feats is primarily for ReID features, which base ByteTrack doesn't use deeply.
        current_input_for_init = img if feats is None and hasattr(self, "gmc") else None # Pass img only if GMC might be used
                                                                                        # Or pass None if not using ReID features in STrack
        
        # If STrack's init_track uses img for something other than features (e.g. GMC related)
        # and you are not using ReID features, pass img. If init_track ONLY uses it for features
        # that ByteTrack isn't using, you can pass None.
        # For safety, let's assume img might be used by some STrack logic.
        detections = self.init_track(dets, scores_keep, cls_keep, img=img) # Pass img

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        # Step 2: First association, with high score detection boxes
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool) # STrack.multi_predict(strack_pool) if static method preferred for clarity

        # GMC (Global Motion Compensation) - Optional
        if hasattr(self, "gmc") and img is not None and dets.shape[0] > 0: # Ensure dets is not empty for warp
            warp = self.gmc.apply(img, dets) # Pass high-score detections for warp calculation
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp) # Also apply to unconfirmed if they exist

        dists = self.get_dists(strack_pool, detections) # Uses IoU distance
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet] # det is an STrack object from init_track
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 3: Second association, with low score detection boxes
        # detections_second = self.init_track(dets_second, scores_second, cls_second, img=img) # Pass img
        # Ensure dets_second is not empty before initializing tracks for it.
        detections_second = []
        if dets_second.shape[0] > 0:
            detections_second = self.init_track(dets_second, scores_second, cls_second, img=img)


        # Prepare r_tracked_stracks: u_track contains indices relative to strack_pool
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        dists = matching.iou_distance(r_tracked_stracks, detections_second) # IoU distance
        # Original ByteTrack often uses a less strict IoU for second association (e.g. 0.3 IoU threshold -> cost threshold 0.7)
        # matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5) 
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.7) # TRY 0.7 (cost, so IoU >= 0.3)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked] # itracked is index for r_tracked_stracks
            det = detections_second[idet]    # idet is index for detections_second
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track: # Indices for r_tracked_stracks
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        # u_detection was from the high-score detections that didn't match strack_pool
        detections_remaining_high_score = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections_remaining_high_score) # Uses IoU
        matches, u_unconfirmed, u_detection_high_again = matching.linear_assignment(dists, thresh=0.7) # Original uses 0.7
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_remaining_high_score[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 4: Init new stracks
        # u_detection_high_again are indices from detections_remaining_high_score
        for inew in u_detection_high_again:
            track = detections_remaining_high_score[inew] # This is an STrack object
            if track.score < self.args.new_track_thresh: # Filter by new_track_thresh
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
            
        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Clean up lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        # Clip removed_stracks to prevent excessive memory use (original logic)
        if len(self.removed_stracks) > 1000: # Max 1000 removed tracks to keep
            self.removed_stracks = self.removed_stracks[len(self.removed_stracks) - 1000:]


        # Output only activated tracks
        output_tracks = [x.result for x in self.tracked_stracks if x.is_activated]
        return np.asarray(output_tracks, dtype=np.float32) if output_tracks else np.empty((0, 8), dtype=np.float32) # Ensure (0,8) if empty
    
    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        """
        Initializes STrack objects from detections.
        `dets` here is expected to be in xywh+idx format (N,5) or xywha+idx (N,6).
        """
        if not dets.any(): # Check if dets is empty
            return []
        return [STrack(xywh_or_xywha_plus_idx, s, c) for (xywh_or_xywha_plus_idx, s, c) in zip(dets, scores, cls)]

    def get_dists(self, tracks, detections):
        """Calculates the distance between tracks and detections using IoU and optionally fuses scores."""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        """Predict the next states for multiple tracks using Kalman filter."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Resets the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Resets the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combines two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """Filters out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Removes duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
