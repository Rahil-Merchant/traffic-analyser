# project/tracking/ocsort.py
"""OC-SORT tracker (Cao et al., "Observation-Centric SORT", CVPR 2023).

Kept in-repo so its association stages can be tuned for road traffic. On top of the
standard stages (IoU + velocity-direction consistency, BYTE low-score matching,
observation-centric recovery) there is a buffered-IoU stage (Yang et al., C-BIoU):
boxes are enlarged before matching so a fast vehicle whose consecutive detections
do not overlap still keeps its ID.

update() takes detections as [x1, y1, x2, y2, score, class_id] and returns
[x1, y1, x2, y2, track_id, score, class_id] for tracks observed in this frame.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(a, b):
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]
    b = b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def _expand(boxes, scale):
    """Grow each box by `scale` x its own width/height on every side."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.stack([boxes[:, 0] - scale * w, boxes[:, 1] - scale * h,
                     boxes[:, 2] + scale * w, boxes[:, 3] + scale * h], axis=1)


def _centers(boxes):
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def _assign(score, threshold_mask):
    """Maximise total score; keep only pairs allowed by threshold_mask."""
    if score.size == 0:
        return np.empty((0, 2), dtype=int)
    rows, cols = linear_sum_assignment(-score)
    keep = threshold_mask[rows, cols]
    return np.stack([rows[keep], cols[keep]], axis=1)


class _KalmanBox:
    """Constant-velocity Kalman filter on [cx, cy, area, aspect, vcx, vcy, varea] (SORT parameterisation)."""

    def __init__(self, box):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.zeros((7, 1))
        self.x[:4] = self.to_z(box)

    @staticmethod
    def to_z(box):
        w, h = box[2] - box[0], box[3] - box[1]
        return np.array([[box[0] + w / 2], [box[1] + h / 2], [w * h], [w / max(h, 1e-6)]])

    def box(self):
        s, r = max(self.x[2, 0], 1e-6), max(self.x[3, 0], 1e-6)
        w = np.sqrt(s * r)
        h = s / max(w, 1e-6)
        cx, cy = self.x[0, 0], self.x[1, 0]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, box):
        y = self.to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def snapshot(self):
        return self.x.copy(), self.P.copy()

    def restore(self, snap):
        self.x, self.P = snap[0].copy(), snap[1].copy()


class _Track:
    def __init__(self, box, score, cls, track_id, delta_t):
        self.kf = _KalmanBox(box)
        self.id = track_id
        self.score = score
        self.cls = cls
        self.delta_t = delta_t
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.last_obs = np.asarray(box, dtype=float)
        self.obs = {0: self.last_obs}          # age -> observed box
        self.velocity = None                   # unit direction of observed motion (OCM)
        self.snap = self.kf.snapshot()         # filter state at the last real observation (ORU)
        self.pred = self.last_obs

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.pred = self.kf.box()
        return self.pred

    def previous_obs(self):
        """Observation ~delta_t steps back, for velocity direction; falls back to the oldest kept one."""
        for dt in range(self.delta_t, 0, -1):
            o = self.obs.get(self.age - dt)
            if o is not None:
                return o
        return self.obs[min(self.obs)] if self.obs else None

    def update(self, box, score, cls):
        box = np.asarray(box, dtype=float)
        prev = self.previous_obs()
        if prev is not None:
            d = _centers(box[None])[0] - _centers(prev[None])[0]
            n = np.linalg.norm(d)
            if n > 1e-6:
                self.velocity = d / n
        gap = self.time_since_update
        if gap > 1:
            # Observation-centric re-update: replay the gap with a linear virtual trajectory
            # instead of trusting the drifted prediction.
            self.kf.restore(self.snap)
            for i in range(1, gap):
                self.kf.predict()
                self.kf.update(self.last_obs + (box - self.last_obs) * i / gap)
            self.kf.predict()
        self.kf.update(box)
        self.snap = self.kf.snapshot()
        self.last_obs = box
        self.obs[self.age] = box
        for k in [k for k in self.obs if k < self.age - 2 * self.delta_t]:
            del self.obs[k]
        self.score, self.cls = score, cls
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1


class OCSort:
    def __init__(self, det_thresh=0.25, low_thresh=0.1, max_age=40, min_hits=1, iou_threshold=0.2,
                 delta_t=3, inertia=0.2, use_byte=True, buffer_scales=(0.3, 0.6), buffer_iou_threshold=0.1,
                 buffer_max_age=5):
        self.det_thresh = det_thresh
        self.low_thresh = low_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_byte = use_byte
        self.buffer_scales = tuple(buffer_scales or ())
        self.buffer_iou_threshold = buffer_iou_threshold
        self.buffer_max_age = buffer_max_age
        self.tracks = []
        self.frame_count = 0
        self._next_id = 1

    def reset(self):
        self.tracks, self.frame_count, self._next_id = [], 0, 1

    def _direction_score(self, dets, track_idx):
        """OCM term: how well each det continues each track's observed direction of motion (in [-0.5, 0.5])."""
        score = np.zeros((len(track_idx), len(dets)))
        det_c = _centers(dets[:, :4])
        for r, ti in enumerate(track_idx):
            t = self.tracks[ti]
            prev = t.previous_obs()
            if t.velocity is None or prev is None:
                continue
            d = det_c - _centers(prev[None])[0]
            n = np.linalg.norm(d, axis=1)
            cos = np.clip((d @ t.velocity) / np.maximum(n, 1e-6), -1, 1)
            score[r] = (np.pi / 2 - np.abs(np.arccos(cos))) / np.pi
        return score * dets[None, :, 4]

    def update(self, dets):
        self.frame_count += 1
        dets = np.asarray(dets, dtype=float).reshape(-1, 6)
        for t in self.tracks:
            t.predict()

        high = dets[dets[:, 4] >= self.det_thresh]
        low = dets[(dets[:, 4] >= self.low_thresh) & (dets[:, 4] < self.det_thresh)]
        unmatched_t = list(range(len(self.tracks)))
        unmatched_h = list(range(len(high)))

        def match(det_set, det_idx, cand_boxes, extra=None, thresh=None):
            nonlocal unmatched_t
            if not det_idx or not unmatched_t:
                return []
            ious = iou_batch(cand_boxes, det_set[det_idx, :4])
            score = ious + (extra if extra is not None else 0)
            pairs = _assign(score, ious >= (self.iou_threshold if thresh is None else thresh))
            out = [(unmatched_t[r], det_idx[c]) for r, c in pairs]
            matched_t = {p[0] for p in out}
            unmatched_t = [i for i in unmatched_t if i not in matched_t]
            return out

        # 1) high-score detections: IoU on predictions + velocity-direction consistency
        preds = np.array([self.tracks[i].pred for i in unmatched_t]) if unmatched_t else np.zeros((0, 4))
        ocm = self.inertia * self._direction_score(high[unmatched_h], unmatched_t) if unmatched_t and unmatched_h else None
        for ti, di in match(high, unmatched_h, preds, ocm):
            self.tracks[ti].update(high[di, :4], high[di, 4], int(high[di, 5]))
            unmatched_h.remove(di)

        # 2) BYTE: low-score detections can only continue existing tracks
        if self.use_byte and len(low) and unmatched_t:
            preds = np.array([self.tracks[i].pred for i in unmatched_t])
            for ti, di in match(low, list(range(len(low))), preds):
                self.tracks[ti].update(low[di, :4], low[di, 4], int(low[di, 5]))

        # 3) observation-centric recovery: match against each track's last real observation
        if unmatched_t and unmatched_h:
            last = np.array([self.tracks[i].last_obs for i in unmatched_t])
            for ti, di in match(high, unmatched_h, last):
                self.tracks[ti].update(high[di, :4], high[di, 4], int(high[di, 5]))
                unmatched_h.remove(di)

        # 4) buffered IoU for fast motion: recently seen tracks only, enlarged boxes, no reversals
        for scale in self.buffer_scales:
            recent = [i for i in unmatched_t if self.tracks[i].time_since_update <= self.buffer_max_age]
            if not recent or not unmatched_h:
                break
            saved, unmatched_t = unmatched_t, recent
            preds = _expand(np.array([self.tracks[i].pred for i in recent]), scale)
            cand = high[unmatched_h]
            reverse_ok = np.ones((len(recent), len(unmatched_h)), dtype=bool)
            for r, ti in enumerate(recent):
                t = self.tracks[ti]
                if t.velocity is not None:
                    d = _centers(cand[:, :4]) - _centers(t.last_obs[None])[0]
                    reverse_ok[r] = (d @ t.velocity) >= 0
            ious = iou_batch(preds, _expand(cand[:, :4], scale)) * reverse_ok
            pairs = _assign(ious, ious >= self.buffer_iou_threshold)
            matched = []
            for r, c in pairs:
                ti, di = recent[r], unmatched_h[c]
                self.tracks[ti].update(high[di, :4], high[di, 4], int(high[di, 5]))
                matched.append((ti, di))
            done_t = {m[0] for m in matched}
            done_h = {m[1] for m in matched}
            unmatched_t = [i for i in saved if i not in done_t]
            unmatched_h = [i for i in unmatched_h if i not in done_h]

        # 5) new tracks from leftover confident detections
        for di in unmatched_h:
            self.tracks.append(_Track(high[di, :4], high[di, 4], int(high[di, 5]), self._next_id, self.delta_t))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        out = [np.r_[t.last_obs, t.id, t.score, t.cls] for t in self.tracks
               if t.time_since_update == 0 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)]
        return np.array(out, dtype=float).reshape(-1, 7)
