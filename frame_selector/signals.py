"""Stages 2-3: track the detections, then flag frames where the model disagrees with its own tracks.

A candidate is a dict: frame, reason, score, box (model space), track (id or None), note.
Reasons: 'fn' (missed), 'fp' (false positive / hard negative), 'class' (wrong class), 'borderline'.
"""
from collections import Counter, defaultdict

import numpy as np

from tracking.ocsort import OCSort, iou_batch

MODEL_SIZE = 640
TRACK_CONF = 0.25          # production threshold: tracks are built from what production would see
MIN_TRACK_HITS = 3         # fewer hits than this is an orphan (likely false positive)
MIN_GAP_TRACK_HITS = 10    # a track must be this established before its gaps count as misses
MAX_GAP_S = 1.0
STATIC_MIN_S = 2.0
BIG_BOX = 60               # px tall in model space: large enough to label with confidence
CLASS_POWER = 4.0


class Track:
    __slots__ = ('tid', 'frames', 'boxes', 'confs', 'classes', 'majority')

    def __init__(self, tid):
        self.tid = tid
        self.frames, self.boxes, self.confs, self.classes = [], [], [], []
        self.majority = None

    @property
    def hits(self):
        return len(self.frames)

    def heights(self):
        b = np.asarray(self.boxes)
        return b[:, 3] - b[:, 1]

    def finish(self):
        votes = Counter()
        for c, s, h in zip(self.classes, self.confs, self.heights()):
            votes[c] += s * max(h, 1.0) ** CLASS_POWER
        self.majority = votes.most_common(1)[0][0]

    def box_at(self, frame):
        """Observed box at that frame, or the interpolation between the neighbouring hits, or None."""
        i = np.searchsorted(self.frames, frame)
        if i < len(self.frames) and self.frames[i] == frame:
            return np.asarray(self.boxes[i], dtype=float)
        if i == 0 or i >= len(self.frames):
            return None
        f0, f1 = self.frames[i - 1], self.frames[i]
        w = (frame - f0) / (f1 - f0)
        return np.asarray(self.boxes[i - 1], dtype=float) * (1 - w) + np.asarray(self.boxes[i], dtype=float) * w


def build_tracks(frames, offs, dets, fps):
    """Run OC-SORT with a long memory over the production-confidence detections."""
    tracker = OCSort(det_thresh=TRACK_CONF, low_thresh=0.1, max_age=int(2.0 * fps), iou_threshold=0.2)
    tracks = {}
    for i, f in enumerate(frames):
        d = dets[offs[i]:offs[i + 1]]
        out = tracker.update(d[d[:, 4] >= 0.1]) if len(d) else tracker.update(np.zeros((0, 6)))
        for row in out:
            tr = tracks.setdefault(int(row[4]), Track(int(row[4])))
            tr.frames.append(int(f))
            tr.boxes.append(row[:4].copy())
            tr.confs.append(float(row[5]))
            tr.classes.append(int(row[6]))
    for tr in tracks.values():
        tr.finish()
    return tracks


def _clipped(box):
    return box[0] <= 1 or box[1] <= 1 or box[2] >= MODEL_SIZE - 2 or box[3] >= MODEL_SIZE - 2


def _size_term(box):
    return min(box[3] - box[1], 200.0) / 200.0


def find_candidates(frames, offs, dets, tracks, fps, names=None):
    """names: class id -> class name, used in the notes shown to the reviewer."""
    by_frame = {int(f): i for i, f in enumerate(frames)}
    name = lambda c: (names or {}).get(int(c), f"class {int(c)}")   # noqa: E731
    out = []

    def dets_of(frame, min_conf=0.0):
        i = by_frame.get(frame)
        if i is None:
            return np.zeros((0, 6))
        d = dets[offs[i]:offs[i + 1]]
        return d[d[:, 4] >= min_conf]

    # Which detections ended up in an established track? Everything else at production confidence is an orphan.
    tracked = defaultdict(set)
    for tr in tracks.values():
        if tr.hits >= MIN_TRACK_HITS:
            for f, b in zip(tr.frames, tr.boxes):
                tracked[f].add(tuple(np.round(np.asarray(b, np.float32), 1)))
    for f in by_frame:
        for d in dets_of(f, TRACK_CONF):
            if tuple(np.round(np.asarray(d[:4], np.float32), 1)) not in tracked[f] and d[3] - d[1] >= 20:
                out.append(dict(frame=f, reason='fp', score=0.5 * d[4] + 0.5 * _size_term(d), box=d[:4].copy(),
                                track=None, note=f"orphan detection, conf {d[4]:.2f}"))

    for tr in tracks.values():
        if tr.hits < MIN_TRACK_HITS:
            continue
        heights = tr.heights()
        # Static object read as a vehicle (bush, pole, parked trailer): hard negative
        span_s = (tr.frames[-1] - tr.frames[0]) / fps
        if span_s >= STATIC_MIN_S:
            c = np.asarray(tr.boxes)
            cx, cy = (c[:, 0] + c[:, 2]) / 2, (c[:, 1] + c[:, 3]) / 2
            if np.hypot(cx.max() - cx.min(), cy.max() - cy.min()) < np.median(heights):
                k = int(np.argmax(tr.confs))
                out.append(dict(frame=tr.frames[k], reason='fp', score=0.6 + 0.4 * _size_term(tr.boxes[k]),
                                box=np.asarray(tr.boxes[k]), track=tr.tid, note=f"static for {span_s:.0f}s"))
                continue
        # Class disagreement: the frame where the vehicle is largest but the class differs from the track majority
        dis = [k for k, c in enumerate(tr.classes) if c != tr.majority and heights[k] >= 0.6 * heights.max()]
        if dis and tr.hits >= MIN_GAP_TRACK_HITS:
            k = max(dis, key=lambda k: heights[k] * tr.confs[k])
            share = sum(1 for c in tr.classes if c != tr.majority) / tr.hits
            out.append(dict(frame=tr.frames[k], reason='class', score=0.4 + 0.3 * _size_term(tr.boxes[k]) + 0.3 * share,
                            box=np.asarray(tr.boxes[k]), track=tr.tid,
                            note=f"this frame says {name(tr.classes[k])}, the track is mostly {name(tr.majority)} ({share:.0%} of frames disagree)"))
        # Borderline confidence on a big box
        k = int(np.argmin(np.where(heights >= BIG_BOX, tr.confs, 9)))
        if heights[k] >= BIG_BOX and tr.confs[k] < 0.5:
            out.append(dict(frame=tr.frames[k], reason='borderline', score=0.3 + 0.4 * (0.5 - tr.confs[k]) / 0.25 + 0.3 * _size_term(tr.boxes[k]),
                            box=np.asarray(tr.boxes[k]), track=tr.tid, note=f"conf {tr.confs[k]:.2f} on a {heights[k]:.0f}px box"))
        # Gaps: the vehicle is between two sightings but nothing was detected near where it must be
        if tr.hits >= MIN_GAP_TRACK_HITS:
            for f0, f1 in zip(tr.frames, tr.frames[1:]):
                if 1 < f1 - f0 <= MAX_GAP_S * fps:
                    mid = (f0 + f1) // 2
                    box = tr.box_at(mid)
                    if box is None or _clipped(box):
                        continue
                    weak = dets_of(mid)
                    ov = iou_batch(box[None], weak[:, :4]).max() if len(weak) else 0.0
                    if ov < 0.3:
                        out.append(dict(frame=mid, reason='fn', score=0.5 + 0.5 * _size_term(box), box=box, track=tr.tid,
                                        note=f"not detected for {f1 - f0 - 1} frames mid-track"))
            # The track survived on a detection production would have dropped (conf < 0.25): a near miss
            weak = [k for k, c in enumerate(tr.confs) if c < TRACK_CONF and not _clipped(tr.boxes[k])]
            if weak:
                k = min(weak, key=lambda k: tr.confs[k])
                out.append(dict(frame=tr.frames[k], reason='fn', score=0.3 + 0.5 * _size_term(tr.boxes[k]), box=np.asarray(tr.boxes[k]),
                                track=tr.tid, note=f"only a weak detection (conf {tr.confs[k]:.2f}) mid-track"))

    # Two classes on one object in one frame
    for f in by_frame:
        d = dets_of(f, TRACK_CONF)
        if len(d) < 2:
            continue
        iou = iou_batch(d[:, :4], d[:, :4])
        for a in range(len(d)):
            for b in range(a + 1, len(d)):
                if iou[a, b] >= 0.7 and d[a, 5] != d[b, 5] and max(d[a, 3] - d[a, 1], d[b, 3] - d[b, 1]) >= BIG_BOX:
                    k = a if d[a, 4] >= d[b, 4] else b
                    out.append(dict(frame=f, reason='class', score=0.3 + 0.4 * _size_term(d[k]) + 0.3 * min(d[a, 4], d[b, 4]),
                                    box=d[k, :4].copy(), track=None, note=f"{name(d[a, 5])} and {name(d[b, 5])} boxes on one object"))
    return out


def pseudo_labels(frame, tracks, extra=()):
    """(class, box) for every established track present at the frame (interpolated through gaps), plus extras."""
    out = []
    for tr in tracks.values():
        if tr.hits < MIN_TRACK_HITS or frame < tr.frames[0] or frame > tr.frames[-1]:
            continue
        box = tr.box_at(frame)
        if box is not None:
            out.append((tr.majority, box))
    out.extend(extra)
    return out
