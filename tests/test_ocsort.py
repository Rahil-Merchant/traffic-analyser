"""Unit tests for the OC-SORT tracker."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tracking.ocsort import OCSort


def det(x, y, w=40, h=40, score=0.9, cls=0):
    return [x, y, x + w, y + h, score, cls]


def run(tracker, frames):
    """Feed frames of detections; returns the tracker output per frame."""
    return [tracker.update(np.array(f, dtype=float).reshape(-1, 6)) for f in frames]


def ids(outputs):
    return {int(i) for out in outputs for i in out[:, 4]}


def test_steady_motion_keeps_one_id():
    assert ids(run(OCSort(), [[det(10 + 5 * i, 100)] for i in range(30)])) == {1}


def test_fast_motion_without_box_overlap_keeps_id():
    # 60 px per frame with 40 px boxes: consecutive detections never overlap
    assert ids(run(OCSort(), [[det(10 + 60 * i, 100)] for i in range(10)])) == {1}


def test_fast_motion_without_buffer_fragments():
    assert len(ids(run(OCSort(buffer_scales=()), [[det(10 + 60 * i, 100)] for i in range(10)]))) > 1


def test_detection_gap_resumes_same_id():
    frames = [[det(10 + 8 * i, 100)] if not 10 <= i < 15 else [] for i in range(30)]
    out = run(OCSort(max_age=10), frames)
    assert ids(out) == {1}
    assert len(out[12]) == 0                     # nothing reported while unseen


def test_crossing_objects_keep_their_ids():
    frames = [[det(10 + 10 * i, 100), det(400 - 10 * i, 110, score=0.85)] for i in range(40)]
    out = run(OCSort(), frames)
    x_first = {int(r[4]): r[0] for r in out[0]}
    x_last = {int(r[4]): r[0] for r in out[-1]}
    assert set(x_first) == set(x_last) == {1, 2}
    assert x_last[1] > x_first[1] and x_last[2] < x_first[2]   # each ID keeps its own direction


def test_low_score_detection_continues_but_does_not_start_tracks():
    trk = OCSort(det_thresh=0.5, low_thresh=0.1)
    assert len(trk.update(np.array([det(10, 10, score=0.3)]))) == 0
    trk.update(np.array([det(100, 100, score=0.9)]))
    out = trk.update(np.array([det(104, 100, score=0.3)]))
    assert list(out[:, 4].astype(int)) == [1]
