"""Unit tests for the frame selector's error signals and frame selection (synthetic detections, no model)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from frame_selector.select import budget_for, select_frames
from frame_selector.signals import build_tracks, find_candidates, pseudo_labels

FPS = 20.0


def make_dump(frame_dets):
    """frame_dets: list over frames of lists of [x1,y1,x2,y2,conf,cls] -> (frames, offs, dets)."""
    frames = np.arange(len(frame_dets))
    dets = [np.array(d, dtype=np.float32).reshape(-1, 6) for d in frame_dets]
    offs = np.cumsum([0] + [len(d) for d in dets])
    return frames, offs, np.concatenate(dets) if dets else np.zeros((0, 6), np.float32)


def car(i, cls=2, conf=0.9, x0=100.0):
    """A car driving right at 4 px/frame, 80 px tall."""
    return [x0 + 4 * i, 300.0, x0 + 4 * i + 120, 380.0, conf, cls]


def run(frame_dets):
    frames, offs, dets = make_dump(frame_dets)
    tracks = build_tracks(frames, offs, dets, FPS)
    return tracks, find_candidates(frames, offs, dets, tracks, FPS)


def reasons(cands):
    return {c['reason'] for c in cands}


def test_clean_track_raises_nothing():
    _, cands = run([[car(i)] for i in range(60)])
    assert cands == []


def test_gap_in_a_track_is_a_missed_vehicle():
    tracks, cands = run([[car(i)] if not 30 <= i < 36 else [] for i in range(60)])
    fn = [c for c in cands if c['reason'] == 'fn']
    assert len(fn) == 1 and 30 <= fn[0]['frame'] < 36
    assert np.allclose(fn[0]['box'], car(fn[0]['frame'])[:4], atol=1)       # interpolated where the car must be
    assert len(pseudo_labels(fn[0]['frame'], tracks)) == 1                  # and the pseudo-label fills the gap


def test_weak_detection_in_a_gap_is_still_flagged():
    _, cands = run([[car(i)] if not 30 <= i < 36 else [car(i, conf=0.12)] for i in range(60)])
    assert any(c['reason'] == 'fn' and 'weak' in c['note'] for c in cands)


def test_class_flicker_is_flagged_at_the_largest_disagreeing_frame():
    _, cands = run([[car(i, cls=8 if i in (20, 21, 40) else 2)] for i in range(60)])
    cl = [c for c in cands if c['reason'] == 'class']
    assert len(cl) == 1 and cl[0]['frame'] in (20, 21, 40) and 'mostly class 2' in cl[0]['note']


def test_orphan_detection_is_a_false_positive():
    _, cands = run([[car(i)] + ([[400, 100, 460, 160, 0.6, 2]] if i == 10 else []) for i in range(60)])
    fp = [c for c in cands if c['reason'] == 'fp']
    assert len(fp) == 1 and fp[0]['frame'] == 10 and fp[0]['track'] is None


def test_static_track_is_a_hard_negative():
    _, cands = run([[car(i), [500, 400, 560, 460, 0.7, 2]] for i in range(80)])
    fp = [c for c in cands if c['reason'] == 'fp']
    assert len(fp) == 1 and 'static' in fp[0]['note']


def test_two_classes_on_one_object_is_a_class_candidate():
    _, cands = run([[car(i)] + ([[car(i)[0] + 2, 301, car(i)[2] - 2, 379, 0.5, 5]] if i == 25 else []) for i in range(60)])
    assert any(c['reason'] == 'class' and 'on one object' in c['note'] for c in cands)


def test_notes_use_class_names_when_given():
    frames, offs, dets = make_dump([[car(i, cls=8 if i == 20 else 2)] for i in range(60)])
    tracks = build_tracks(frames, offs, dets, FPS)
    cands = find_candidates(frames, offs, dets, tracks, FPS, {2: 'Light Vehicle', 8: 'Short Truck'})
    note = next(c['note'] for c in cands if c['reason'] == 'class')
    assert 'Short Truck' in note and 'Light Vehicle' in note and 'class 8' not in note


def test_low_confidence_big_box_is_borderline():
    _, cands = run([[car(i, conf=0.3 if i == 15 else 0.9)] for i in range(60)])
    assert any(c['reason'] == 'borderline' and c['frame'] == 15 for c in cands)


def test_selection_respects_budget_spacing_and_quotas():
    cands = []
    for k in range(200):
        cands.append(dict(frame=k * 10, reason=['fn', 'fp', 'class', 'borderline'][k % 4], score=0.5 + (k % 7) / 10,
                          box=np.array([0, 0, 50, 50.0]), track=k, note=''))
    chosen = select_frames(cands, 20, FPS)
    frames = [f for f, _ in chosen]
    assert len(chosen) == 20
    assert all(b - a >= 60 for a, b in zip(frames, frames[1:]))             # 3 s apart
    picked = [cs[0]['reason'] for _, cs in chosen]
    assert {picked.count(r) for r in ('fn', 'fp', 'class', 'borderline')} <= {3, 4, 5, 6, 7}


def test_one_frame_per_track_per_reason():
    cands = [dict(frame=f, reason='class', score=0.5, box=np.zeros(4), track=1, note='') for f in range(0, 2000, 100)]
    assert len(select_frames(cands, 50, FPS)) == 1


def test_budget_scales_with_duration():
    assert budget_for(7200) == 100 and budget_for(3600) == 50 and budget_for(60) == 20
