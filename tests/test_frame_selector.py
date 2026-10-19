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


def test_detection_where_the_vehicle_must_be_is_not_a_miss():
    # production saw the vehicle in the gap (the tracker just didn't link it): not a missed vehicle
    frames, offs, dets = make_dump([[car(i)] for i in range(60)])
    tracks = build_tracks(*make_dump([[car(i)] if not 30 <= i < 36 else [] for i in range(60)]), FPS)
    assert not [c for c in find_candidates(frames, offs, dets, tracks, FPS) if c['reason'] == 'fn']


def test_class_flicker_is_flagged_at_the_largest_disagreeing_frame():
    _, cands = run([[car(i, cls=8 if i in (20, 21, 40) else 2)] for i in range(60)])
    cl = [c for c in cands if c['reason'] == 'class']
    assert len(cl) == 1 and cl[0]['frame'] in (20, 21, 40) and cl[0]['note'] == 'class 8 / class 2'


def test_orphan_detection_is_a_false_positive():
    _, cands = run([[car(i)] + ([[400, 100, 460, 160, 0.6, 2]] if i == 10 else []) for i in range(60)])
    fp = [c for c in cands if c['reason'] == 'fp']
    assert len(fp) == 1 and fp[0]['frame'] == 10 and fp[0]['track'] is None


def test_static_track_is_a_hard_negative():
    _, cands = run([[car(i), [500, 400, 560, 460, 0.7, 2]] for i in range(80)])
    fp = [c for c in cands if c['reason'] == 'fp']
    assert len(fp) == 1 and 'static' in fp[0]['note']


def test_distant_static_box_is_not_a_hard_negative():
    # a far-away vehicle barely moves in pixels; below 20 px tall it is too small to call static
    _, cands = run([[car(i), [500, 100, 512, 112, 0.7, 2]] for i in range(80)])
    assert not [c for c in cands if c['reason'] == 'fp' and 'static' in c['note']]


def test_two_classes_on_one_object_is_a_class_candidate():
    _, cands = run([[car(i)] + ([[car(i)[0] + 2, 301, car(i)[2] - 2, 379, 0.5, 5]] if i == 25 else []) for i in range(60)])
    assert any(c['reason'] == 'class' and c['track'] is None and '/' in c['note'] for c in cands)


def test_notes_use_class_names_when_given():
    frames, offs, dets = make_dump([[car(i, cls=8 if i == 20 else 2)] for i in range(60)])
    tracks = build_tracks(frames, offs, dets, FPS)
    cands = find_candidates(frames, offs, dets, tracks, FPS, {2: 'Light Vehicle', 8: 'Short Truck'})
    note = next(c['note'] for c in cands if c['reason'] == 'class')
    assert note == 'Short Truck / Light Vehicle'


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


# ---- detection, caches, manifest and the run as a whole (tiny synthetic video, no real model) ----

import json
import cv2
import pytest
from frame_selector import detect as fsd
from frame_selector.export import read_manifest, update_manifest


def make_video(path, n=64, fps=20.0, size=(320, 180), seed=0):
    rng = np.random.default_rng(seed)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(n):
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.putText(img, str(i), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        img[-10:, :] = rng.integers(0, 255, (10, size[0], 3), dtype=np.uint8)   # makes videos differ
        w.write(img)
    w.release()
    return str(path)


def write_dump(path, video, frame_dets, meta_extra=None, stamped=True):
    frames, offs, dets = make_dump(frame_dets)
    meta = dict(video_path=os.path.abspath(video), video_name=os.path.basename(video), video_size=os.path.getsize(video),
                start_datetime=None, original_fps=FPS, total_frames=len(frames), effective_fps=FPS, frame_shape=[320, 180],
                model_input_size=640, letterbox_params=None, class_names={2: 'Light Vehicle', 8: 'Short Truck'})
    if stamped:
        meta['detector'] = dict(version=fsd.DETECTOR_VERSION, model=fsd.model_id(fsd.configured_model()), conf=fsd.CONF, iou=fsd.IOU)
    meta.update(meta_extra or {})
    np.savez_compressed(path, frame=frames, offs=offs, dets=dets, meta=np.array(json.dumps(meta)))
    return str(path)


def test_detection_feeds_the_model_contiguous_tensors_and_stamps_the_dump(tmp_path):
    torch = pytest.importorskip('torch')
    video = make_video(tmp_path / 'v.mp4', n=40)
    seen = []

    class Boxes:
        def __init__(self):
            self.data = torch.tensor([[10.0, 20.0, 50.0, 60.0, 0.9, 2.0]])
        def __len__(self):
            return 1

    class Result:
        names = {2: 'Light Vehicle'}
        boxes = Boxes()

    def model(x, **kw):
        seen.append((x.is_contiguous(), tuple(x.shape), kw['conf'], kw['iou']))
        return [Result() for _ in range(len(x))]

    out = fsd.detect_video(video, model, str(tmp_path / 'd.npz'), device='cpu', progress=lambda m: None)
    assert seen and all(c for c, *_ in seen)                               # TensorRT reads raw memory
    assert all(shape[1:] == (3, 640, 640) for _, shape, _, _ in seen)
    assert {(conf, iou) for *_, conf, iou in seen} == {(0.25, 0.7)}          # production's thresholds
    frames, offs, dets, meta = fsd.load_dump(out)
    assert len(frames) == 40 and len(dets) == 40 and meta['detector']['version'] == fsd.DETECTOR_VERSION
    assert fsd.dump_status(out, video) is None


def test_old_and_mismatched_caches_are_not_reused(tmp_path):
    a = make_video(tmp_path / '4068.mp4', seed=1)
    b = make_video(tmp_path / '4068 2.mp4', n=70, seed=2)
    dets = [[car(i)] for i in range(64)]
    assert fsd.dump_status(write_dump(tmp_path / 'ok.npz', a, dets), a) is None
    assert 'older version' in fsd.dump_status(write_dump(tmp_path / 'v1.npz', a, dets, {'conf': 0.05}, stamped=False), a)
    assert 'older version' in fsd.dump_status(write_dump(tmp_path / 'old.npz', a, dets, {'detector': {'version': 1}}), a)
    assert 'different video' in fsd.dump_status(write_dump(tmp_path / 'b.npz', b, dets), a)
    # production app dumps: named after the app's temp file, accepted only for the matching video
    app = write_dump(tmp_path / 'app.npz', a, dets, {'video_path': r'C:\Temp\4068_conv_ab12cd.mp4'}, stamped=False)
    assert fsd.dump_status(app, a) is None
    app2 = write_dump(tmp_path / 'app2.npz', b, dets, {'video_path': r'C:\Temp\4068 2_conv_ab12cd.mp4'}, stamped=False)
    assert 'different video' in fsd.dump_status(app2, a)                    # '4068 2' is not '4068'
    assert fsd.video_key(a) != fsd.video_key(b)


def test_same_file_name_from_different_folders_gets_different_keys(tmp_path):
    (tmp_path / 'day1').mkdir(); (tmp_path / 'day2').mkdir()
    a = make_video(tmp_path / 'day1' / '06.00.00-08.00.00.mp4', seed=1)
    b = make_video(tmp_path / 'day2' / '06.00.00-08.00.00.mp4', seed=2)
    assert fsd.video_key(a) != fsd.video_key(b)
    assert fsd.video_key(a).startswith('06.00.00-08.00.00_')


def test_rerun_replaces_rows_and_deletes_stale_files(tmp_path):
    out = str(tmp_path)
    for d in ('images', 'labels', 'review'):
        os.makedirs(os.path.join(out, d))

    def row(name, video, vid, frame):
        for p in (f'images/{name}.jpg', f'labels/{name}.txt', f'review/{name}.jpg'):
            open(os.path.join(out, p), 'w').close()
        return dict(image=f'images/{name}.jpg', label=f'labels/{name}.txt', video=video, video_id=vid, frame=frame,
                    time_s=0, clock='', reasons='fn', score=1, tracks='', notes='', n_labels=0)

    update_manifest(out, 'x_aaaaaa', 'x.dav', [row('x_aaaaaa_0000010', 'x.dav', 'x_aaaaaa', 10)])
    update_manifest(out, 'y_bbbbbb', 'y.dav', [row('y_bbbbbb_0000010', 'y.dav', 'y_bbbbbb', 10)])
    legacy = row('x_0000099', 'x.dav', '', 99)                               # a row from before video ids
    with open(os.path.join(out, 'manifest.csv'), 'a') as fh:
        fh.write(','.join(str(legacy[k]) for k in ['image', 'label', 'video', 'video_id', 'frame', 'time_s', 'clock',
                                                   'reasons', 'score', 'tracks', 'notes', 'n_labels']) + '\n')
    update_manifest(out, 'x_aaaaaa', 'x.dav', [row('x_aaaaaa_0000020', 'x.dav', 'x_aaaaaa', 20)])
    assert sorted(r['image'] for r in read_manifest(out)) == ['images/x_aaaaaa_0000020.jpg', 'images/y_bbbbbb_0000010.jpg']
    assert sorted(os.listdir(os.path.join(out, 'images'))) == ['x_aaaaaa_0000020.jpg', 'y_bbbbbb_0000010.jpg']
    assert sorted(os.listdir(os.path.join(out, 'review'))) == ['x_aaaaaa_0000020.jpg', 'y_bbbbbb_0000010.jpg']


def test_a_failing_video_does_not_stop_the_run_or_lose_the_others(tmp_path):
    from frame_selector.pipeline import run as run_all
    good = make_video(tmp_path / 'good.mp4')
    dump = write_dump(tmp_path / 'good.npz', good, [[car(i)] if not 30 <= i < 36 else [] for i in range(64)])
    log = []
    rows, failed = run_all([str(tmp_path / 'missing.mp4'), good], str(tmp_path / 'out'), [None, dump], budget=20, progress=log.append)
    assert [os.path.basename(v) for v, _ in failed] == ['missing.mp4']
    assert rows and all(r['video'] == 'good.mp4' for r in rows)
    assert read_manifest(str(tmp_path / 'out')) == [{k: str(v) for k, v in r.items()} for r in rows]
    assert os.path.exists(os.path.join(tmp_path / 'out', rows[0]['image']))
    assert any(l.startswith('FAILED missing.mp4') for l in log)


def test_results_survive_dead_worker_processes(tmp_path, monkeypatch):
    from concurrent.futures.process import BrokenProcessPool
    import frame_selector.pipeline as P

    class DeadPool:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def submit(self, *a, **k):
            raise BrokenProcessPool('a child process terminated abruptly')

    monkeypatch.setattr(P, 'ProcessPoolExecutor', DeadPool)
    good = make_video(tmp_path / 'good.mp4')
    dump = write_dump(tmp_path / 'good.npz', good, [[car(i)] if not 30 <= i < 36 else [] for i in range(64)])
    rows, failed = P.run([good], str(tmp_path / 'out'), [dump], budget=20, progress=lambda m: None)
    assert rows and not failed and len(read_manifest(str(tmp_path / 'out'))) == len(rows)


def test_cache_with_undecoded_frames_is_not_reused(tmp_path):
    a = make_video(tmp_path / 'a.mp4')
    p = write_dump(tmp_path / 'part.npz', a, [[car(i)] for i in range(64)])
    z = dict(np.load(p))
    z['frame'] = z['frame'] + 10                                              # frames 0-9 were never decoded
    np.savez_compressed(p, **z)
    assert 'incomplete' in fsd.dump_status(p, a)


def test_detection_process_failure_is_reported_not_hung(tmp_path):
    from frame_selector.pipeline import detect_in_subprocess
    with pytest.raises(RuntimeError):
        detect_in_subprocess(str(tmp_path / 'missing.mp4'), str(tmp_path / 'd.npz'), progress=lambda m: None)


def test_frames_are_read_back_in_decode_order(tmp_path):
    # frames differ (each shows its index); the reader must return exactly what a sequential read returns
    video = make_video(tmp_path / 'v.mp4', n=64)
    cap = cv2.VideoCapture(video)
    seq = [cap.read()[1] for _ in range(64)]
    got = dict(fsd.read_frames(video, [50, 3, 3, 17, 63]))
    assert sorted(got) == [3, 17, 50, 63]
    assert all(np.array_equal(got[i], seq[i]) for i in got)
