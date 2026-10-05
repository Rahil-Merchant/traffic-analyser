"""Replay a detection dump through the counting pipeline, without video or GPU.

Record a dump by running the app with DUMP_DETECTIONS = True (config.py); it is saved next to the
reports as detections_<video>.npz. Replaying it runs the same tracker and counting code as the app,
in seconds, so tracker and counting settings can be compared on identical detections.

    python utility_scripts/replay_detections.py <dump.npz> [--tracker ocsort|bytetrack] [--legacy]
        [--ocsort '{"max_age_sec": 0.5}'] [--counter '{"dup_iou": 0.7}'] [--out <dir>] [--no-reports]
"""
import argparse
import json
import os
import sys
import threading
import types
from collections import Counter
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _mod, _stub in (('video_reader', types.SimpleNamespace(PyVideoReader=None)),       # GPU decoder: not needed
                    ('pynvml', types.SimpleNamespace(nvmlInit=lambda: None))):          # NVIDIA-only: not needed
    try:
        __import__(_mod)
    except Exception:
        sys.modules[_mod] = _stub

import core.video_processor as vp                                    # noqa: E402
import reporting.excel_report as excel_report                        # noqa: E402
from core.finalized_paths_store import FinalizedPathsStore           # noqa: E402
from core.perspective_transform import PerspectiveTransformer        # noqa: E402
from tracking.zone_tracker import ZoneTracker                        # noqa: E402


class _Boxes:
    """Minimal stand-in for an ultralytics result: what _process_inference_results reads."""
    def __init__(self, dets):
        self.data = self
        self._d = dets

    def cpu(self):
        return self

    def numpy(self):
        return self._d

    def __len__(self):
        return len(self._d)


class _Result:
    def __init__(self, dets):
        self.boxes = _Boxes(dets)


def build_processor(meta, tracker, legacy, ocsort_overrides, counter_overrides):
    p = vp.VideoProcessor.__new__(vp.VideoProcessor)
    p.target_fps = vp.TARGET_FPS
    p.effective_fps = meta['effective_fps']
    p.frame_shape = tuple(meta['frame_shape'])
    p.model_input_size_config = meta['model_input_size']
    p.letterbox_params = tuple(meta['letterbox_params']) if meta['letterbox_params'] else None
    p.class_id_to_name_map = {int(k): v for k, v in meta['class_names'].items()}
    p.tracker_lock = threading.Lock()
    p.perf = None
    p.active_track_data = {}
    p.finalized_paths_store = FinalizedPathsStore()
    p.last_vehicle_info = None
    p.detection_dump = []
    zones = {k: np.array(v, dtype=np.int32) for k, v in meta['zones'].items()}
    p.detection_zones_polygons = zones
    p.zone_tracker = ZoneTracker(zones)
    p.perspective_transformer = None
    p.zone_pair, p.zone_axis = None, None
    if meta['road_corners'] and meta['road_width_m'] and meta['road_length_m']:
        p.perspective_transformer = PerspectiveTransformer(meta['road_corners'], meta['road_width_m'], meta['road_length_m'])
    if len(zones) == 2:
        p.zone_pair = tuple(zones)
        if p.perspective_transformer is not None:                     # as in _initialize_run
            c = np.array(meta['road_corners'], dtype=np.float32)
            axis = (c[2] + c[1]) / 2 - (c[3] + c[0]) / 2
        else:
            a, b = (np.mean(zones[z], axis=0) for z in p.zone_pair)
            axis = b - a
        p.zone_axis = axis / (np.linalg.norm(axis) + 1e-6)
    vp.TRACKER_TYPE = tracker
    vp.OCSORT_CONFIG = dict(vp.OCSORT_CONFIG, **ocsort_overrides)
    p.tracker = p._build_tracker()
    p.vehicle_counter = None if legacy else vp.VehicleCounter(
        zones, vp.is_valid_movement, vp.NON_VEHICLE_TYPES, dict(vp.VEHICLE_COUNTER_CONFIG, **counter_overrides),
        ground_fn=p.perspective_transformer.ground_points if p.perspective_transformer else None,
        frame_size=p.frame_shape)
    return p


def replay(dump_path, tracker, legacy, ocsort_overrides, counter_overrides, batch=64):
    z = np.load(dump_path)
    meta = json.loads(str(z['meta']))
    frames, offs, dets = z['frame'], z['offs'], z['dets']
    p = build_processor(meta, tracker, legacy, ocsort_overrides, counter_overrides)
    start = datetime.fromisoformat(meta['start_datetime'])
    fps = meta['original_fps']
    for b in range(0, len(frames), batch):
        idx = range(b, min(b + batch, len(frames)))
        p._process_inference_results([_Result(dets[offs[i]:offs[i + 1]]) for i in idx], [(None, {})] * len(idx),
                                     [int(frames[i]) for i in idx], [frames[i] / fps for i in idx], start)
    end_t = meta['total_frames'] / fps
    if p.vehicle_counter is not None:
        for ev in p.vehicle_counter.finalize(end_t):
            p._record_vehicle_event(ev, start)
    return p, meta, start


def summarise(paths, start):
    vehicles = [q for q in paths if vp.is_vehicle_path(q)]
    print(f"\nVehicles counted: {len(vehicles)}   (rows incl. pedestrians/cyclists: {sum(q['status'] == 'exited' for q in paths)})")
    print("By movement:", dict(Counter(f"{q['entry_direction']}>{q['exit_direction']}" for q in vehicles)))
    print("By class:   ", dict(Counter(q['type'] for q in vehicles).most_common()))
    day = datetime.combine(start.date(), datetime.min.time())
    bins = Counter()
    for q in vehicles:
        b = day + timedelta(minutes=15 * ((q['exit_time'] - day).total_seconds() // 900))
        bins[(b.strftime('%H:%M'), f"{q['entry_direction']}>{q['exit_direction']}")] += 1
    moves = sorted({m for _, m in bins})
    print("\n" + "15 min".ljust(8) + "".join(m.rjust(16) for m in moves))
    for t in sorted({t for t, _ in bins}):
        print(t.ljust(8) + "".join(str(bins[(t, m)]).rjust(16) for m in moves))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dump')
    ap.add_argument('--tracker', default=vp.TRACKER_TYPE, choices=['ocsort', 'bytetrack'])
    ap.add_argument('--legacy', action='store_true', help='per-track counting with motion inference instead of stitching')
    ap.add_argument('--ocsort', default='{}', help='JSON overrides for OCSORT_CONFIG')
    ap.add_argument('--counter', default='{}', help='JSON overrides for VEHICLE_COUNTER_CONFIG')
    ap.add_argument('--out', default=None, help='report folder (default: replay_<settings> next to the dump)')
    ap.add_argument('--no-reports', action='store_true')
    a = ap.parse_args()
    p, meta, start = replay(a.dump, a.tracker, a.legacy, json.loads(a.ocsort), json.loads(a.counter))
    paths = p.finalized_paths_store.get_all()
    summarise(paths, start)
    if not a.no_reports:
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.dump)),
                                    f"replay_{a.tracker}_{'legacy' if a.legacy else 'stitch'}")
        os.makedirs(out, exist_ok=True)
        excel_report.REPORT_OUTPUT_DIR = out
        video = meta['video_path']
        excel_report.create_excel_report(paths, start, meta['primary_direction'], video, report_type='standard')
        excel_report.create_excel_report(paths, start, meta['primary_direction'], video, report_type='tia')
        if len(meta['zones']) == 2:
            secondary = [d for d in meta['zones'] if d != meta['primary_direction']]
            if secondary:
                excel_report.create_bidirectional_link_count_report(
                    paths, start, video, meta['primary_direction'], secondary[0],
                    str(timedelta(seconds=int(meta['total_frames'] / meta['original_fps']))))
        print(f"\nReports written to {out}")


if __name__ == '__main__':
    main()
