"""Unit tests for vehicle-level counting (fragment stitching, once-per-vehicle counts)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from core.perspective_transform import PerspectiveTransformer
from core.utils import is_valid_movement, is_vehicle_path
from tracking.vehicle_counter import VehicleCounter, points_in_polygon

# Zones and road calibration of the EMAAN 08 site, in processing coordinates (853x640), as logged by the production run
ZONES = {
    'north': np.array([[233, 358], [423, 304], [266, 215], [165, 226]]),
    'south': np.array([[236, 391], [453, 316], [804, 462], [369, 629]]),
}
ROAD = PerspectiveTransformer([(198, 241), (233, 233), (847, 522), (416, 636)], width_meters=7.0, length_meters=50.0)
FPS = 20.0
NEAR = np.array([560.0, 520.0])   # bottom-centre deep inside the south zone (near the camera)
FAR = np.array([250.0, 260.0])    # bottom-centre inside the north zone (far from the camera)


def make_counter(ground=False, **cfg):
    return VehicleCounter(ZONES, is_valid_movement, ['Pedestrian', 'Cyclist'], cfg,
                          ground_fn=ROAD.ground_points if ground else None)


def box_at(p, h):
    w = 1.5 * h
    return (p[0] - w / 2, p[1] - h, p[0] + w / 2, p[1])


def path(a, b, n):
    """n bottom-centres from a to b, with box height shrinking with distance from the camera."""
    out = []
    for s in np.linspace(0, 1, n):
        p = a + (b - a) * s
        h = 120 - 100 * (1 - (p[1] - FAR[1]) / (NEAR[1] - FAR[1]))
        out.append((p, max(h, 12)))
    return out


def run(counter, frames, t0=0.0):
    """frames: list of per-frame lists of (track_id, point, height, type). Returns all events incl. finalize."""
    events = []
    t = t0
    for i, dets in enumerate(frames):
        t = t0 + i / FPS
        tracks = [{'track_id': tid, 'box': box_at(p, h), 'type': vt, 'score': 0.9} for tid, p, h, vt in dets]
        events += counter.update(t, tracks)
    events += counter.update(t + 10.0, [])          # long quiet period: ended vehicles get decided
    events += counter.finalize(t + 10.0)
    return events


def test_site_geometry_sanity():
    assert points_in_polygon([NEAR], ZONES['south'])[0]
    assert points_in_polygon([FAR], ZONES['north'])[0]


def test_single_track_crossing_counts_once():
    frames = [[(1, p, h, 'Light Vehicle')] for p, h in path(NEAR, FAR, 60)]
    events = run(make_counter(), frames)
    assert len(events) == 1
    assert (events[0]['entry_zone'], events[0]['exit_zone'], events[0]['method']) == ('south', 'north', 'crossing')


def test_fragmented_vehicle_counted_once():
    frames = []
    for i, (p, h) in enumerate(path(NEAR, FAR, 60)):
        tid = 100 + i // 10                          # new tracker ID every 10 frames...
        frames.append([] if i % 10 in (8, 9) else [(tid, p, h, 'Light Vehicle')])   # ...after a 2-frame gap
    events = run(make_counter(), frames)
    assert len(events) == 1
    assert len(events[0]['track_ids']) >= 5


def test_fragment_receding_after_count_is_not_counted_again():
    pts = path(NEAR, FAR, 60) + path(FAR, FAR + np.array([-60.0, -30.0]), 30)
    frames = []
    for i, (p, h) in enumerate(pts):
        if 60 <= i < 70:
            frames.append([])                        # vehicle lost for 0.5 s after reaching the north zone
        else:
            frames.append([(1 if i < 60 else 2, p, h, 'Light Vehicle')])
    events = run(make_counter(), frames)
    assert len(events) == 1


def road_obs(gx, gy):
    """Image bottom-centre and box height of a car standing at road-plane position (gx, gy) metres."""
    to_px = lambda x, y: cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), ROAD.M_inv)[0, 0].astype(float)
    p = to_px(gx, gy)
    car_width_px = np.linalg.norm(to_px(gx + 0.9, gy) - to_px(gx - 0.9, gy))
    return p, 0.8 * car_width_px


def test_fast_vehicle_long_gap_is_stitched_on_road_plane():
    # 120 km/h: 3 detections near the camera, missed for 1.4 s, then a new ID far away in the north zone.
    # Unstitched, the far fragment would be counted a second time (an orphan receding from the south zone).
    speed, lane = 33.0, 2.0
    seen = {0: 1, 1: 1, 2: 1, 30: 2, 31: 2, 32: 2, 33: 2, 34: 2}    # frame -> tracker ID
    frames = []
    for i in range(35):
        p, h = road_obs(lane, 46.0 - speed * i / FPS)
        frames.append([(seen[i], p, h, 'Light Vehicle')] if i in seen else [])
    assert points_in_polygon([frames[0][0][1]], ZONES['south'])[0]
    assert points_in_polygon([frames[30][0][1]], ZONES['north'])[0]
    counter = make_counter(ground=True)
    events = run(counter, frames)
    assert len(events) == 1
    assert (events[0]['entry_zone'], events[0]['exit_zone']) == ('south', 'north')
    assert counter.vehicle_id_of(1) == counter.vehicle_id_of(2)


def _receding_then_oncoming(oncoming_track_id, gap_frames):
    """A car drives away into the far north zone and vanishes; an oncoming car appears where it was
    predicted to be and drives towards the camera through the south zone."""
    frames = []
    for i in range(50):                               # receding, 20 m/s, from the south zone to far away
        p, h = road_obs(2.0, 46.0 - 20.0 * i / FPS)
        frames.append([(1, p, h, 'Light Vehicle')])
    frames += [[] for _ in range(gap_frames)]
    gy0 = 46.0 - 20.0 * (50 + gap_frames) / FPS       # where the receding car would be by now
    for i in range(70):                               # oncoming, 20 m/s, towards the camera
        p, h = road_obs(2.4, gy0 + 20.0 * i / FPS)
        frames.append([(oncoming_track_id, p, h, 'Light Vehicle')])
    return frames


def test_oncoming_car_stitched_to_a_receding_one_is_split_and_counted():
    events = run(make_counter(ground=True), _receding_then_oncoming(2, gap_frames=4))
    counts = [e for e in events if not e.get('retract')]
    assert sorted((e['entry_zone'], e['exit_zone']) for e in counts) == [('north', 'south'), ('south', 'north')]


def test_tracker_id_swap_between_opposite_cars_is_split_and_counted():
    events = run(make_counter(ground=True), _receding_then_oncoming(1, gap_frames=1))
    counts = [e for e in events if not e.get('retract')]
    assert sorted((e['entry_zone'], e['exit_zone']) for e in counts) == [('north', 'south'), ('south', 'north')]


def test_class_comes_from_close_views_not_many_tiny_far_ones():
    # Oncoming car: a long approach far away (small boxes, misread as 'Bus'), then a few close frames
    pts = path(FAR + np.array([-30.0, -20.0]), FAR, 50) + path(FAR, NEAR, 20)
    frames = [[(1, p, h, 'Light Vehicle' if h >= 60 else 'Bus')] for p, h in pts]
    events = run(make_counter(), frames)
    assert sum(1 for _, h in pts if h < 60) > 3 * sum(1 for _, h in pts if h >= 60)
    assert len(events) == 1 and events[0]['type'] == 'Light Vehicle'


def test_vehicles_in_opposite_lanes_are_not_stitched_together_on_road_plane():
    away = [road_obs(1.75, 46.0 - 25.0 * i / FPS) for i in range(10)]
    toward = [road_obs(5.25, 20.0 + 25.0 * i / FPS) for i in range(12, 30)]
    frames = [[(1, p, h, 'Light Vehicle')] for p, h in away] + [[] for _ in range(2)]
    frames += [[(2, p, h, 'Light Vehicle')] for p, h in toward]
    counter = make_counter(ground=True)
    run(counter, frames)
    assert counter.vehicle_id_of(1) != counter.vehicle_id_of(2)


def test_zone_visits_are_taken_along_the_path_between_detections():
    # Detections only before the south zone and beyond it in the north zone: the path crosses both
    before = np.array([700.0, 600.0])
    after = np.array([215.0, 240.0])
    assert not points_in_polygon([before], ZONES['south'])[0]
    frames = [[(1, before, 110, 'Light Vehicle')], [(1, after, 14, 'Light Vehicle')]]
    events = run(make_counter(), frames)
    assert len(events) == 1 and events[0]['entry_zone'] == 'south' and events[0]['exit_zone'] == 'north'


def test_opposite_vehicles_passing_are_both_counted():
    away = path(NEAR, FAR, 60)
    toward = path(FAR + np.array([40.0, 0.0]), NEAR + np.array([40.0, 0.0]), 60)
    frames = [[(1, pa, ha, 'Light Vehicle'), (2, pt, ht, 'Minibus Taxi')] for (pa, ha), (pt, ht) in zip(away, toward)]
    events = run(make_counter(), frames)
    assert sorted((e['entry_zone'], e['exit_zone']) for e in events) == [('north', 'south'), ('south', 'north')]


def test_partial_box_duplicate_is_not_an_extra_count():
    # A second, smaller box on the same receding car (e.g. its rear only) once it is in the north zone
    frames = []
    for i, (p, h) in enumerate(path(NEAR, FAR, 60)):
        dets = [(1, p, h, 'Light Vehicle')]
        if i >= 50:
            dets.append((2, p + np.array([0.3 * h, -0.2 * h]), 0.5 * h, 'Light Vehicle'))
        frames.append(dets)
    events = run(make_counter(), frames)
    assert len(events) == 1


def test_orphan_direction_comes_from_motion_never_same_zone():
    # Seen only inside the north zone, receding from the south zone: it came from the south
    inside_north = path(np.array([300.0, 320.0]), np.array([215.0, 235.0]), 25)
    events = run(make_counter(), [[(1, p, h, 'Light Vehicle')] for p, h in inside_north])
    assert len(events) == 1
    assert (events[0]['entry_zone'], events[0]['exit_zone'], events[0]['method']) == ('south', 'north', 'inferred')


def test_jitter_inside_one_zone_is_not_counted():
    rng = np.random.default_rng(0)
    frames = [[(1, NEAR + rng.normal(0, 2, 2), 110, 'Light Vehicle')] for _ in range(80)]
    assert run(make_counter(), frames) == []


def test_pedestrian_fragment_is_not_stitched_to_a_vehicle():
    pts = path(NEAR, FAR, 60)
    frames = [[(1, p, h, 'Light Vehicle')] for p, h in pts[:30]] + [[] for _ in range(4)]
    frames += [[(2, p, h, 'Pedestrian')] for p, h in pts[34:]]
    counter = make_counter()
    run(counter, frames)
    assert counter.vehicle_id_of(1) != counter.vehicle_id_of(2)


def test_is_vehicle_path_excludes_non_vehicles_and_same_zone():
    base = {'status': 'exited', 'entry_direction': 'south', 'exit_direction': 'north', 'type': 'Light Vehicle'}
    assert is_vehicle_path(base)
    assert not is_vehicle_path(dict(base, type='Pedestrian'))
    assert not is_vehicle_path(dict(base, type='Cyclist'))
    assert not is_vehicle_path(dict(base, exit_direction='south'))
    assert not is_vehicle_path(dict(base, status='timed_out'))
