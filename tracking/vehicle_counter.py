# project/tracking/vehicle_counter.py
"""Vehicle-level counting on top of tracker IDs.

A tracker ID is only a fragment of a vehicle: a missed detection, occlusion or fast motion
can end one ID and start another on the same vehicle. This layer stitches fragments into
vehicles and counts each vehicle at most once, so ID switches cannot inflate counts.

- Stitching: a new track joins a vehicle that is not seen in this frame when it appears
  where that vehicle is predicted to be. With a road calibration the prediction runs in
  metres on the road plane (accurate over multi-second gaps, lane-level across the road);
  without one it runs in image space with distances measured in box heights, so one gate
  works near and far from the camera. Tolerance grows with the gap and is tight across the
  direction of travel.
- Duplicates: a new track that overlaps a vehicle seen in the same frame is that vehicle.
- Crossings: zone visits are taken along the path between consecutive observations, so a
  fast vehicle with only a few detections still registers every zone it passed through.
- Orphans (2-zone links only): a vehicle that ends after visiting one zone is counted only
  with clear travel along the link, when no counted vehicle explains it. Its direction comes
  from its motion, so entry and exit are never the same zone.
"""
import math
from collections import Counter, deque

import numpy as np


def polygon_edges(poly):
    """Precomputed edges for points_in_polygon."""
    poly = np.asarray(poly, dtype=float)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    dy = y2 - y1
    return x1[None], y1[None], y2[None], ((x2 - x1) / np.where(dy == 0, 1e-12, dy))[None]


def points_in_polygon(pts, poly):
    """Vectorised even-odd point-in-polygon test. pts (N,2), poly (M,2) or polygon_edges(poly) -> (N,) bool."""
    x1, y1, y2, slope = poly if isinstance(poly, tuple) else polygon_edges(poly)
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    x, y = pts[:, :1], pts[:, 1:2]
    crosses = (y1 > y) != (y2 > y)
    return (crosses & (x < x1 + (y - y1) * slope)).sum(axis=1) % 2 == 1


def box_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


DEFAULT_CONFIG = {
    # Image-space stitching (no road calibration, or point outside it). Distances in box heights.
    'stitch_max_gap_s': 2.0,         # longest detection gap a vehicle can be stitched across
    'stitch_unknown_vel_gap_s': 0.6, # gap limit when the vehicle's velocity is still unknown
    'stitch_along_tol_h': 1.5,       # base tolerance along the direction of travel
    'stitch_along_rate_h': 2.0,      # extra along-track tolerance per second of gap
    'stitch_lateral_tol_h': 1.0,     # base tolerance across the direction of travel
    'stitch_lateral_rate_h': 0.5,    # extra lateral tolerance per second of gap
    'stitch_max_speed_h': 15.0,      # max plausible speed (box heights / s) when velocity is unknown
    'stitch_size_ratio': 2.5,        # max height ratio between a vehicle and a candidate fragment
    # Road-plane stitching (road calibration available). Distances in metres.
    'stitch_max_gap_ground_s': 3.0,
    'stitch_ground_along_m': 3.0,
    'stitch_ground_along_rate': 3.0, # m per s of gap: speed changes and calibration error
    'stitch_ground_lateral_m': 1.5,  # about half a lane
    'stitch_ground_lateral_rate': 0.5,
    'stitch_ground_max_speed': 45.0, # m/s (~160 km/h) when velocity is unknown
    'stitch_ground_width_ratio': 2.0,# max ratio of bottom-edge widths on the road plane
    'ground_jitter_px': 3.0,         # box-edge noise (px) converted to metres and added to the tolerances
    # Duplicates, vehicle end, orphans
    'dup_iou': 0.6,                  # same-frame overlap that makes two tracks one vehicle
    'dup_frames': 3,                 # consecutive frames of that overlap before two vehicles merge
    'end_gap_s': 3.5,                # unseen this long -> vehicle ended (must exceed both stitch gaps)
    'orphan_min_travel_h': 1.5,      # orphan must travel this many box heights along the link...
    'orphan_min_frac': 0.10,         # ...and this fraction of the distance between the two zones
    'orphan_explain_radius_h': 3.0,  # a counted vehicle this close (box heights) explains an orphan
    'orphan_explain_window_s': 2.0,
    'keep_s': 15.0,                  # history kept after a vehicle ends (for orphan explanation)
    'velocity_window_s': 0.75,
    'max_positions': 2400,
}

# Observation tuple layout
_T, _X, _Y, _H, _BOX, _GX, _GY, _GW = range(8)


class _Vehicle:
    def __init__(self, vid, t):
        self.vid = vid
        self.track_ids = set()
        self.obs = deque()                  # (t, x, y, h, box, ground_x, ground_y, ground_width)
        self.type_votes = Counter()
        self.zones = []                     # ordered zone visits: (zone, t)
        self.counted = False
        self.decided = False                # final decision taken (counted, dropped or merged)
        self.merged_into = None
        self.first_t = t
        self.last_t = t
        self.last_box = None

    @property
    def last_xy(self):
        return np.array(self.obs[-1][_X:_Y + 1])

    @property
    def last_ground(self):
        return np.array(self.obs[-1][_GX:_GY + 1])

    def ref_height(self):
        hs = [o[_H] for o in list(self.obs)[-10:]]
        return float(np.median(hs)) if hs else 0.0

    def ref_ground_width(self):
        ws = [o[_GW] for o in list(self.obs)[-10:] if np.isfinite(o[_GW])]
        return float(np.median(ws)) if ws else float('nan')

    def velocity(self, window_s, ground=False):
        """Least-squares velocity of the bottom-centre over the last window_s seconds
        (px/s, or m/s on the road plane with ground=True)."""
        c0 = _GX if ground else _X
        pts = [o for o in self.obs if np.isfinite(o[c0])]
        if len(pts) < 2:
            return None
        recent = [o for o in pts if o[_T] >= self.last_t - window_s]
        if len(recent) < 2:
            recent = pts[-2:]
        ts = np.array([o[_T] for o in recent])
        if ts[-1] - ts[0] < 0.04:
            return None
        xy = np.array([o[c0:c0 + 2] for o in recent], dtype=float)
        tc = ts - ts.mean()
        return (tc[:, None] * (xy - xy.mean(axis=0))).sum(axis=0) / (tc ** 2).sum()

    def position_at(self, t, window_s):
        """Observed (interpolated) or extrapolated image bottom-centre at time t."""
        ts = [o[_T] for o in self.obs]
        if ts[0] <= t <= ts[-1]:
            i = int(np.searchsorted(ts, t))
            if i == 0 or ts[i] == t:
                return np.array(self.obs[i][_X:_Y + 1])
            a, b = self.obs[i - 1], self.obs[i]
            w = (t - a[_T]) / max(b[_T] - a[_T], 1e-9)
            return np.array(a[_X:_Y + 1]) * (1 - w) + np.array(b[_X:_Y + 1]) * w
        v = self.velocity(window_s)
        if t > ts[-1]:
            return self.last_xy + (v * (t - ts[-1]) if v is not None else 0)
        return np.array(self.obs[0][_X:_Y + 1]) - (v * (ts[0] - t) if v is not None else 0)

    def vehicle_type(self):
        return self.type_votes.most_common(1)[0][0] if self.type_votes else 'Light Vehicle'


class VehicleCounter:
    """Feed one frame at a time with update(); events are counted vehicle paths (dicts).

    ground_fn, if given, maps (N,2) image points to (N,2) road-plane metres (NaN where unreliable);
    the second ground axis must run along the road.
    """

    def __init__(self, zones, is_valid_movement=None, non_vehicle_types=(), config=None, ground_fn=None):
        self.zones = {k: np.asarray(v, dtype=float) for k, v in zones.items()}
        self._zone_edges = {k: polygon_edges(v) for k, v in self.zones.items()}
        self.zone_names = list(self.zones)
        self.centroids = {k: v.mean(axis=0) for k, v in self.zones.items()}
        self.is_valid_movement = is_valid_movement or (lambda a, b, names: a != b)
        self.non_vehicle_types = set(non_vehicle_types)
        self.cfg = dict(DEFAULT_CONFIG, **(config or {}))
        self.ground_fn = ground_fn
        self.vehicles = {}
        self.track_to_vid = {}
        self._next_vid = 1
        self._dup_streak = Counter()

    # ---------- public API ----------
    def update(self, t, tracks):
        """tracks: iterable of dicts with track_id, box (x1,y1,x2,y2 in zone coordinates), type, score."""
        events = []
        seen = {}                                     # vid -> box observed this frame
        tracks = [dict(tr, ground=self._ground(tr['box'])) for tr in tracks]
        known = [tr for tr in tracks if self._vid_of(tr['track_id']) is not None]
        new = sorted((tr for tr in tracks if self._vid_of(tr['track_id']) is None),
                     key=lambda tr: -tr.get('score', 0.0))
        for tr in known:
            v = self.vehicles[self._vid_of(tr['track_id'])]
            if v.vid not in seen:                     # a second track on the same vehicle adds nothing
                self._observe(v, tr, t, events, seen)
        for tr in new:
            v = (self._duplicate_of(tr, seen) or self._stitch_target(tr, t, seen)
                 or self._new_vehicle(tr, t))
            self.track_to_vid[tr['track_id']] = v.vid
            v.track_ids.add(tr['track_id'])
            if v.vid not in seen:
                self._observe(v, tr, t, events, seen)
        self._merge_concurrent(seen)
        events += self._close_ended(t)
        return events

    def finalize(self, t):
        """End of video: decide every vehicle still pending."""
        events = []
        for v in list(self.vehicles.values()):
            if not v.decided and v.merged_into is None:
                ev = self._orphan_event(v)
                if ev:
                    events.append(ev)
        return events

    def vehicle_id_of(self, track_id):
        return self._vid_of(track_id)

    def status_of(self, vid):
        v = self.vehicles.get(vid)
        if v is None:
            return 'tracked'
        if v.counted:
            return 'exited'
        return 'active' if v.zones else 'tracked'

    def vehicle_type_of(self, vid):
        v = self.vehicles.get(vid)
        return v.vehicle_type() if v else None

    # ---------- internals ----------
    def _vid_of(self, track_id):
        vid = self.track_to_vid.get(track_id)
        while vid is not None and vid in self.vehicles and self.vehicles[vid].merged_into is not None:
            vid = self.vehicles[vid].merged_into
        return vid if vid in self.vehicles else None

    def _kind(self, vtype):
        return 'nonvehicle' if vtype in self.non_vehicle_types else 'vehicle'

    def _new_vehicle(self, tr, t):
        v = _Vehicle(self._next_vid, t)
        self.vehicles[v.vid] = v
        self._next_vid += 1
        return v

    @staticmethod
    def _bottom_centre(box):
        return np.array([(box[0] + box[2]) / 2.0, float(box[3])])

    def _ground(self, box):
        """Road-plane (x, y, bottom-edge width, along-road jitter, across-road jitter) in metres; NaN
        without a usable calibration. Jitter is what a few pixels of box noise mean at this point:
        far from the camera one pixel spans metres of road."""
        if self.ground_fn is None:
            return (math.nan,) * 5
        cx, by = (box[0] + box[2]) / 2.0, box[3]
        px = self.cfg['ground_jitter_px']
        g = np.asarray(self.ground_fn(np.array([[cx, by], [box[0], by], [box[2], by], [cx, by + px], [cx + px, by]])),
                       dtype=float)
        # Across-road extent only: far away a horizontal pixel span also covers metres along the road
        width = float(abs(g[2, 0] - g[1, 0])) if np.all(np.isfinite(g[1:3])) else math.nan
        dv, dh = g[3] - g[0], g[4] - g[0]
        along_jit = float(max(abs(dv[1]), abs(dh[1])))
        lat_jit = float(max(abs(dv[0]), abs(dh[0])))
        return float(g[0, 0]), float(g[0, 1]), width, along_jit, lat_jit

    def _zone_at(self, pts):
        """Zone name (or None) for each point."""
        out = np.full(len(pts), None, dtype=object)
        for name, edges in self._zone_edges.items():
            inside = points_in_polygon(pts, edges) & (out == None)  # noqa: E711
            out[inside] = name
        return out

    def _zones_along(self, p0, p1, t0, t1):
        """Zones entered along the straight path p0 -> p1, in order, with interpolated times."""
        step = float(np.linalg.norm(p1 - p0))
        n = 1 if step <= 6.0 else min(60, int(math.ceil(step / 4.0)))
        fr = np.linspace(1.0 / n, 1.0, n)
        pts = p0[None] + (p1 - p0)[None] * fr[:, None]
        visits = []
        for f, name in zip(fr, self._zone_at(pts)):
            if name is not None and (not visits or visits[-1][0] != name):
                visits.append((name, t0 + (t1 - t0) * f))
        return visits

    def _observe(self, v, tr, t, events, seen):
        box = [float(c) for c in tr['box']]
        p = self._bottom_centre(box)
        h = max(box[3] - box[1], 1.0)
        if v.obs:
            visits = self._zones_along(v.last_xy, p, v.last_t, t)
        else:
            z = self._zone_at(p[None])[0]
            visits = [(z, t)] if z is not None else []
        for zone, tz in visits:
            if not v.zones or v.zones[-1][0] != zone:
                v.zones.append((zone, tz))
        gx, gy, gw = tr['ground'][:3]
        v.obs.append((t, p[0], p[1], h, box, gx, gy, gw))
        while len(v.obs) > self.cfg['max_positions']:
            v.obs.popleft()
        v.type_votes[tr['type']] += 1
        v.last_t, v.last_box = t, box
        seen[v.vid] = box
        if not v.counted:
            self._try_crossing(v, events, seen)

    def _try_crossing(self, v, events, seen):
        entry = v.zones[0][0] if v.zones else None
        for zone, tz in v.zones[1:]:
            if zone != entry and self.is_valid_movement(entry, zone, self.zone_names):
                twin = self._counted_twin(v, seen)
                if twin is not None:
                    self._merge(v, twin)            # duplicate track of an already counted vehicle
                    return
                v.counted = v.decided = True
                events.append(self._event(v, entry, zone, v.zones[0][1], tz, 'crossing'))
                return

    def _counted_twin(self, v, seen):
        for vid, box in seen.items():
            u = self.vehicles.get(vid)
            if u is not None and u is not v and u.counted and box_iou(box, v.last_box) >= self.cfg['dup_iou']:
                return u
        return None

    def _duplicate_of(self, tr, seen):
        # Class is ignored: a box this similar in the same frame is the same object whatever it is called
        best, best_iou = None, self.cfg['dup_iou']
        for vid, box in seen.items():
            iou = box_iou(box, tr['box'])
            if iou >= best_iou:
                best, best_iou = self.vehicles[vid], iou
        return best

    @staticmethod
    def _anisotropic(r, d, along_tol, lat_tol):
        """Normalised distance of residual r with tolerance along unit direction d and across it."""
        if d is None:
            dist = float(np.linalg.norm(r)) / along_tol
        else:
            along = float(r @ d)
            dist = math.hypot(along / along_tol, float(np.linalg.norm(r - along * d)) / lat_tol)
        return dist if dist <= 1.0 else None

    def _gate_ground(self, v, ground_new, dt):
        c = self.cfg
        if dt > c['stitch_max_gap_ground_s']:
            return None
        gx, gy, gw_new, along_jit, lat_jit = ground_new
        gw_ref = v.ref_ground_width()
        if np.isfinite(gw_ref) and np.isfinite(gw_new) and gw_ref > 0.2 and gw_new > 0.2:
            if not (1 / c['stitch_ground_width_ratio'] <= gw_new / gw_ref <= c['stitch_ground_width_ratio']):
                return None
        vel = v.velocity(c['velocity_window_s'], ground=True)
        lat_tol = c['stitch_ground_lateral_m'] + c['stitch_ground_lateral_rate'] * dt + lat_jit
        if vel is not None and np.linalg.norm(vel) > 1.0:            # moving: constant speed on the road
            pred, d = v.last_ground + vel * dt, vel / np.linalg.norm(vel)
            along_tol = c['stitch_ground_along_m'] + c['stitch_ground_along_rate'] * dt
        elif vel is not None:                                         # (nearly) stationary
            pred, d = v.last_ground, None
            along_tol = c['stitch_ground_along_m'] + c['stitch_ground_along_rate'] * dt
        else:                                                         # single sighting: along the road only
            if dt > c['stitch_unknown_vel_gap_s']:
                return None
            pred, d = v.last_ground, np.array([0.0, 1.0])
            along_tol = c['stitch_ground_along_m'] + c['stitch_ground_max_speed'] * dt
        return self._anisotropic(np.array([gx, gy]) - pred, d, along_tol + along_jit, lat_tol)

    def _gate_image(self, v, p, h_new, dt, along_dir):
        c = self.cfg
        if dt > c['stitch_max_gap_s']:
            return None
        h_ref = v.ref_height()
        if h_ref <= 0 or not (1 / c['stitch_size_ratio'] <= h_new / h_ref <= c['stitch_size_ratio']):
            return None
        href = max(math.sqrt(h_ref * h_new), 8.0)
        vel = v.velocity(c['velocity_window_s'])
        lat_tol = (c['stitch_lateral_tol_h'] + c['stitch_lateral_rate_h'] * dt) * href
        if vel is not None and np.linalg.norm(vel) > 0.5 * href:     # moving: predict along its track
            pred, d = v.last_xy + vel * dt, vel / np.linalg.norm(vel)
            along_tol = (c['stitch_along_tol_h'] + c['stitch_along_rate_h'] * dt) * href
        elif vel is not None:                                         # (nearly) stationary
            pred, d = v.last_xy, None
            along_tol = (c['stitch_along_tol_h'] + c['stitch_along_rate_h'] * dt) * href
        else:                                                         # single sighting: any plausible motion
            if dt > c['stitch_unknown_vel_gap_s']:
                return None
            pred, d = v.last_xy, along_dir
            along_tol = (c['stitch_along_tol_h'] + c['stitch_max_speed_h'] * dt) * href
        return self._anisotropic(p - pred, d, along_tol, lat_tol)

    def _link_axis(self):
        if len(self.zone_names) != 2:
            return None
        a, b = (self.centroids[z] for z in self.zone_names)
        u = b - a
        n = np.linalg.norm(u)
        return u / n if n > 0 else None

    def _stitch_target(self, tr, t, seen):
        p = self._bottom_centre(tr['box'])
        h_new = max(tr['box'][3] - tr['box'][1], 1.0)
        g_new = np.array(tr['ground'][:2])
        kind = self._kind(tr['type'])
        axis = self._link_axis()
        best, best_d = None, None
        for v in self.vehicles.values():
            if v.vid in seen or v.merged_into is not None or self._kind(v.vehicle_type()) != kind or (v.decided and not v.counted):
                continue
            dt = t - v.last_t
            if dt <= 0:
                continue
            if np.all(np.isfinite(g_new)) and np.all(np.isfinite(v.last_ground)):
                d = self._gate_ground(v, tr['ground'], dt)
            else:
                d = self._gate_image(v, p, h_new, dt, axis)
            if d is not None and (best_d is None or d < best_d):
                best, best_d = v, d
        return best

    def _merge_concurrent(self, seen):
        """Merge two vehicles that keep overlapping in the same frames (duplicate tracks)."""
        vids = sorted(seen)
        live = set()
        w = self.cfg['velocity_window_s']
        for i, a in enumerate(vids):
            for b in vids[i + 1:]:
                va, vb = self.vehicles[a], self.vehicles[b]
                if box_iou(seen[a], seen[b]) < self.cfg['dup_iou']:
                    continue
                vel_a, vel_b = va.velocity(w), vb.velocity(w)
                if vel_a is not None and vel_b is not None and np.linalg.norm(vel_a) > 1 and np.linalg.norm(vel_b) > 1:
                    if float(vel_a @ vel_b) / (np.linalg.norm(vel_a) * np.linalg.norm(vel_b)) < 0.7:
                        continue                     # passing each other, not duplicates
                live.add((a, b))
                self._dup_streak[(a, b)] += 1
                if self._dup_streak[(a, b)] >= self.cfg['dup_frames'] and not (va.counted and vb.counted):
                    older, younger = (va, vb) if va.first_t <= vb.first_t else (vb, va)
                    self._merge(younger, older)
        for k in [k for k in self._dup_streak if k not in live]:
            del self._dup_streak[k]

    def _merge(self, src, dst):
        """Fold vehicle src into dst (dst keeps its id; a count on either side carries over)."""
        if src is dst or src.merged_into is not None:
            return
        dst.track_ids |= src.track_ids
        for tid in src.track_ids:
            self.track_to_vid[tid] = dst.vid
        dst.obs = deque(sorted(list(dst.obs) + list(src.obs), key=lambda o: o[_T]))
        dst.type_votes.update(src.type_votes)
        if src.counted and not dst.counted:
            dst.counted = dst.decided = True
            dst.zones = src.zones
        elif not dst.counted:
            zones = []
            for zone, tz in sorted(dst.zones + src.zones, key=lambda z: z[1]):
                if not zones or zones[-1][0] != zone:
                    zones.append((zone, tz))
            dst.zones = zones
        dst.first_t = min(dst.first_t, src.first_t)
        if src.last_t > dst.last_t:
            dst.last_t, dst.last_box = src.last_t, src.last_box
        src.merged_into = dst.vid
        src.decided = True

    def _close_ended(self, t):
        events = []
        for vid in list(self.vehicles):
            v = self.vehicles[vid]
            idle = t - v.last_t
            if not v.decided and v.merged_into is None and idle >= self.cfg['end_gap_s']:
                ev = self._orphan_event(v)
                if ev:
                    events.append(ev)
            if idle >= self.cfg['end_gap_s'] + self.cfg['keep_s'] or (v.merged_into is not None and idle >= self.cfg['keep_s']):
                del self.vehicles[vid]
        if len(self.track_to_vid) > 4 * max(len(self.vehicles), 64):
            self.track_to_vid = {k: x for k, x in self.track_to_vid.items() if x in self.vehicles}
        return events

    def _orphan_event(self, v):
        """Count a vehicle that ended after visiting a single zone, if its motion makes the path clear."""
        v.decided = True
        axis = self._link_axis()
        visited = [z for z, _ in v.zones]
        if axis is None or len(set(visited)) != 1 or len(v.obs) < 2:
            return None
        zone = visited[0]
        other = next(z for z in self.zone_names if z != zone)
        u = self.centroids[other] - self.centroids[zone]
        link_len = float(np.linalg.norm(u))
        u = u / max(link_len, 1e-9)
        travel = float((v.last_xy - np.array(v.obs[0][_X:_Y + 1])) @ u)
        h = max(v.ref_height(), 1.0)
        if abs(travel) / h < self.cfg['orphan_min_travel_h'] or abs(travel) / link_len < self.cfg['orphan_min_frac']:
            return None
        if self._explained_by_other(v):
            return None
        entry, exit_ = (zone, other) if travel > 0 else (other, zone)
        if not self.is_valid_movement(entry, exit_, self.zone_names):
            return None
        v.counted = True
        return self._event(v, entry, exit_, v.zones[0][1], v.last_t, 'inferred')

    def _explained_by_other(self, v):
        """True when a counted vehicle heading the same way was where this one started or ended
        (it is a leftover fragment of that vehicle)."""
        c = self.cfg
        h = max(v.ref_height(), 8.0)
        start = np.array(v.obs[0][_X:_Y + 1])
        ends = [(v.obs[0][_T], start), (v.last_t, v.last_xy)]
        heading = v.last_xy - start
        for u in self.vehicles.values():
            if u is v or not u.counted or not u.obs:
                continue
            if float(heading @ (u.last_xy - np.array(u.obs[0][_X:_Y + 1]))) < 0:
                continue                                  # opposite direction: a different vehicle
            for te, pe in ends:
                if u.first_t - c['orphan_explain_window_s'] <= te <= u.last_t + c['orphan_explain_window_s']:
                    if np.linalg.norm(u.position_at(te, c['velocity_window_s']) - pe) <= c['orphan_explain_radius_h'] * h:
                        return True
        return False

    def _event(self, v, entry, exit_, entry_t, exit_t, method):
        return {
            'vehicle_id': v.vid,
            'type': v.vehicle_type(),
            'entry_zone': entry,
            'exit_zone': exit_,
            'entry_t': float(entry_t),
            'exit_t': float(exit_t),
            'positions': [((o[_X], o[_Y]), o[_T]) for o in v.obs],
            'method': method,
            'track_ids': sorted(v.track_ids),
        }
