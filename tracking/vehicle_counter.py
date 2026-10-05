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
- Duplicates: a new track that overlaps a vehicle seen in the same frame, on the same
  footprint, is that vehicle. Two live vehicles merge only when they overlap and move together.
- Reversals: vehicles do not turn around on the road. If a vehicle's motion reverses, two
  vehicles were joined (far from the camera, an oncoming car can appear exactly where a
  receding one vanished, and the tracker can swap IDs there). The vehicle is split at the
  join, or at the turning point, and each part is counted on its own merits; a count that
  only existed because of the other vehicle's observations is retracted.
- Crossings: zone visits are taken along the path between consecutive observations, so a
  fast vehicle with only a few detections still registers every zone it passed through.
- Orphans (2-zone links only): a vehicle that ends after visiting one zone is counted only
  with clear travel along the link, when no counted vehicle explains it. Its direction comes
  from its motion, so entry and exit are never the same zone.
- Class: votes are weighted by confidence and box size, so the class comes from close, clear
  views rather than from many tiny detections far from the camera.
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


def same_footprint(a, b):
    """Bottom edges coincide: what two boxes on one vehicle share and two passing vehicles do not."""
    w = min(a[2] - a[0], b[2] - b[0])
    h = min(a[3] - a[1], b[3] - b[1])
    return abs((a[0] + a[2]) - (b[0] + b[2])) / 2 <= 0.5 * w and abs(a[3] - b[3]) <= 0.25 * h


def _ls_velocity(obs):
    """Least-squares image velocity (px/s) of observation tuples, or None."""
    if len(obs) < 2:
        return None
    ts = np.array([o[_T] for o in obs])
    if ts[-1] - ts[0] < 0.04:
        return None
    xy = np.array([(o[_X], o[_Y]) for o in obs], dtype=float)
    tc = ts - ts.mean()
    return (tc[:, None] * (xy - xy.mean(axis=0))).sum(axis=0) / (tc ** 2).sum()


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
    # Duplicates
    'dup_iou': 0.6,                  # same-frame overlap that can make two tracks one vehicle
    'dup_frames': 3,                 # consecutive frames of overlap + common motion before two vehicles merge
    'co_moving_cos': 0.7,            # min cosine between velocities of two boxes on one vehicle
    # Reversal split
    'reversal_window_s': 0.6,        # motion over the last window is compared with the window before it
    'reversal_min_obs': 5,           # observations needed in each window
    'reversal_min_disp_h': 1.0,      # each window must travel at least this many box heights
    'reversal_cos': -0.5,            # directions this opposed mean two different vehicles
    'reversal_check_every_s': 0.25,
    # Vehicle end, orphans
    'end_gap_s': 3.5,                # unseen this long -> vehicle ended (must exceed both stitch gaps)
    'orphan_min_travel_h': 1.5,      # orphan must travel this many box heights along the link...
    'orphan_min_frac': 0.10,         # ...and this fraction of the distance between the two zones
    'orphan_explain_radius_h': 3.0,  # a counted vehicle this close (box heights) explains an orphan
    'orphan_explain_window_s': 2.0,
    'keep_s': 15.0,                  # history kept after a vehicle ends (for orphan explanation)
    'velocity_window_s': 0.75,
    'max_positions': 2400,
    'track_history': 200,            # per-track observations kept for splitting vehicles
    # Class votes: weight = confidence * min(box height, cap)^2, so close views dominate
    'class_min_h': 12.0,             # boxes shorter than this (px) only vote if nothing larger was seen
    'class_h_cap': 200.0,
    'class_edge_factor': 0.25,       # boxes cut off by the frame edge vote at this weight
}

# Observation tuple layout
_T, _X, _Y, _H, _BOX, _GX, _GY, _GW, _TID, _TYPE, _W = range(11)


class _Vehicle:
    def __init__(self, vid, t):
        self.vid = vid
        self.track_ids = set()
        self.track_joined = {}              # track id -> time it joined this vehicle
        self.obs = deque()                  # one observation per frame, see layout above
        self.type_votes = Counter()         # class -> summed vote weight (boxes >= class_min_h)
        self.small_votes = Counter()        # same for tiny boxes, used only if nothing larger was seen
        self.zones = []                     # ordered zone visits: (zone, t)
        self.counted = False
        self.count_t = None
        self.count_move = None              # (entry, exit, entry_t, exit_t, method) once counted
        self.emitted = False                # count published (when the vehicle has left the scene)
        self.decided = False                # final decision taken (counted, dropped or merged)
        self.merged_into = None
        self.split_partner = None           # the other half of a reversal split
        self.first_t = t
        self.last_t = t
        self.last_box = None
        self.next_reversal_check = t

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

    def vote(self, o, min_h):
        (self.type_votes if o[_H] >= min_h else self.small_votes)[o[_TYPE]] += o[_W]

    def recount_votes(self, min_h):
        self.type_votes, self.small_votes = Counter(), Counter()
        for o in self.obs:
            self.vote(o, min_h)

    def vehicle_type(self):
        votes = self.type_votes or self.small_votes
        return votes.most_common(1)[0][0] if votes else 'Light Vehicle'


class VehicleCounter:
    """Feed one frame at a time with update(); returns events.

    Whether and where a vehicle is counted is decided when it crosses; the count is published once
    the vehicle has left the scene, so its class and path use every observation (an oncoming
    vehicle's closest views come after it crosses). An event is a counted vehicle path (dict), or
    {'retract': True, 'vehicle_id': ...} if a published count turns out to rest on another
    vehicle's observations.

    ground_fn, if given, maps (N,2) image points to (N,2) road-plane metres (NaN where unreliable);
    the second ground axis must run along the road. frame_size (w, h) lets class votes discount
    boxes cut off by the frame edge.
    """

    def __init__(self, zones, is_valid_movement=None, non_vehicle_types=(), config=None, ground_fn=None,
                 frame_size=None):
        self.zones = {k: np.asarray(v, dtype=float) for k, v in zones.items()}
        self._zone_edges = {k: polygon_edges(v) for k, v in self.zones.items()}
        self.zone_names = list(self.zones)
        self.centroids = {k: v.mean(axis=0) for k, v in self.zones.items()}
        self.is_valid_movement = is_valid_movement or (lambda a, b, names: a != b)
        self.non_vehicle_types = set(non_vehicle_types)
        self.cfg = dict(DEFAULT_CONFIG, **(config or {}))
        self.ground_fn = ground_fn
        self.frame_size = frame_size
        self.vehicles = {}
        self.track_to_vid = {}
        self.track_hist = {}                # track id -> deque of its own observations
        self._next_vid = 1
        self._dup_streak = Counter()

    # ---------- public API ----------
    def update(self, t, tracks):
        """tracks: iterable of dicts with track_id, box (x1,y1,x2,y2 in zone coordinates), type, score."""
        events = []
        seen = {}                                     # vid -> box observed this frame
        obs_of = {}
        for tr in tracks:
            o = self._make_obs(tr, t)
            obs_of[tr['track_id']] = o
            hist = self.track_hist.setdefault(tr['track_id'], deque(maxlen=self.cfg['track_history']))
            hist.append(o)
        known = [tr for tr in tracks if self._vid_of(tr['track_id']) is not None]
        # The longest-attached track speaks for its vehicle, so one vehicle's path is never a zig-zag of two tracks
        known.sort(key=lambda tr: self.vehicles[self._vid_of(tr['track_id'])].track_joined.get(tr['track_id'], math.inf))
        new = sorted((tr for tr in tracks if self._vid_of(tr['track_id']) is None),
                     key=lambda tr: -tr.get('score', 0.0))
        for tr in known:
            v = self.vehicles[self._vid_of(tr['track_id'])]
            if v.vid not in seen:
                self._observe(v, obs_of[tr['track_id']], events, seen)
        for tr in new:
            o = obs_of[tr['track_id']]
            v = (self._duplicate_of(tr, seen) or self._stitch_target(tr, o, t, seen)
                 or self._new_vehicle(t))
            self.track_to_vid[tr['track_id']] = v.vid
            v.track_ids.add(tr['track_id'])
            v.track_joined.setdefault(tr['track_id'], t)
            if v.vid not in seen:
                self._observe(v, o, events, seen)
        for vid in list(seen):
            v = self.vehicles.get(vid)
            if v is not None and v.merged_into is None and t >= v.next_reversal_check:
                v.next_reversal_check = t + self.cfg['reversal_check_every_s']
                self._check_reversal(v, t, events)
        self._merge_concurrent(seen)
        events += self._close_ended(t)
        return events

    def finalize(self, t):
        """End of video: decide every vehicle still pending."""
        events = []
        for v in list(self.vehicles.values()):
            if v.merged_into is not None:
                continue
            if not v.decided:
                self._orphan_decision(v)
            if v.counted and not v.emitted:
                events.append(self._emit(v))
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

    def _new_vehicle(self, t):
        v = _Vehicle(self._next_vid, t)
        self.vehicles[v.vid] = v
        self._next_vid += 1
        return v

    @staticmethod
    def _bottom_centre(box):
        return np.array([(box[0] + box[2]) / 2.0, float(box[3])])

    def _clipped(self, box):
        """Box cut off by the frame edge (its bottom-centre and size are not the vehicle's)."""
        if self.frame_size is None:
            return False
        fw, fh = self.frame_size
        return box[0] <= 1 or box[1] <= 1 or box[2] >= fw - 2 or box[3] >= fh - 2

    def _class_weight(self, box, score):
        h = box[3] - box[1]
        w = max(score, 0.01) * min(h, self.cfg['class_h_cap']) ** 2
        return w * self.cfg['class_edge_factor'] if self._clipped(box) else w

    def _make_obs(self, tr, t):
        box = [float(c) for c in tr['box']]
        p = self._bottom_centre(box)
        gx, gy, gw = self._ground(box)[:3]
        return (t, p[0], p[1], max(box[3] - box[1], 1.0), box, gx, gy, gw, tr['track_id'], tr['type'],
                self._class_weight(box, tr.get('score', 0.5)))

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

    def _zones_of(self, obs):
        """Zone visits along a whole observation sequence."""
        zones = []
        prev = None
        for o in obs:
            p = np.array([o[_X], o[_Y]])
            if prev is None:
                z = self._zone_at(p[None])[0]
                visits = [(z, o[_T])] if z is not None else []
            else:
                visits = self._zones_along(np.array([prev[_X], prev[_Y]]), p, prev[_T], o[_T])
            for zone, tz in visits:
                if not zones or zones[-1][0] != zone:
                    zones.append((zone, tz))
            prev = o
        return zones

    def _observe(self, v, o, events, seen):
        if v.obs:
            visits = self._zones_along(v.last_xy, np.array([o[_X], o[_Y]]), v.last_t, o[_T])
        else:
            z = self._zone_at(np.array([[o[_X], o[_Y]]]))[0]
            visits = [(z, o[_T])] if z is not None else []
        for zone, tz in visits:
            if not v.zones or v.zones[-1][0] != zone:
                v.zones.append((zone, tz))
        v.obs.append(o)
        while len(v.obs) > self.cfg['max_positions']:
            v.obs.popleft()
        v.vote(o, self.cfg['class_min_h'])
        v.last_t, v.last_box = o[_T], o[_BOX]
        seen[v.vid] = o[_BOX]
        if not v.counted:
            self._try_crossing(v, events, seen)

    def _crossing(self, v):
        """(entry, exit, entry_t, exit_t) of the first valid zone-to-zone move, or None."""
        entry = v.zones[0][0] if v.zones else None
        for zone, tz in v.zones[1:]:
            if zone != entry and self.is_valid_movement(entry, zone, self.zone_names):
                return entry, zone, v.zones[0][1], tz
        return None

    def _try_crossing(self, v, events, seen):
        move = self._crossing(v)
        if move is None:
            return
        twin = self._counted_twin(v, seen)
        if twin is not None:
            self._merge(v, twin)                # duplicate track of an already counted vehicle
            return
        if self._same_as_partner(v, move[0], move[1]):
            return
        v.counted = v.decided = True
        v.count_t = move[3]
        v.count_move = (*move, 'crossing')

    def _same_as_partner(self, v, entry, exit_):
        """Halves of a real reversal travel opposite ways; if this half would repeat its partner's
        counted movement, the split was a false alarm, so fold it back into the partner."""
        u = self.vehicles.get(v.split_partner) if v.split_partner is not None else None
        if u is None or u.merged_into is not None or not u.counted or u.count_move[:2] != (entry, exit_):
            return False
        self._merge(v, u)
        return True

    def _co_moving(self, a, b):
        """True if two vehicles move together, False if not, None if motion is unknown for either."""
        w = self.cfg['velocity_window_s']
        va, vb = a.velocity(w), b.velocity(w)
        if va is None or vb is None:
            return None
        ha, hb = max(a.ref_height(), 1.0), max(b.ref_height(), 1.0)
        slow_a, slow_b = np.linalg.norm(va) < 0.5 * ha, np.linalg.norm(vb) < 0.5 * hb
        if slow_a or slow_b:
            return slow_a and slow_b
        return float(va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb)) >= self.cfg['co_moving_cos']

    def _counted_twin(self, v, seen):
        for vid, box in seen.items():
            u = self.vehicles.get(vid)
            if u is not None and u is not v and u.counted and box_iou(box, v.last_box) >= self.cfg['dup_iou'] \
                    and same_footprint(box, v.last_box) and self._co_moving(u, v):
                return u
        return None

    def _duplicate_of(self, tr, seen):
        # Class is ignored: a box this similar on the same footprint is the same object whatever it is called.
        # Two vehicles passing far away can still look like this; a reversal split separates them later.
        best, best_iou = None, self.cfg['dup_iou']
        for vid, box in seen.items():
            iou = box_iou(box, tr['box'])
            if iou >= best_iou and same_footprint(box, tr['box']):
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

    def _stitch_target(self, tr, o, t, seen):
        p = np.array([o[_X], o[_Y]])
        g_new = np.array([o[_GX], o[_GY]])
        ground_new = self._ground(o[_BOX])
        kind = self._kind(tr['type'])
        axis = self._link_axis()
        best, best_d = None, None
        for v in self.vehicles.values():
            if v.vid in seen or v.merged_into is not None or self._kind(v.vehicle_type()) != kind \
                    or (v.decided and not v.counted) or not v.obs:
                continue
            dt = t - v.last_t
            if dt <= 0:
                continue
            if np.all(np.isfinite(g_new)) and np.all(np.isfinite(v.last_ground)):
                d = self._gate_ground(v, ground_new, dt)
            else:
                d = self._gate_image(v, p, o[_H], dt, axis)
            if d is not None and (best_d is None or d < best_d):
                best, best_d = v, d
        return best

    def _check_reversal(self, v, t, events):
        """Split a vehicle whose motion reverses: it is two vehicles joined by a stitch, a duplicate
        match or a tracker ID swap."""
        c = self.cfg
        w = c['reversal_window_s']
        clean = [o for o in v.obs if o[_T] >= t - 2 * w and not self._clipped(o[_BOX])]   # clipped boxes shift
        recent = [o for o in clean if o[_T] >= t - w]
        prev = [o for o in clean if o[_T] < t - w]
        if len(recent) < c['reversal_min_obs'] or len(prev) < c['reversal_min_obs']:
            return
        v_new, v_old = _ls_velocity(recent), _ls_velocity(prev)
        if v_new is None or v_old is None:
            return
        h = float(np.median([o[_H] for o in clean]))
        n_new, n_old = np.linalg.norm(v_new), np.linalg.norm(v_old)
        # Each window must show real travel (a box height or more), not jitter
        if n_new * w < c['reversal_min_disp_h'] * h or n_old * w < c['reversal_min_disp_h'] * h:
            return
        if float(v_new @ v_old) / (n_new * n_old) > c['reversal_cos']:
            return
        # Turning point: furthest progress along the earlier direction
        d_old = v_old / n_old
        t_turn = clean[int(np.argmax([o[_X] * d_old[0] + o[_Y] * d_old[1] for o in clean]))][_T]
        joins = [tj for tid, tj in v.track_joined.items() if tj > v.first_t and abs(tj - t_turn) <= 0.75]
        if joins:                       # a track joined at the turn: it belongs to the other vehicle
            t_split = min(joins, key=lambda tj: abs(tj - t_turn))
            moved = {tid for tid, tj in v.track_joined.items() if tj >= t_split}
            # Confirm with the joined tracks' own motion, measured without the vehicle's other tracks
            own = [o for tid in moved for o in self.track_hist.get(tid, ()) if not self._clipped(o[_BOX])]
            v_own = _ls_velocity(sorted(own, key=lambda o: o[_T])) if len(own) >= c['reversal_min_obs'] else None
            if v_own is None or float(v_own @ d_old) / max(np.linalg.norm(v_own), 1e-9) > c['reversal_cos']:
                return
            swapped = set()
        else:                           # the tracker itself swapped vehicles: split its tracks in time
            t_split = t_turn + 1e-6
            moved = set()
            swapped = {o[_TID] for o in v.obs if o[_T] >= t_split}
        self._split(v, t_split, moved, swapped, events)

    def _split(self, v, t_split, moved, swapped, events):
        tids = moved | swapped
        take = [o for o in v.obs if o[_TID] in moved or (o[_TID] in swapped and o[_T] >= t_split)]
        keep = [o for o in v.obs if not (o[_TID] in moved or (o[_TID] in swapped and o[_T] >= t_split))]
        if not take or not keep:
            return
        # Secondary observations of the moved tracks (frames where another track spoke for the vehicle)
        have = {o[_T] for o in take}
        for tid in tids:
            for o in self.track_hist.get(tid, ()):
                if o[_T] >= t_split and o[_T] not in have and (tid in moved or o[_T] >= t_split):
                    take.append(o)
                    have.add(o[_T])
        take.sort(key=lambda o: o[_T])
        nv = self._new_vehicle(take[0][_T])
        nv.obs = deque(take)
        nv.recount_votes(self.cfg['class_min_h'])
        nv.zones = self._zones_of(take)
        nv.last_t, nv.last_box = take[-1][_T], take[-1][_BOX]
        nv.next_reversal_check = v.next_reversal_check
        nv.split_partner, v.split_partner = v.vid, nv.vid
        for tid in tids:
            v.track_ids.discard(tid)
            nv.track_ids.add(tid)
            nv.track_joined[tid] = max(v.track_joined.pop(tid, t_split), t_split)
            self.track_to_vid[tid] = nv.vid
        v.obs = deque(keep)
        v.recount_votes(self.cfg['class_min_h'])
        v.last_t, v.last_box = keep[-1][_T], keep[-1][_BOX]
        v.next_reversal_check = nv.next_reversal_check + 1.0
        if not v.counted or v.count_t >= t_split:
            v.zones = self._zones_of(keep)
            if v.counted and self._crossing(v) is None:
                # The count rested on the other vehicle's observations
                if v.emitted:
                    events.append({'retract': True, 'vehicle_id': v.vid})
                v.counted = v.decided = v.emitted = False
                v.count_t = v.count_move = None
            if not v.counted:
                self._try_crossing(v, events, {})
        self._try_crossing(nv, events, {})

    def _merge_concurrent(self, seen):
        """Merge two vehicles that keep overlapping on one footprint and moving together (duplicate tracks)."""
        vids = sorted(seen)
        live = set()
        for i, a in enumerate(vids):
            for b in vids[i + 1:]:
                va, vb = self.vehicles.get(a), self.vehicles.get(b)
                if va is None or vb is None or box_iou(seen[a], seen[b]) < self.cfg['dup_iou'] \
                        or not same_footprint(seen[a], seen[b]) or not self._co_moving(va, vb):
                    continue
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
            dst.track_joined.setdefault(tid, src.track_joined.get(tid, src.first_t))
        dst.obs = deque(sorted(list(dst.obs) + list(src.obs), key=lambda o: o[_T]))
        dst.type_votes.update(src.type_votes)
        dst.small_votes.update(src.small_votes)
        if src.counted and not dst.counted:
            dst.counted = dst.decided = True
            dst.count_t, dst.count_move, dst.emitted = src.count_t, src.count_move, src.emitted
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
            if v.merged_into is None and idle >= self.cfg['end_gap_s']:
                if not v.decided:
                    self._orphan_decision(v)
                if v.counted and not v.emitted:
                    events.append(self._emit(v))
            if idle >= self.cfg['end_gap_s'] + self.cfg['keep_s'] or (v.merged_into is not None and idle >= self.cfg['keep_s']):
                del self.vehicles[vid]
        if len(self.track_to_vid) > 4 * max(len(self.vehicles), 64):
            self.track_to_vid = {k: x for k, x in self.track_to_vid.items() if x in self.vehicles}
        if len(self.track_hist) > 4 * max(len(self.vehicles), 64):
            self.track_hist = {k: h for k, h in self.track_hist.items() if h and t - h[-1][_T] < self.cfg['keep_s']}
        return events

    def _orphan_decision(self, v):
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
        if not self.is_valid_movement(entry, exit_, self.zone_names) or self._same_as_partner(v, entry, exit_):
            return None
        v.counted = True
        v.count_t = v.last_t
        v.count_move = (entry, exit_, v.zones[0][1], v.last_t, 'inferred')

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

    def _emit(self, v):
        v.emitted = True
        return self._event(v, *v.count_move)

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
