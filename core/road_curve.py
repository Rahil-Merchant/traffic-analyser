# project/core/road_curve.py
"""
Arc-length based road coordinate system for speed estimation.

Instead of using a homography (which fails on curved / skewed roads),
this module builds a 1-D metric ruler along the road's centerline.

Usage:
    curve = RoadCurve(polygon_points, road_length_meters)
    s = curve.project(px, py)       # → meters along road
    speed = (s2 - s1) / dt * 3.6    # → km/h
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import cv2
from loguru import logger


# Number of points in the dense centerline resample
_CENTERLINE_DENSITY = 1500


class RoadCurve:
    """A 1-D metric coordinate system glued to the road surface.

    Parameters
    ----------
    polygon_points : list of (x, y)
        N >= 4 boundary points drawn by the user, tracing the road perimeter.
    road_length_meters : float
        Real-world length of the road segment in meters.
    road_width_meters : float or None
        Real-world width.  Only used for orientation heuristics when the
        polygon has exactly 4 points; otherwise optional.

    After construction the instance exposes:
        centerline       – (M, 2) dense centerline array in pixel coords
        s_meters         – (M,) cumulative arc-length in meters
        tree             – cKDTree built from centerline for O(log M) queries
    """

    def __init__(self, polygon_points, road_length_meters, road_width_meters=None):
        pts = np.array(polygon_points, dtype=np.float32)
        n = len(pts)
        if n < 4:
            raise ValueError(f"RoadCurve needs >= 4 boundary points, got {n}")

        # ---- 1. Order polygon coherently (CW in screen coords) ----------
        pts_ordered = self._order_polygon(pts)

        # ---- 2. Determine orientation to identify length-sides -----------
        #         Also returns the MABR box [TL, TR, BR, BL] for splitting.
        pts_ordered, box = self._orient_for_length_axis(
            pts_ordered, road_length_meters, road_width_meters
        )

        # ---- 3. Split into two boundaries using MABR corners -------------
        left_boundary, right_boundary = self._split_boundaries(pts_ordered, box)

        logger.debug(
            f"Road boundaries: left={len(left_boundary)} pts, "
            f"right={len(right_boundary)} pts"
        )

        # ---- 4. Pair boundaries → raw centerline -------------------------
        raw_center = self._pair_boundaries(left_boundary, right_boundary)

        # ---- 5. Dense resample -------------------------------------------
        self.centerline = self._resample(raw_center, _CENTERLINE_DENSITY)

        # ---- 6. Cumulative arc-length in pixels --------------------------
        diffs = np.diff(self.centerline, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        s_pixel = np.concatenate([[0.0], np.cumsum(seg_lens)])

        # ---- 7. Scale to meters ------------------------------------------
        total_px = s_pixel[-1]
        if total_px < 1e-6:
            raise ValueError("Degenerate centerline (zero pixel length)")
        meters_per_pixel = road_length_meters / total_px
        self.s_meters = s_pixel * meters_per_pixel

        # ---- 8. KD-tree for fast projection ------------------------------
        self.tree = cKDTree(self.centerline)

        logger.info(
            f"RoadCurve built: {n} boundary pts → {len(self.centerline)} "
            f"centerline pts, {road_length_meters:.1f} m, "
            f"px_scale={meters_per_pixel:.4f} m/px"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, px, py):
        """Project a pixel point onto the road curve.

        Returns the arc-length distance (meters) along the centerline
        to the closest centerline point.
        """
        _, idx = self.tree.query([px, py])
        return float(self.s_meters[idx])

    def project_batch(self, points):
        """Project multiple pixel points at once.

        Parameters
        ----------
        points : array-like of shape (K, 2)

        Returns
        -------
        s_values : ndarray of shape (K,) – meters along road for each point.
        """
        pts = np.asarray(points, dtype=np.float64)
        _, indices = self.tree.query(pts)
        return self.s_meters[indices]

    # ------------------------------------------------------------------
    # Construction helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _order_polygon(pts):
        """Angular sort around centroid → clockwise in screen coords."""
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        return pts[np.argsort(angles)]

    @staticmethod
    def _orient_for_length_axis(pts_ordered, road_length, road_width):
        """Rotate the ordered polygon so that the *length* dimension runs
        along the right and left sides of the CW walk (i.e. the polygon
        starts at TL of the MABR with the top side being the width).

        Returns
        -------
        pts_ordered : ndarray
            The reoriented polygon points.
        box : ndarray of shape (4, 2)
            MABR corners ordered [TL, TR, BR, BL].
        """
        n = len(pts_ordered)

        # Minimum-area bounding rectangle
        rect = cv2.minAreaRect(pts_ordered)
        box = cv2.boxPoints(rect).astype(np.float32)

        # Order box as [TL, TR, BR, BL] CW
        bc = box.mean(axis=0)
        box = box[np.argsort(np.arctan2(box[:, 1] - bc[1], box[:, 0] - bc[0]))]
        tl_idx = int(np.argmin(box[:, 0] + box[:, 1]))
        box = np.roll(box, -tl_idx, axis=0)
        v1, v2 = box[1] - box[0], box[2] - box[1]
        if v1[0] * v2[1] - v1[1] * v2[0] < 0:
            box = np.array([box[0], box[3], box[2], box[1]], dtype=np.float32)

        # Decide which MABR side is the width vs length
        side_top = float(np.linalg.norm(box[1] - box[0]))
        side_right = float(np.linalg.norm(box[2] - box[1]))

        if road_width is not None and road_width > 0:
            target_ratio = road_width / (road_length + 1e-8)
            ratio_top_is_w = side_top / (side_right + 1e-8)
            ratio_right_is_w = side_right / (side_top + 1e-8)
            if abs(ratio_top_is_w - target_ratio) > abs(ratio_right_is_w - target_ratio):
                box = np.roll(box, -1, axis=0)
        else:
            # Without explicit width, assume the shorter MABR side is the width
            if side_top > side_right:
                box = np.roll(box, -1, axis=0)

        # Align polygon start to box TL, walking CW toward TR
        dists_to_tl = np.linalg.norm(pts_ordered - box[0], axis=1)
        start_idx = int(np.argmin(dists_to_tl))
        pts_ordered = np.roll(pts_ordered, -start_idx, axis=0)

        if n >= 3:
            if float(np.linalg.norm(pts_ordered[1] - box[3])) < float(
                np.linalg.norm(pts_ordered[1] - box[1])
            ):
                pts_ordered = np.concatenate(
                    [pts_ordered[:1], pts_ordered[1:][::-1]]
                )

        return pts_ordered, box

    @staticmethod
    def _split_boundaries(pts_ordered, box):
        """Split the CW-ordered polygon into left and right road boundaries
        using MABR corner matching.

        The polygon walks CW from TL: TL → TR → BR → BL → TL.
        We identify the polygon points nearest to each MABR corner and
        extract:
          - Right boundary: polygon segment from near-TR to near-BR
            (runs along the right edge of the road, start → end)
          - Left boundary: polygon segment from near-BL wrapping to near-TL
            (runs along the left edge, end → start), then reversed so both
            boundaries run start → end.

        Parameters
        ----------
        pts_ordered : ndarray of shape (N, 2)
            CW-ordered polygon starting near MABR TL.
        box : ndarray of shape (4, 2)
            MABR corners [TL, TR, BR, BL].

        Returns
        -------
        (left_boundary, right_boundary) : tuple of ndarrays
            Both run from the near-end to the far-end of the road.
        """
        n = len(pts_ordered)

        # Find polygon indices nearest to each MABR corner
        # Index 0 is already near TL by construction
        idx_tr = int(np.argmin(np.linalg.norm(pts_ordered - box[1], axis=1)))
        idx_br = int(np.argmin(np.linalg.norm(pts_ordered - box[2], axis=1)))
        idx_bl = int(np.argmin(np.linalg.norm(pts_ordered - box[3], axis=1)))

        # Right boundary: CW walk from near-TR to near-BR
        # These indices should increase (TR comes before BR in CW walk)
        if idx_tr <= idx_br:
            right_boundary = pts_ordered[idx_tr:idx_br + 1]
        else:
            # Wrapping case (shouldn't normally happen)
            right_boundary = np.concatenate([
                pts_ordered[idx_tr:], pts_ordered[:idx_br + 1]
            ])

        # Left boundary: CW walk from near-BL, wrapping around to near-TL
        # In CW order: BL → left-side points → TL (index 0)
        # We include index 0 (TL) in the chain
        if idx_bl < n:
            left_cw = np.concatenate([
                pts_ordered[idx_bl:], pts_ordered[:1]
            ])
        else:
            left_cw = pts_ordered[:1]

        # Reverse so left boundary also runs start→end (TL→BL direction)
        left_boundary = left_cw[::-1]

        # Sanity: each boundary must have >= 2 points
        if len(right_boundary) < 2 or len(left_boundary) < 2:
            # Fallback: use the polygon endpoints as a simple 2-point boundary
            logger.warning(
                f"Boundary split produced degenerate chains "
                f"(right={len(right_boundary)}, left={len(left_boundary)}). "
                f"Falling back to MABR corners."
            )
            # Right: TR → BR, Left: TL → BL
            right_boundary = np.array([box[1], box[2]], dtype=np.float32)
            left_boundary = np.array([box[0], box[3]], dtype=np.float32)

        return left_boundary, right_boundary

    @staticmethod
    def _pair_boundaries(left, right):
        """Build centerline from two boundary chains that run in the
        same direction (start-of-road → end-of-road).

        Each chain may have a different number of points.  We resample
        both to a common count, then average.
        """
        num = max(len(left), len(right), 100)

        def _resample_chain(chain, count):
            chain = np.asarray(chain, dtype=np.float64)
            diffs = np.diff(chain, axis=0)
            seg = np.sqrt((diffs ** 2).sum(axis=1))
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            total = cum[-1]
            if total < 1e-6:
                return np.tile(chain[0], (count, 1))
            t_old = cum / total
            t_new = np.linspace(0, 1, count)
            fx = interp1d(t_old, chain[:, 0], kind="linear")
            fy = interp1d(t_old, chain[:, 1], kind="linear")
            return np.column_stack([fx(t_new), fy(t_new)])

        left_r = _resample_chain(left, num)
        right_r = _resample_chain(right, num)
        return (left_r + right_r) / 2.0

    @staticmethod
    def _resample(curve, count):
        """Resample an (N, 2) polyline to exactly *count* equi-spaced points."""
        curve = np.asarray(curve, dtype=np.float64)
        diffs = np.diff(curve, axis=0)
        seg = np.sqrt((diffs ** 2).sum(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total < 1e-6:
            return np.tile(curve[0], (count, 1))
        t_old = cum / total
        t_new = np.linspace(0, 1, count)
        fx = interp1d(t_old, curve[:, 0], kind="linear")
        fy = interp1d(t_old, curve[:, 1], kind="linear")
        return np.column_stack([fx(t_new), fy(t_new)])
