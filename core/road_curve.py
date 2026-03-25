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

import os
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
    debug_frame : np.ndarray or None
        If provided, a debug visualization image will be saved showing
        the polygon, boundaries, centerline, and arc-length markers.

    After construction the instance exposes:
        centerline       – (M, 2) dense centerline array in pixel coords
        s_meters         – (M,) cumulative arc-length in meters
        tree             – cKDTree built from centerline for O(log M) queries
    """

    def __init__(self, polygon_points, road_length_meters, road_width_meters=None,
                 debug_frame=None):
        pts = np.array(polygon_points, dtype=np.float32)
        n = len(pts)
        if n < 4:
            raise ValueError(f"RoadCurve needs >= 4 boundary points, got {n}")

        # ---- 1. Order polygon coherently (CW in screen coords) ----------
        pts_ordered = self._order_polygon(pts)
        logger.debug(f"[RoadCurve] Ordered polygon ({n} pts): {pts_ordered.tolist()}")

        # ---- 2. Determine orientation & identify which sides are length --
        pts_ordered, box = self._orient_for_length_axis(
            pts_ordered, road_length_meters, road_width_meters
        )
        logger.debug(f"[RoadCurve] Oriented polygon: {pts_ordered.tolist()}")
        logger.debug(f"[RoadCurve] MABR box [TL,TR,BR,BL]: {box.tolist()}")

        # ---- 3. Split into two length-side boundaries --------------------
        #       Using MABR corners to identify the correct split points
        right_boundary, left_boundary = self._split_boundaries_via_box(
            pts_ordered, box
        )
        logger.info(
            f"[RoadCurve] Boundaries: right={len(right_boundary)} pts, "
            f"left={len(left_boundary)} pts"
        )
        logger.debug(f"[RoadCurve] Right boundary: {right_boundary.tolist()}")
        logger.debug(f"[RoadCurve] Left boundary: {left_boundary.tolist()}")

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
            f"[RoadCurve] Built: {n} boundary pts → {len(self.centerline)} "
            f"centerline pts, total_px={total_px:.1f}, "
            f"{road_length_meters:.1f} m, "
            f"px_scale={meters_per_pixel:.4f} m/px"
        )

        # Store for external use / debug
        self._pts_ordered = pts_ordered
        self._box = box
        self._right_boundary = right_boundary
        self._left_boundary = left_boundary
        self._raw_center = raw_center
        self._meters_per_pixel = meters_per_pixel
        self._road_length_meters = road_length_meters

        # ---- 9. Optional debug visualization -----------------------------
        if debug_frame is not None:
            self.save_debug_image(debug_frame)

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

    def save_debug_image(self, frame, output_path=None):
        """Save a debug visualization showing how the road curve was built.

        The image shows:
        - Original polygon (yellow)
        - MABR box (gray dashed)
        - Right boundary (red) and Left boundary (blue)
        - Centerline (green) with distance markers every 10m
        - Numbered polygon points
        """
        try:
            vis = frame.copy() if frame is not None else np.zeros((600, 800, 3), dtype=np.uint8)

            # Draw MABR box in gray
            box_int = self._box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [box_int], True, (128, 128, 128), 1, cv2.LINE_AA)

            # Draw original polygon in yellow
            poly_int = self._pts_ordered.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [poly_int], True, (0, 255, 255), 2, cv2.LINE_AA)

            # Number each polygon point
            for i, pt in enumerate(self._pts_ordered):
                px_pt = (int(pt[0]), int(pt[1]))
                cv2.circle(vis, px_pt, 5, (0, 255, 255), -1)
                cv2.putText(vis, str(i), (px_pt[0] + 7, px_pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Draw right boundary in red (with arrows)
            for i in range(len(self._right_boundary) - 1):
                p1 = tuple(self._right_boundary[i].astype(int))
                p2 = tuple(self._right_boundary[i + 1].astype(int))
                cv2.line(vis, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)
            # Mark start/end of right boundary
            if len(self._right_boundary) > 0:
                cv2.circle(vis, tuple(self._right_boundary[0].astype(int)), 8, (0, 0, 255), -1)
                cv2.putText(vis, "R-start", tuple(self._right_boundary[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # Draw left boundary in blue (with arrows)
            for i in range(len(self._left_boundary) - 1):
                p1 = tuple(self._left_boundary[i].astype(int))
                p2 = tuple(self._left_boundary[i + 1].astype(int))
                cv2.line(vis, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)
            if len(self._left_boundary) > 0:
                cv2.circle(vis, tuple(self._left_boundary[0].astype(int)), 8, (255, 0, 0), -1)
                cv2.putText(vis, "L-start", tuple(self._left_boundary[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

            # Draw centerline in green with distance markers
            cl = self.centerline.astype(np.int32)
            for i in range(len(cl) - 1):
                cv2.line(vis, tuple(cl[i]), tuple(cl[i + 1]), (0, 255, 0), 2, cv2.LINE_AA)

            # Draw distance markers every 10m
            marker_step = max(10.0, self._road_length_meters / 20.0)
            next_marker = 0.0
            for i in range(len(self.s_meters)):
                if self.s_meters[i] >= next_marker:
                    pt = tuple(cl[i])
                    cv2.circle(vis, pt, 4, (255, 255, 255), -1)
                    cv2.putText(vis, f"{self.s_meters[i]:.0f}m", (pt[0] + 5, pt[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    next_marker += marker_step

            # Add info text
            info_lines = [
                f"Road: {self._road_length_meters:.1f}m",
                f"Scale: {self._meters_per_pixel:.4f} m/px",
                f"CL pts: {len(self.centerline)}",
                f"R-bnd: {len(self._right_boundary)} pts",
                f"L-bnd: {len(self._left_boundary)} pts",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(vis, line, (10, 20 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Legend
            y_leg = vis.shape[0] - 80
            cv2.putText(vis, "Yellow: polygon", (10, y_leg), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(vis, "Red: right boundary", (10, y_leg + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(vis, "Blue: left boundary", (10, y_leg + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(vis, "Green: centerline", (10, y_leg + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            if output_path is None:
                from config import BASE_OUTPUT_DIR
                os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
                output_path = os.path.join(BASE_OUTPUT_DIR, "road_curve_debug.jpg")
            cv2.imwrite(output_path, vis)
            logger.info(f"[RoadCurve] Debug image saved to: {output_path}")

        except Exception as e:
            logger.warning(f"[RoadCurve] Failed to save debug image: {e}")

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
        along the right and left sides of the CW walk.

        Returns (pts_ordered, box) where box is the MABR [TL, TR, BR, BL].
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
    def _split_boundaries_via_box(pts_ordered, box):
        """Split the CW-ordered polygon into two length-side boundary chains
        using the MABR corner positions to correctly identify side transitions.

        The box is [TL, TR, BR, BL] (CW).  The polygon walks CW from near TL.

        For the centerline we need the two LENGTH-running sides:
          - Right boundary:  TR → BR  (right side of road)
          - Left boundary:   TL → BL  (left side of road)

        We find the polygon vertices nearest to each MABR corner to identify
        which points belong to which side.
        """
        n = len(pts_ordered)
        TL, TR, BR, BL = box[0], box[1], box[2], box[3]

        # Find polygon indices closest to each corner
        def nearest_idx(target):
            dists = np.linalg.norm(pts_ordered - target, axis=1)
            return int(np.argmin(dists))

        idx_TL = nearest_idx(TL)
        idx_TR = nearest_idx(TR)
        idx_BR = nearest_idx(BR)
        idx_BL = nearest_idx(BL)

        logger.debug(
            f"[RoadCurve] Corner indices: TL={idx_TL}, TR={idx_TR}, "
            f"BR={idx_BR}, BL={idx_BL}"
        )

        # Extract the right boundary: from TR → BR (CW walk)
        # These are the polygon points between idx_TR and idx_BR
        right_boundary = _extract_cw_segment(pts_ordered, idx_TR, idx_BR)

        # Extract the left boundary: from BL → TL (CW walk), then reverse
        # So we get points running TL → BL (same direction as right: top→bottom)
        left_boundary_raw = _extract_cw_segment(pts_ordered, idx_BL, idx_TL)
        left_boundary = left_boundary_raw[::-1]  # Reverse so it runs TL → BL

        return right_boundary, left_boundary

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
            if len(chain) < 2:
                return np.tile(chain[0], (count, 1))
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
        if len(curve) < 2:
            return np.tile(curve[0], (count, 1))
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


def _extract_cw_segment(pts_ordered, idx_start, idx_end):
    """Extract a segment of the CW-ordered polygon from idx_start to idx_end
    (inclusive), wrapping around if necessary."""
    n = len(pts_ordered)
    indices = []
    i = idx_start
    while True:
        indices.append(i)
        if i == idx_end:
            break
        i = (i + 1) % n
        if len(indices) > n:  # Safety: prevent infinite loop
            break
    return np.array([pts_ordered[i] for i in indices], dtype=np.float32)
