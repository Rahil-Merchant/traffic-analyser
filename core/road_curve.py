# project/core/road_curve.py
"""
Arc-length based road coordinate system for speed estimation.

The user draws a series of points along the road's centre-line (the axis
of travel) and provides the real-world length of that axis in meters.

This module builds a dense 1-D metric ruler from those points and provides
fast projection of arbitrary pixel positions onto the curve.

Usage:
    curve = RoadCurve(axis_points, axis_length_meters)
    s = curve.project(px, py)       # → meters along road
    speed = (s2 - s1) / dt * 3.6    # → km/h
"""

import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import cv2
from loguru import logger


# Number of points in the dense centreline resample
_CENTERLINE_DENSITY = 1500


class RoadCurve:
    """A 1-D metric coordinate system along the road's axis.

    Parameters
    ----------
    axis_points : list of (x, y)
        Ordered points drawn by the user along the road centreline / axis,
        from one end of the measured segment to the other.
        Minimum 2 points.
    axis_length_meters : float
        Real-world length of the axis in meters.
    debug_frame : np.ndarray or None
        If provided, a debug image is saved showing the axis and distance
        markers overlaid on this frame.

    After construction the instance exposes:
        centerline  – (M, 2) dense centreline array in pixel coords
        s_meters    – (M,)   cumulative arc-length in meters
        tree        – cKDTree built from centerline for O(log M) queries
    """

    def __init__(self, axis_points, axis_length_meters, debug_frame=None):
        pts = np.array(axis_points, dtype=np.float64)
        n = len(pts)
        if n < 2:
            raise ValueError(f"RoadCurve needs >= 2 axis points, got {n}")
        if axis_length_meters <= 0:
            raise ValueError(f"axis_length_meters must be > 0, got {axis_length_meters}")

        # ---- 0. Calculate axis origin and direction from furthest points ----
        max_dist = -1
        idx1, idx2 = 0, len(pts)-1
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > max_dist:
                    max_dist = d
                    idx1, idx2 = i, j

        # axis_origin must be the NEAR-CAMERA end (larger y in image space).
        # This ensures z=0 → near camera, z=1 → far camera for correct
        # perspective weighting regardless of user drawing order.
        if pts[idx1][1] < pts[idx2][1]:
            # idx1 is higher in image (far), idx2 is lower (near) → swap
            idx1, idx2 = idx2, idx1

        self.axis_origin = pts[idx1].copy()   # near-camera end
        axis_vec = pts[idx2] - pts[idx1]      # points towards far end
        self.axis_length_px = max(float(np.linalg.norm(axis_vec)), 1e-6)
        self.axis_dir = axis_vec / self.axis_length_px

        logger.debug(
            f"[RoadCurve] Axis corners: origin(near)={self.axis_origin} "
            f"→ far={pts[idx2]}, length_px={self.axis_length_px:.1f}"
        )

        # ---- 1. Dense resample -------------------------------------------
        self.centerline = self._resample(pts, _CENTERLINE_DENSITY)

        # ---- 2. Cumulative arc-length in pixels --------------------------
        diffs = np.diff(self.centerline, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        s_pixel = np.concatenate([[0.0], np.cumsum(seg_lens)])

        # ---- 3. Scale to meters ------------------------------------------
        total_px = s_pixel[-1]
        if total_px < 1e-6:
            raise ValueError("Degenerate axis (zero pixel length)")
        self._meters_per_pixel = axis_length_meters / total_px
        self.s_meters = s_pixel * self._meters_per_pixel

        # ---- 4. KD-tree for fast projection ------------------------------
        self.tree = cKDTree(self.centerline)

        self._axis_length_meters = axis_length_meters
        self._raw_pts = pts

        logger.info(
            f"[RoadCurve] Built: {n} axis pts → {len(self.centerline)} "
            f"centreline pts, total_px={total_px:.1f}, "
            f"{axis_length_meters:.1f} m, "
            f"scale={self._meters_per_pixel:.4f} m/px"
        )

        # ---- 5. Optional debug visualisation -----------------------------
        if debug_frame is not None:
            self.save_debug_image(debug_frame)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, px, py):
        """Project a pixel point onto the road axis.

        Returns the arc-length distance (meters) from the start of the axis
        to the closest point on the axis.
        """
        _, idx = self.tree.query([px, py])
        return float(self.s_meters[idx])

    def project_metric(self, px, py):
        """
        Projects a point onto the road axis and applies a non-linear
        depth-weighting correction to account for perspective, returning
        the distance in meters.

        Convention (enforced by __init__):
            axis_origin = near-camera end  (large y in image)
            axis_dir    = points towards far end (small y in image)
            z = 0 → near camera,  z = 1 → far camera

        Perspective correction:
            Near camera: pixels are inflated  → COMPRESS (low weight)
            Far  camera: pixels are compressed → EXPAND   (high weight)
            W(z) = 1 + k·z   (weight increases with depth)
        """
        v = np.array([px, py], dtype=np.float64) - self.axis_origin
        s_px = float(np.dot(v, self.axis_dir))

        # z is the relative position along the axis (0=near, 1=far)
        z = s_px / self.axis_length_px

        # Depth-correction weight: W(z) = 1 + k·z
        #   near (z≈0): weight ≈ 1.0  (compress inflated pixel distances)
        #   far  (z≈1): weight ≈ 1+k  (expand compressed pixel distances)
        k = 2.0
        c1, c2 = 0.05, 0.95

        def W_int(u):
            """Integral of clamped W(t) = 1 + k·t from 0 to u."""
            if u < c1:
                # Below clamp: use W(c1) = 1 + k*c1 as constant
                return u * (1.0 + k * c1)
            elif u <= c2:
                # Main region: ∫(1 + k·t)dt = t + k·t²/2
                int_c1 = c1 * (1.0 + k * c1)
                return int_c1 + (u - c1) + (k / 2.0) * (u**2 - c1**2)
            else:
                # Above clamp: use W(c2) = 1 + k*c2 as constant
                int_c1 = c1 * (1.0 + k * c1)
                int_c2 = int_c1 + (c2 - c1) + (k / 2.0) * (c2**2 - c1**2)
                return int_c2 + (u - c2) * (1.0 + k * c2)

        mapped_z = W_int(z)

        # Scale so W_int(1.0) maps exactly to axis_length_meters
        total_mapped = W_int(1.0)
        return float(mapped_z * (self._axis_length_meters / total_mapped))

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
        """Save a debug image showing the road axis and distance markers."""
        try:
            vis = frame.copy() if frame is not None else np.zeros((600, 800, 3), dtype=np.uint8)

            # Draw raw user-drawn axis points (yellow circles with lines)
            raw = self._raw_pts.astype(np.int32)
            for i in range(len(raw)):
                cv2.circle(vis, tuple(raw[i]), 7, (0, 255, 255), -1)
                cv2.putText(vis, str(i), (raw[i][0] + 10, raw[i][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if i > 0:
                    cv2.line(vis, tuple(raw[i - 1]), tuple(raw[i]),
                             (0, 200, 200), 1, cv2.LINE_AA)

            # Draw dense centreline in green
            cl = self.centerline.astype(np.int32)
            for i in range(len(cl) - 1):
                cv2.line(vis, tuple(cl[i]), tuple(cl[i + 1]),
                         (0, 255, 0), 2, cv2.LINE_AA)

            # Distance markers every 10m (or every road_length/20)
            marker_step = max(10.0, self._axis_length_meters / 20.0)
            next_marker = 0.0
            for i in range(len(self.s_meters)):
                if self.s_meters[i] >= next_marker:
                    pt = tuple(cl[i])
                    cv2.circle(vis, pt, 5, (255, 255, 255), -1)
                    cv2.putText(vis, f"{self.s_meters[i]:.0f}m",
                                (pt[0] + 6, pt[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1)
                    next_marker += marker_step

            # Info overlay
            info = [
                f"Axis: {self._axis_length_meters:.1f}m",
                f"Scale: {self._meters_per_pixel:.4f} m/px",
                f"CL pts: {len(self.centerline)}",
                f"User pts: {len(self._raw_pts)}",
            ]
            for i, line in enumerate(info):
                cv2.putText(vis, line, (10, 22 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

            if output_path is None:
                from config import BASE_OUTPUT_DIR
                os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
                output_path = os.path.join(BASE_OUTPUT_DIR, "road_curve_debug.jpg")

            cv2.imwrite(output_path, vis)
            logger.info(f"[RoadCurve] Debug image saved to: {output_path}")

        except Exception as e:
            logger.warning(f"[RoadCurve] Failed to save debug image: {e}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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
