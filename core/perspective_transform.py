# project/core/perspective_transform.py
"""
Perspective-transform based coordinate system for speed estimation.

The user marks 4 corners of a road rectangle (TL, TR, BR, BL) on the
video frame and provides the real-world width and length of that rectangle.

A homography matrix maps pixel coordinates to a metric 2D plane, enabling
perspective-free distance and speed computation.

Usage:
    pt = PerspectiveTransformer(corners, width_m, length_m)
    x, y = pt.transform_point(px, py)  # → (meters_x, meters_y)
    speed = euclidean_dist / dt * 3.6   # → km/h
"""

import os
import numpy as np
import cv2
from loguru import logger


class PerspectiveTransformer:
    """Perspective transform from pixel space to metric 2D plane.

    Parameters
    ----------
    corners : list of 4 (x, y) tuples
        Ordered corners in pixel space: [TL, TR, BR, BL].
    width_meters : float
        Real-world width of the rectangle (TL→TR edge) in meters.
    length_meters : float
        Real-world length of the rectangle (TL→BL edge) in meters.
    debug_frame : np.ndarray or None
        If provided, a warped bird's-eye debug image is saved.

    After construction the instance exposes:
        M             – (3, 3) perspective transform matrix
        M_inv         – (3, 3) inverse transform (metric → pixel)
        width_meters  – real-world width
        length_meters – real-world length
    """

    def __init__(self, corners, width_meters, length_meters, debug_frame=None):
        if len(corners) != 4:
            raise ValueError(f"PerspectiveTransformer needs exactly 4 corners, got {len(corners)}")
        if width_meters <= 0:
            raise ValueError(f"width_meters must be > 0, got {width_meters}")
        if length_meters <= 0:
            raise ValueError(f"length_meters must be > 0, got {length_meters}")

        self.width_meters = float(width_meters)
        self.length_meters = float(length_meters)
        self._raw_corners = np.array(corners, dtype=np.float32)

        # Source points: user-drawn corners in pixel space [TL, TR, BR, BL]
        src_pts = self._raw_corners

        # Destination points: metric rectangle
        # TL → (0, 0), TR → (W, 0), BR → (W, L), BL → (0, L)
        dst_pts = np.array([
            [0.0, 0.0],
            [self.width_meters, 0.0],
            [self.width_meters, self.length_meters],
            [0.0, self.length_meters],
        ], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

        logger.info(
            f"[PerspectiveTransformer] Built: corners={corners}, "
            f"width={self.width_meters:.1f}m, length={self.length_meters:.1f}m"
        )

        # Optional debug visualisation
        if debug_frame is not None:
            self.save_debug_frame(debug_frame)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform_point(self, px, py):
        """Transform a single pixel coordinate to metric (x, y).

        Returns
        -------
        (x_meters, y_meters) : tuple of float
        """
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.M)
        return float(result[0, 0, 0]), float(result[0, 0, 1])

    def transform_batch(self, points):
        """Transform multiple pixel coordinates to metric space.

        Parameters
        ----------
        points : array-like of shape (K, 2)

        Returns
        -------
        metric_coords : np.ndarray of shape (K, 2)
        """
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        result = cv2.perspectiveTransform(pts, self.M)
        return result.reshape(-1, 2)

    def save_debug_frame(self, frame, output_dir=None):
        """Save a warped bird's-eye view of the full frame for debugging."""
        try:
            if output_dir is None:
                from config import BASE_OUTPUT_DIR
                output_dir = BASE_OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)

            # Determine output size in pixels for the warped image.
            # Use a scale factor so the output is reasonably sized.
            # Determine output size in pixels for the warped image.
            scale_px_per_m = 20.0  # 20 pixels per meter
            out_w = int(self.width_meters * scale_px_per_m)
            out_h = int(self.length_meters * scale_px_per_m)
            
            # Create a scaling matrix for the visualization
            # This converts the metric output of self.M back into pixels for the debug image
            scale_matrix = np.array([
                [scale_px_per_m, 0, 0],
                [0, scale_px_per_m, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Combine the original matrix with the scaling matrix
            viz_M = scale_matrix @ self.M

            # Create the birds-eye view using the scaled matrix
            birds_eye = cv2.warpPerspective(frame, viz_M, (out_w, out_h))

            # Draw the quadrilateral on original frame for visual reference
            vis = frame.copy()
            corners_int = self._raw_corners.astype(np.int32)
            labels = ['TL', 'TR', 'BR', 'BL']
            colors = [(0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255)]

            for i in range(4):
                pt1 = tuple(corners_int[i])
                pt2 = tuple(corners_int[(i + 1) % 4])
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(vis, pt1, 8, colors[i], -1)
                cv2.putText(vis, f"{labels[i]} ({i+1})", (pt1[0] + 12, pt1[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

            # Info overlay
            info = [
                f"Width: {self.width_meters:.1f}m",
                f"Length: {self.length_meters:.1f}m",
            ]
            for i, line in enumerate(info):
                cv2.putText(vis, line, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save both images
            corners_path = os.path.join(output_dir, "perspective_corners_debug.jpg")
            birds_eye_path = os.path.join(output_dir, "perspective_birds_eye_debug.jpg")

            cv2.imwrite(corners_path, vis)
            cv2.imwrite(birds_eye_path, birds_eye)
            logger.info(f"[PerspectiveTransformer] Debug images saved to: {output_dir}")

        except Exception as e:
            logger.warning(f"[PerspectiveTransformer] Failed to save debug images: {e}")
