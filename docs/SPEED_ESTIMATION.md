# Perspective Transformation & Speed Estimation

This document explains the mathematical foundations, trajectory signal processing, and calibration workflow for speed estimation in the Urban Reconnaissance platform.

---

## Mathematical Formulation: Homography Mapping

Traffic surveillance cameras capture road scenes under severe foreshortening and perspective distortion: a pixel distance in the foreground represents a few centimeters, while the same pixel distance near the horizon represents tens of meters.

To measure real-world velocities ($v = \Delta d / \Delta t$) accurately, pixel coordinates must be mapped to a metric 2D ground plane.

```
 Camera Frame (Pixel Space)                      Metric Ground Plane (Meters)
 
       TL --------- TR                                 (0,0) -------- (W,0)
       /             \                                   |              |
      /   [Vehicle]   \       --- Homography H --->      |   [Vehicle]  |
     /                 \                                 |              |
   BL ----------------- BR                             (0,L) -------- (W,L)
```

### Homography Matrix Computation
Given 4 non-collinear user-marked calibration corners in pixel space:
$$\mathbf{p}_i = [u_i, v_i, 1]^T, \quad i \in \{\text{TL}, \text{TR}, \text{BR}, \text{BL}\}$$
and their corresponding metric coordinates on the road surface:
$$\mathbf{p}'_{\text{TL}} = [0, 0, 1]^T, \quad \mathbf{p}'_{\text{TR}} = [W, 0, 1]^T, \quad \mathbf{p}'_{\text{BR}} = [W, L, 1]^T, \quad \mathbf{p}'_{\text{BL}} = [0, L, 1]^T$$
where $W$ is the measured road width in meters and $L$ is the road length in meters.

The $3 \times 3$ perspective transformation matrix $\mathbf{H}$ satisfies:
$$\mathbf{p}'_i \sim \mathbf{H} \mathbf{p}_i$$
$\mathbf{H}$ is computed via OpenCV's `cv2.getPerspectiveTransform()`. 

To transform any arbitrary pixel coordinate $(u, v)$ to real-world ground meters $(x, y)$:
$$\begin{bmatrix} x_h \\ y_h \\ w_h \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad x = \frac{x_h}{w_h}, \quad y = \frac{y_h}{w_h}$$

---

## Vehicle Ground Contact Point

Speed estimation requires tracking the point where the vehicle touches the road surface, not the bounding box centroid (which floats in 3D space above the road and shifts depending on vehicle height):
$$\text{Contact Point} = \left( \frac{x_1 + x_2}{2}, \; y_2 \right)$$
where $[x_1, y_1, x_2, y_2]$ is the bounding box in processing coordinates.

---

## Trajectory Signal Processing Pipeline

Raw detection bounding boxes jitter by several pixels frame-to-frame. In perspective-transformed metric space, high-frequency jitter inflates measured travel distances dramatically.

The platform applies a 4-stage filter pipeline (`core/video_processor.py`):

```
 Raw Trajectory Points (x, y, t)
                |
                v
 +------------------------------+
 | 1. Temporal Decimation Check |  Requires min duration (0.4s) & points (>= 3)
 +--------------+---------------+
                |
                v
 +------------------------------+
 | 2. Median Outlier Filter     |  Window size = 5 (Drops single-frame detection spikes)
 +--------------+---------------+
                |
                v
 +------------------------------+
 | 3. Moving Average Smoothing  |  Window size = 9 (Eliminates high-frequency jitter)
 +--------------+---------------+
                |
                v
 +------------------------------+
 | 4. Cumulative Distance / dt  |  Sum of Euclidean segment lengths divided by delta t
 +--------------+---------------+
                |
                v
 +------------------------------+
 | 5. Ceiling & Anomaly Clamping|  Per-class physical limits & low-conf median multiplier
 +------------------------------+
```

### Filtering Stages:
1. **Median Filter (`SPEED_OUTLIER_MEDIAN_WINDOW = 5`)**:
   Runs a sliding 1D median filter independently along the $X$ and $Y$ coordinates to remove isolated detection glitches without phase distortion.
2. **Moving Average (`SPEED_SMOOTHING_WINDOW = 9`)**:
   Smooths the median-filtered points to obtain a stable, continuous velocity vector.
3. **Plausibility & Anomaly Clamping**:
   - **Per-Class Ceilings**: Measurements are clamped to physically realistic class maxima (`VEHICLE_MAX_SPEED_KMH`):
     - Light Vehicle: 150 km/h
     - Minibus Taxi: 140 km/h
     - Bus / Medium Truck: 140 km/h
     - Long Truck: 130 km/h
     - Pedestrian: 20 km/h
     - Cyclist: 70 km/h
   - **Low-Confidence Clamp**: For short tracks ($< 10$ positions or $< 1.0\text{s}$ duration), speed is clamped to $2.0 \times$ the median segment speed to prevent exit boundary spikes.

---

## Live Post-Analysis Recalibration (`/api/v1/recalibrate-speed`)

A major strength of the architecture is **instant speed recalibration without re-processing the video**:

1. During initial analysis, the system archives raw metric trajectory points and timestamps for every vehicle in `FinalizedPathsStore` (SQLite).
2. If the user later determines the road was actually 32.5 meters instead of 30.0 meters, or adjusts skew factors:
   - The user inputs the updated measurements in the UI.
   - The backend rebuilds the `PerspectiveTransformer` with the new homography.
   - Trajectories are re-projected through the signal filtering pipeline in seconds.
   - All Excel reports are re-generated with updated speeds and speed distributions.
