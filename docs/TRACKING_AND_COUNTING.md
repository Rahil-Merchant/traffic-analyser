# Multi-Object Tracking & Vehicle Counting Engine

This document provides a deep technical analysis of the multi-object tracking (MOT) and vehicle-level counting architecture implemented in the Urban Reconnaissance platform.

---

## The Core Problem: Tracker IDs $\neq$ Vehicles

In automated traffic surveillance, standard trackers (such as SORT, DeepSORT, or ByteTrack) assign a new numeric ID whenever an association fails due to:
1. **Temporary Occlusion**: A vehicle passing behind a pole, bridge, or larger vehicle.
2. **Detection Misses**: A missed detection for 2 to 3 frames at high speed or low contrast.
3. **High Velocity**: Fast-moving vehicles whose bounding boxes between consecutive frames do not overlap (zero IoU).

If a naive system simply counts "unique tracker IDs that crossed a line", **traffic counts become drastically inflated by 30% to 150%**.

The platform addresses this through a two-tier architecture:
1. **Tier 1: Low-Level Tracker (`OCSort`)**: Associates frame-to-frame bounding boxes into short track fragments.
2. **Tier 2: Vehicle Counter & Stitching (`VehicleCounter`)**: Fuses track fragments into unified physical vehicles, eliminates duplicates, and detects impossible movements.

---

## Tier 1: Custom OC-SORT Tracker (`tracking/ocsort.py`)

The system integrates an enhanced implementation of **OC-SORT** (*Observation-Centric SORT*, Cao et al., CVPR 2023) with custom extensions tailored for road traffic:

### State Vector Representation
Each track maintains a Kalman filter with a 7-dimensional state vector:
$$\mathbf{x} = [c_x, c_y, s, r, \dot{c}_x, \dot{c}_y, \dot{s}]^T$$
Where:
- $c_x, c_y$: Bounding box center coordinates
- $s$: Box area (scale)
- $r$: Box aspect ratio (width / height)
- $\dot{c}_x, \dot{c}_y$: Center velocity
- $\dot{s}$: Area change rate

### Enhanced 4-Stage Association Cascade

1. **High-Confidence Association (IoU + Direction Consistency)**:
   - Associates detections with confidence $\ge \text{det\_thresh}$ ($0.25$).
   - Cost matrix balances spatial overlap with motion direction:
     $$\text{Cost} = \text{IoU}(D, T) + \text{inertia} \times \cos(\theta_{obs}, \theta_{track})$$
   - Prevents matching vehicles travelling in opposite lanes even if their predicted bounding boxes intersect.

2. **Buffered-IoU (C-BIoU) Expansion**:
   - For fast vehicles where boxes jump between frames (zero raw IoU), bounding boxes are scaled outward by `buffer_scales` ($0.3$ and $0.6$).
   - Enables re-acquisition of fast vehicles without expanding search regions across the entire frame.

3. **ByteTrack Second-Chance Pass**:
   - Matches remaining unmatched tracks against low-confidence detections ($0.10 \le \text{conf} < 0.25$) using strict IoU.
   - Preserves tracks during motion blur or partial occlusion.

4. **Observation-Centric Online Smoothing (OOS)**:
   - When an unobserved track is re-associated after several frames of missed detections, the Kalman state is back-propagated and corrected along the virtual trajectory between the last actual observation and the new one.

---

## Tier 2: Vehicle Counter & Stitching (`tracking/vehicle_counter.py`)

The `VehicleCounter` sits directly above the tracker, maintaining an independent ledger of physical vehicles.

### 1. Track Fragment Stitching
When a new tracker ID begins, rather than immediately registering a new vehicle, the engine checks if it is a continuation of a previously lost vehicle:

```
               Frame t-5             Frame t-2            Frame t (Now)
Vehicle A ---[Track ID #12]---------(Track Lost)
                                          |
                                   Predicted Position
                                          |
                                          v
                                    [Track ID #29] --> Stitched to Vehicle A!
```

- **Metric Road Prediction**: If road calibration is enabled, predicted motion runs in real-world meters on the road plane. The spatial gate tolerates longitudinal gaps while being strictly constrained laterally across lanes.
- **Image-Space Prediction**: If uncalibrated, motion extrapolation runs in normalized box heights, ensuring gates adapt naturally to both foreground and distant background perspective.

### 2. Duplicate Suppression & Footprint Matching
When two bounding boxes overlap in the same frame, the engine evaluates their **ground footprint** (bottom horizontal edge):
$$\text{same\_footprint}(a, b) \iff |\text{mid}_x(a) - \text{mid}_x(b)| \le 0.5 \min(w_a, w_b) \land |y_{\text{bottom}}(a) - y_{\text{bottom}}(b)| \le 0.25 \min(h_a, h_b)$$
Two live tracks are merged only when they share the same ground contact line and move in lockstep.

### 3. Reversal Detection & Split
Vehicles do not execute instantaneous 180° U-turns in free-flowing traffic. When a stitched vehicle's velocity reverses direction along the road axis:
- The system recognizes that an oncoming vehicle was mistakenly stitched onto a receding vehicle (a common failure mode near camera vanishing points).
- The vehicle is automatically bifurcated at the inflection point, and both resulting trajectories are evaluated independently.
- Any zone transition event generated erroneously by the joined path is retracted.

### 4. Confidence- and Size-Weighted Class Voting
Instead of trusting the most recent detection or taking a simple majority vote:
$$\text{Class Weight} = \text{Detection Confidence} \times \text{Box Area}$$
Detections captured when the vehicle is close to the camera (large bounding box, high sharpness) heavily outweigh noisy or ambiguous classifications made when the vehicle was a distant speck on the horizon.

### 5. Zone Crossing & Trajectory Segment Interpolation
- Zone polygons are tested using a vectorised even-odd point-in-polygon algorithm (`points_in_polygon`).
- To prevent fast vehicles from skipping over narrow zone boundaries between consecutive frames, the path connecting $(x_{t-1}, y_{t-1})$ and $(x_t, y_t)$ is evaluated as a continuous line segment against zone boundaries.
