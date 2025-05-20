# project/tracking/kalman_filter_wrapper.py
import numpy as np
from filterpy.kalman import KalmanFilter
from config import DEBUG_MODE_TRACKING

class KalmanFilterXYAH:
    """
    A Kalman filter wrapper for tracking objects with state (x, y, a, h, vx, vy, va, vh).
    x, y: center coordinates
    a: aspect ratio (width/height)
    h: height
    vx, vy, va, vh: velocities of the above
    """
    def __init__(self, dt=1.0, std_weight_position=1./20, std_weight_velocity=1./160):
        self._ndim_meas = 4  # x, y, a, h (measurement)
        self.dim_x = 8    # x, y, a, h, vx, vy, va, vh (state)
        self.dt = dt

        self.kf = KalmanFilter(dim_x=self.dim_x, dim_z=self._ndim_meas)

        # State Transition Matrix (F) - Constant Velocity Model
        self.kf.F = np.eye(self.dim_x)
        for i in range(self._ndim_meas):
            self.kf.F[i, self._ndim_meas + i] = self.dt
        
        # Measurement Function (H)
        self.kf.H = np.eye(self._ndim_meas, self.dim_x)

        # Process Noise Covariance (Q) - uncertainty in motion model
        # Tunable: reflects how much we expect the object's velocity to change
        q_pos = std_weight_position
        q_vel = std_weight_velocity
        q_diag = np.array([q_pos**2, q_pos**2, q_pos**2, q_pos**2,
                           q_vel**2, q_vel**2, q_vel**2, q_vel**2], dtype=np.float32)
        self.kf.Q = np.diag(q_diag)

        # Measurement Noise Covariance (R) - uncertainty in detection
        # Tunable: reflects detector noise
        r_std_xy = std_weight_position * 5  # Example: detection x,y noise
        r_std_ah = std_weight_position * 10 # Example: detection a,h noise (often less stable)
        r_diag = np.array([r_std_xy**2, r_std_xy**2, r_std_ah**2, r_std_ah**2], dtype=np.float32)
        self.kf.R = np.diag(r_diag)

        # Initial State Covariance (P)
        self.kf.P = np.eye(self.dim_x) * 1.0 # Initial uncertainty
        self.kf.P[self._ndim_meas:, self._ndim_meas:] *= 1000.0 # High uncertainty for initial velocities
        self.kf.P *= 10. # Overall high initial uncertainty

        self.kf.x = np.zeros((self.dim_x, 1)) # Initial state estimate (will be set on first update)

    def predict(self):
        """Predicts the state and covariance."""
        if self.kf.x[0,0] == 0 and self.kf.x[1,0] == 0: # If not initialized, don't predict nonsense
            return self.kf.x, self.kf.P
        self.kf.predict()
        # Ensure aspect ratio and height remain positive after prediction (optional, but good practice)
        if self.kf.x[2] < 0.1: self.kf.x[2] = 0.1 # Min aspect ratio
        if self.kf.x[3] < 1: self.kf.x[3] = 1 # Min height
        return self.kf.x, self.kf.P

    def update(self, measurement_xyah):
        """Updates the state with the given measurement."""
        # Ensure measurement is a column vector
        z = np.asarray(measurement_xyah).reshape(self._ndim_meas, 1)
        self.kf.update(z)
        # Ensure aspect ratio and height remain positive after update
        if self.kf.x[2] < 0.1: self.kf.x[2] = 0.1
        if self.kf.x[3] < 1: self.kf.x[3] = 1
        
    def get_current_state_xyah(self):
        """Returns current estimated [x, y, a, h] from the state vector."""
        return self.kf.x[:self._ndim_meas, 0].flatten()

    def get_predicted_state_xyah(self):
        """Returns the predicted [x, y, a, h] from the state vector (after predict() call)."""
        # This is useful for matching if you want to match against prediction
        return self.kf.x_prior[:self._ndim_meas, 0].flatten()

    def initiate(self, measurement_xyah):
        z = np.asarray(measurement_xyah).reshape(self._ndim_meas, 1)
        self.kf.x[:self._ndim_meas] = z
        self.kf.x[self._ndim_meas:] = 0 
        
        # Reset P to initial uncertainty for a new track
        self.kf.P = np.eye(self.dim_x)
        self.kf.P[:self._ndim_meas, :self._ndim_meas] *= 10.0 # Slightly higher for pos
        self.kf.P[self._ndim_meas:, self._ndim_meas:] *= 1000.0 # Much higher for vel
        # self.kf.P *= 10. # This might make initial P too large, try without or smaller factor

        if DEBUG_MODE_TRACKING: # No STrack ID here, this is the KF object itself
            print(f"    KF.initiate: Meas XYAH: {np.array2string(measurement_xyah, formatter={'float_kind':lambda x: '%.2f' % x})}, Initial KF.x: {np.array2string(self.kf.x.flatten(), formatter={'float_kind':lambda x: '%.2f' % x})}")