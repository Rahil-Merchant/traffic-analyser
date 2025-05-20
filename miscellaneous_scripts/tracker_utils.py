# tracker_utils.py

import numpy as np
import torch
# scipy.linalg is not needed for GPUKalmanFilterXYAH if using torch.linalg

def xywh2ltwh(x): # Expects list, np.ndarray, or torch.Tensor (for x,y,w,h center)
    """Convert [x, y, w, h] center to [x1, y1, w, h] top-left."""
    if isinstance(x, list):
        x_arr = np.asarray(x, dtype=np.float32)
    elif isinstance(x, torch.Tensor):
        x_arr = x.cpu().numpy() if x.is_cuda else x.numpy()
        x_arr = np.copy(x_arr) # Ensure it's a copy
    elif isinstance(x, np.ndarray):
        x_arr = np.copy(x)
    else:
        raise TypeError(f"xywh2ltwh: Expected list, np.ndarray, or torch.Tensor, got {type(x)}")

    y = np.copy(x_arr)
    if x_arr.ndim == 1 and x_arr.shape[0] == 4:
        y[0] = x_arr[0] - x_arr[2] / 2.0  # top left x (x1)
        y[1] = x_arr[1] - x_arr[3] / 2.0  # top left y (y1)
    elif x_arr.ndim == 2 and x_arr.shape[-1] == 4:
        y[..., 0] = x_arr[..., 0] - x_arr[..., 2] / 2.0
        y[..., 1] = x_arr[..., 1] - x_arr[..., 3] / 2.0
    else:
        raise ValueError(f"Input to xywh2ltwh must be of shape (4,) or (N, 4), got {x_arr.shape}")
    return y


class GPUKalmanFilterXYAH:
    """
    Kalman Filter for (x, y, aspect_ratio, height) state, implemented with PyTorch for GPU.
    State: [cx, cy, a, h, vx, vy, va, vh]
    Measurement: [cx, cy, a, h]
    """
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        ndim, dt = 4, 1.0

        self._motion_mat = torch.eye(2 * ndim, device=self.device, dtype=torch.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        self._update_mat = torch.eye(ndim, 2 * ndim, device=self.device, dtype=torch.float32)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement_gpu: torch.Tensor):
        mean_pos = measurement_gpu 
        mean_vel = torch.zeros_like(mean_pos, device=self.device)
        mean = torch.cat([mean_pos, mean_vel])

        std = torch.zeros(8, device=self.device, dtype=torch.float32)
        std[0] = 2 * self._std_weight_position * measurement_gpu[3] 
        std[1] = 2 * self._std_weight_position * measurement_gpu[3] 
        std[2] = 1e-2  
        std[3] = 2 * self._std_weight_position * measurement_gpu[3] 
        std[4] = 10 * self._std_weight_velocity * measurement_gpu[3]
        std[5] = 10 * self._std_weight_velocity * measurement_gpu[3]
        std[6] = 1e-5  
        std[7] = 10 * self._std_weight_velocity * measurement_gpu[3]
        covariance = torch.diag(torch.square(std))
        return mean, covariance

    def predict(self, mean_gpu: torch.Tensor, covariance_gpu: torch.Tensor):
        std_pos = torch.zeros(4, device=self.device, dtype=torch.float32)
        std_pos[0] = self._std_weight_position * mean_gpu[3]
        std_pos[1] = self._std_weight_position * mean_gpu[3]
        std_pos[2] = 1e-2
        std_pos[3] = self._std_weight_position * mean_gpu[3]

        std_vel = torch.zeros(4, device=self.device, dtype=torch.float32)
        std_vel[0] = self._std_weight_velocity * mean_gpu[3]
        std_vel[1] = self._std_weight_velocity * mean_gpu[3]
        std_vel[2] = 1e-5
        std_vel[3] = self._std_weight_velocity * mean_gpu[3]
        
        motion_cov = torch.diag(torch.square(torch.cat([std_pos, std_vel])))

        mean_gpu = self._motion_mat @ mean_gpu
        covariance_gpu = self._motion_mat @ covariance_gpu @ self._motion_mat.T + motion_cov
        return mean_gpu, covariance_gpu

    def project(self, mean_gpu: torch.Tensor, covariance_gpu: torch.Tensor):
        std = torch.zeros(4, device=self.device, dtype=torch.float32)
        std[0] = self._std_weight_position * mean_gpu[3]
        std[1] = self._std_weight_position * mean_gpu[3]
        std[2] = 1e-1 
        std[3] = self._std_weight_position * mean_gpu[3]
        innovation_cov = torch.diag(torch.square(std))

        mean_gpu_proj = self._update_mat @ mean_gpu
        covariance_gpu_proj = self._update_mat @ covariance_gpu @ self._update_mat.T
        return mean_gpu_proj, covariance_gpu_proj + innovation_cov

    def update(self, mean_gpu: torch.Tensor, covariance_gpu: torch.Tensor, measurement_gpu: torch.Tensor):
        projected_mean_gpu, projected_cov_gpu = self.project(mean_gpu, covariance_gpu)
        PHt = covariance_gpu @ self._update_mat.T 
        
        try:
            L = torch.linalg.cholesky(projected_cov_gpu) 
            S_inv = torch.cholesky_inverse(L, upper=False) 
            kalman_gain = PHt @ S_inv
        except RuntimeError as e: 
            # print(f"Warning: GPU Kalman Filter Cholesky or inverse failed: {e}. Using pseudo-inverse.") # Less verbose
            try:
                S_pinv = torch.linalg.pinv(projected_cov_gpu)
                kalman_gain = PHt @ S_pinv
            except RuntimeError as pinv_e:
                print(f"CRITICAL WARNING: KF pseudo-inverse also failed: {pinv_e}. Update skipped.")
                return mean_gpu, covariance_gpu 
            if torch.isnan(kalman_gain).any() or torch.isinf(kalman_gain).any():
                print(f"CRITICAL WARNING: Kalman gain NaN/Inf after pinv. Update skipped.")
                return mean_gpu, covariance_gpu
        
        innovation = measurement_gpu - projected_mean_gpu
        new_mean_gpu = mean_gpu + kalman_gain @ innovation 
        new_covariance_gpu = covariance_gpu - kalman_gain @ projected_cov_gpu @ kalman_gain.T
        
        if torch.isnan(new_mean_gpu).any() or torch.isinf(new_mean_gpu).any() or \
           torch.isnan(new_covariance_gpu).any() or torch.isinf(new_covariance_gpu).any():
            print(f"CRITICAL WARNING: KF update resulted in NaN/Inf state. Reverting to pre-update state for track.")
            return mean_gpu, covariance_gpu 
        return new_mean_gpu, new_covariance_gpu

    def multi_predict(self, means_gpu: torch.Tensor, covariances_gpu: torch.Tensor):
        N = means_gpu.shape[0]
        if N == 0: return means_gpu, covariances_gpu

        std_pos_h = self._std_weight_position * means_gpu[:, 3]
        std_vel_h = self._std_weight_velocity * means_gpu[:, 3]

        motion_cov_diag_elements = torch.stack([
            std_pos_h, std_pos_h, 
            torch.full_like(std_pos_h, 1e-2, device=self.device), std_pos_h,
            std_vel_h, std_vel_h,
            torch.full_like(std_vel_h, 1e-5, device=self.device), std_vel_h
        ], dim=1)
        motion_cov = torch.diag_embed(torch.square(motion_cov_diag_elements))
        
        means_gpu_pred = means_gpu @ self._motion_mat.T
        cov_Mt = covariances_gpu @ self._motion_mat.T 
        covariances_gpu_pred = self._motion_mat @ cov_Mt + motion_cov
        return means_gpu_pred, covariances_gpu_pred