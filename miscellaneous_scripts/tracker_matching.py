# tracker_matching.py

import numpy as np
import scipy
from scipy.spatial.distance import cdist 
import torch
import torchvision.ops 

try:
    import lap
    assert lap.__version__ 
except (ImportError, AssertionError, AttributeError):
    print("Warning: 'lap' library not found or import failed. Linear assignment will use scipy if available.")
    lap = None 


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True) -> tuple:
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    use_lap_actually = use_lap and lap is not None
    
    if use_lap_actually:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh) # type: ignore
        matches = np.asarray([[ix, mx] for ix, mx in enumerate(x) if mx >= 0], dtype=int)
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else: 
        if use_lap and lap is None: print("lapjv requested but not available, falling back to scipy.")
        try:
            if scipy.__version__ < "1.4.0" and (cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0):
                 return np.empty((0,2), dtype=int), np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            valid_matches_mask = cost_matrix[row_ind, col_ind] <= thresh
            matches = np.stack((row_ind[valid_matches_mask], col_ind[valid_matches_mask]), axis=1).astype(int)
        except AttributeError: # If scipy or linear_sum_assignment is not available
             print("Scipy linear_sum_assignment not available or failed. Returning no matches.")
             matches = np.empty((0,2), dtype=int)


        if matches.shape[0] == 0:
            unmatched_a = np.arange(cost_matrix.shape[0])
            unmatched_b = np.arange(cost_matrix.shape[1])
        else:
            unmatched_a = np.setdiff1d(np.arange(cost_matrix.shape[0]), matches[:, 0], assume_unique=True)
            unmatched_b = np.setdiff1d(np.arange(cost_matrix.shape[1]), matches[:, 1], assume_unique=True)
            
    return matches, list(unmatched_a), list(unmatched_b)


def iou_distance(atracks: list, btracks: list) -> np.ndarray: # atracks, btracks are lists of GPUSTrack
    if not atracks or not btracks:
        return np.empty((len(atracks), len(btracks)), dtype=np.float32)

    try:
        # GPUSTrack.xyxy_gpu returns a [x1,y1,x2,y2] GPU tensor
        atlbrs_gpu_list = [track.xyxy_gpu for track in atracks if track.xyxy_gpu is not None and track.xyxy_gpu.numel() == 4 and track.is_activated and track.mean_gpu is not None]
        btlbrs_gpu_list = [track.xyxy_gpu for track in btracks if track.xyxy_gpu is not None and track.xyxy_gpu.numel() == 4 and track.is_activated and track.mean_gpu is not None] # Only consider activated tracks for matching

        if not atlbrs_gpu_list or not btlbrs_gpu_list:
             return np.ones((len(atracks), len(btracks)), dtype=np.float32) 

        atlbrs_gpu = torch.stack(atlbrs_gpu_list) 
        btlbrs_gpu = torch.stack(btlbrs_gpu_list)
        
        ious_gpu = torchvision.ops.box_iou(atlbrs_gpu, btlbrs_gpu) 
        cost_matrix_gpu = 1.0 - ious_gpu
        return cost_matrix_gpu.cpu().numpy()
        
    except Exception as e:
        print(f"Error during GPU IoU calculation in iou_distance: {e}. Returning max cost matrix.")
        return np.ones((len(atracks), len(btracks)), dtype=np.float32)


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    # This is a placeholder, actual ReID feature handling would be complex
    # print("Warning: embedding_distance is placeholder and not using GPU features.")
    cost_matrix = np.ones((len(tracks), len(detections)), dtype=np.float32)
    if not tracks or not detections: # Handle empty inputs to avoid error with asarray
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    try: 
        # Assuming tracks/detections are STrack-like and might have .curr_feat or .smooth_feat (NumPy)
        det_features_list = [track.curr_feat for track in detections if hasattr(track, 'curr_feat') and track.curr_feat is not None]
        track_features_list = [track.smooth_feat for track in tracks if hasattr(track, 'smooth_feat') and track.smooth_feat is not None]

        if det_features_list and track_features_list:
            det_features = np.asarray(det_features_list, dtype=np.float32)
            track_features = np.asarray(track_features_list, dtype=np.float32)
            if det_features.ndim > 1 and track_features.ndim > 1 and det_features.shape[1] == track_features.shape[1]: # Basic check
                cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    except AttributeError: pass # .curr_feat or .smooth_feat not present
    except Exception as e: print(f"Error in embedding_distance: {e}")
    return cost_matrix

def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray: 
    if cost_matrix.size == 0 or not detections: return cost_matrix
    
    iou_sim = 1.0 - cost_matrix 
    det_scores = np.array([det.score for det in detections], dtype=np.float32)
    
    if cost_matrix.shape[1] != len(det_scores): 
        # print(f"Warning: fuse_score shape mismatch. Cost matrix cols {cost_matrix.shape[1]} vs det_scores len {len(det_scores)}. Skipping fuse.")
        return cost_matrix # Return original cost matrix if shapes mismatch

    det_scores_expanded = np.expand_dims(det_scores, axis=0)
    fuse_sim = iou_sim * det_scores_expanded 
    return 1.0 - fuse_sim