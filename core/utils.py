# project/core/utils.py
import os
import torch
import gc
from datetime import datetime
from config import VALID_MOVEMENTS
import psutil
from loguru import logger

def format_timestamp(dt: datetime):
    """Convert datetime to date (YYMMDD) and time (HHMMSSmmm) strings."""
    date_str = dt.strftime('%y%m%d')
    time_str = dt.strftime('%H%M%S')
    ms_str = f"{dt.microsecond // 1000:03d}"
    return date_str, f"{time_str}{ms_str}"

def setup_torch_global_settings():
    """Configure global PyTorch settings for performance."""
    try:
        if torch.cuda.is_available():
            print("Configuring PyTorch for GPU performance...")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            torch.cuda.empty_cache()

            if torch.cuda.get_device_capability(0)[0] >= 8: # Ampere+
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch, 'set_float32_matmul_precision'):
                     torch.set_float32_matmul_precision('high')

            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            print("PyTorch GPU settings applied.")
        else:
            print("Configuring PyTorch for CPU.")
    except Exception as e:
        print(f"Warning: Could not apply all PyTorch optimizations: {e}")

def get_display_direction(direction):
    """Format direction for display (e.g., 'north' -> 'North')."""
    return direction.capitalize() if direction else "N/A"

def normalize_direction(direction):
    """Convert direction to lowercase for consistent comparison."""
    return direction.lower() if direction else ""

def is_valid_movement(from_dir, to_dir, active_zones):
    """
    Check if movement from from_dir to to_dir is valid based on config
    AND ensures both directions are present in the active_zones list/set.
    Args:
        from_dir (str): The entry direction.
        to_dir (str): The exit direction.
        active_zones (list or set): Collection of lowercase active zone names (e.g., ['north', 'west', 'south']).
    """
    from_dir_norm = normalize_direction(from_dir)
    to_dir_norm = normalize_direction(to_dir)

    # Basic checks
    if not from_dir_norm or not to_dir_norm: return False
    if from_dir_norm == to_dir_norm: return False # U-turn

    # --- Check if both directions are actually active zones ---
    if not active_zones or from_dir_norm not in active_zones or to_dir_norm not in active_zones:
        logger.debug(f"Invalid movement: One or both directions ('{from_dir_norm}' -> '{to_dir_norm}') not in active zones: {active_zones}")
        return False

    # Check against the base VALID_MOVEMENTS rules from config
    if from_dir_norm not in VALID_MOVEMENTS:
        logger.debug(f"Invalid movement: Entry direction '{from_dir_norm}' not found in VALID_MOVEMENTS config.")
        return False

    # Check if the target direction is a valid exit from the entry direction per config rules
    valid_config_targets = VALID_MOVEMENTS.get(from_dir_norm, [])
    is_valid_target = to_dir_norm in valid_config_targets

    if not is_valid_target:
        logger.debug(f"Invalid movement: '{to_dir_norm}' not a valid exit from '{from_dir_norm}' per VALID_MOVEMENTS. Valid targets: {valid_config_targets}")

    return is_valid_target

def setup_thread_affinity(perf_cores=8):
    """Attempt to set thread affinity for hybrid CPUs (best effort)."""
    try:
        p = psutil.Process(os.getpid())
        if hasattr(p, 'cpu_affinity'):
            total_logical_cores = psutil.cpu_count(logical=True)
            p_core_indices = [c for c in range(perf_cores * 2) if c < total_logical_cores]
            if p_core_indices:
                p.cpu_affinity(p_core_indices)
                logger.debug(f"Set main process affinity to P-cores: {p_core_indices}")
    except Exception as e:
        logger.info(f"Failed to set thread affinity: {e}")

def cleanup_memory():
    """Explicitly run garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("Performed memory cleanup (GC + CUDA Cache)")

