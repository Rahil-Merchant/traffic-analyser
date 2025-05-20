# project/tracking/vehicle_tracker.py
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from config import (
    VEHICLE_TIMEOUTS,
    TRACK_HISTORY_LENGTH,
)
from core.utils import debug_print

class VehicleTracker:
    """
    Manages application-specific state for vehicles tracked by ByteTrack.
    Handles entry/exit events, time in intersection, and completed path logging.
    """
    def __init__(self):
        self.active_vehicles = {}  # {str(bytetrack_id): {app_data}}
        self.tracking_history = {} # {str(bytetrack_id): deque([(x,y), ...])} # Scaled display coordinates
        self.completed_paths = []  # Stores data of exited/timed_out vehicles
        self.vehicle_last_seen_app_time = {} # {str(bytetrack_id): datetime object of last update}

    def update_vehicle_history(self, bytetrack_id_str, center_point_scaled, current_frame_app_time):
        """
        Updates basic history and last seen time for a vehicle ID from ByteTrack.
        This is called for every active track output by ByteTrack.
        """
        if bytetrack_id_str not in self.tracking_history:
            self.tracking_history[bytetrack_id_str] = deque(maxlen=TRACK_HISTORY_LENGTH)
        self.tracking_history[bytetrack_id_str].append(center_point_scaled)
        self.vehicle_last_seen_app_time[bytetrack_id_str] = current_frame_app_time

    def register_entry(self, bytetrack_id_str, entry_direction, entry_time_dt, 
                       entry_point_scaled, vehicle_type_consistent):
        # Check if already active with the same entry to avoid re-triggering unless it exited and re-entered
        current_app_state = self.active_vehicles.get(bytetrack_id_str)
        if current_app_state and current_app_state['status'] == 'active' and \
           current_app_state['entry_direction'] == entry_direction:
            # Already active from this direction, likely a bounce or slight re-entry. Ignore.
            return False

        self.active_vehicles[bytetrack_id_str] = {
            'id': bytetrack_id_str,
            'type': vehicle_type_consistent,
            'entry_direction': entry_direction,
            'entry_time': entry_time_dt,
            'entry_point': entry_point_scaled, # Scaled coords
            'exit_direction': None, 'exit_time': None, 'exit_point': None,
            'status': 'active'
        }
        debug_print(f"APP_ENTRY: ID:{bytetrack_id_str} ({vehicle_type_consistent}) from {entry_direction}")
        return True

    def register_exit(self, bytetrack_id_str, exit_direction, exit_time_dt, exit_point_scaled):
        if bytetrack_id_str in self.active_vehicles and self.active_vehicles[bytetrack_id_str]['status'] == 'active':
            data = self.active_vehicles[bytetrack_id_str]
            # Vehicle type should be the consistent one recorded at entry or updated if STrack changes it.
            # For simplicity, assume data['type'] holds the relevant consistent type.

            data.update({
                'exit_direction': exit_direction, 
                'exit_time': exit_time_dt, 
                'exit_point': exit_point_scaled, # Scaled coords
                'status': 'exited'
            })
            time_in_int = (exit_time_dt - data['entry_time']).total_seconds()
            data['time_in_intersection'] = round(time_in_int, 2)
            self.completed_paths.append(data.copy())
            debug_print(f"APP_EXIT: ID:{bytetrack_id_str} ({data['type']}) to {exit_direction}. Time: {time_in_int:.2f}s")
            self._clear_active_vehicle_app_state(bytetrack_id_str) # Clear from active after exit
            return True, time_in_int
        return False, None

    def check_timeouts(self, current_time_dt: datetime):
        timed_out_ids_app = []
        for v_id_str, data in list(self.active_vehicles.items()): # Iterate on copy
            if data['status'] == 'active':
                time_in_int = (current_time_dt - data['entry_time']).total_seconds()
                v_type_current = data.get('type', 'default')
                timeout_duration = VEHICLE_TIMEOUTS.get(v_type_current, VEHICLE_TIMEOUTS['default'])
                
                if time_in_int > timeout_duration:
                    debug_print(f"APP_TIMEOUT: ID:{v_id_str} ({data['type']}) after {time_in_int:.1f}s (Limit: {timeout_duration}s).")
                    data.update({
                        'status': 'timed_out', 
                        'exit_time': current_time_dt,
                        'exit_direction': 'TIMEOUT', 
                        'time_in_intersection': round(time_in_int, 2)
                    })
                    self.completed_paths.append(data.copy())
                    timed_out_ids_app.append(v_id_str)
                    self._clear_active_vehicle_app_state(v_id_str)
        return timed_out_ids_app

    def force_exit_vehicle(self, bytetrack_id_str, exit_time_dt: datetime):
        if bytetrack_id_str in self.active_vehicles and self.active_vehicles[bytetrack_id_str]['status'] == 'active':
            data = self.active_vehicles[bytetrack_id_str]
            time_in_int = (exit_time_dt - data['entry_time']).total_seconds()
            data.update({
                'status': 'forced_exit', 
                'exit_time': exit_time_dt,
                'exit_direction': 'FORCED', 
                'time_in_intersection': round(time_in_int, 2)
            })
            self.completed_paths.append(data.copy())
            self._clear_active_vehicle_app_state(bytetrack_id_str)
            debug_print(f"APP_FORCED_EXIT: ID:{bytetrack_id_str} ({data['type']})")
            return True
        return False

    def _clear_active_vehicle_app_state(self, bytetrack_id_str):
        """Removes an ID from the active_vehicles list."""
        self.active_vehicles.pop(bytetrack_id_str, None)
        # History and last_seen_app_time might be kept for a bit longer for lingering visualization
        # or can be removed here as well. Let's keep them for now, cleanup can handle stale history.

    def remove_all_data_for_id(self, bytetrack_id_str):
        """Removes all tracking data associated with a vehicle ID from this app-level tracker."""
        self._clear_active_vehicle_app_state(bytetrack_id_str)
        self.tracking_history.pop(bytetrack_id_str, None)
        self.vehicle_last_seen_app_time.pop(bytetrack_id_str, None)
        # debug_print(f"APP_Cleanup: Removed all app data for ID:{bytetrack_id_str}")

    def get_active_vehicle_count(self): return len(self.active_vehicles)
    def get_completed_paths(self): return self.completed_paths
    def get_tracking_trail(self, bytetrack_id_str): return list(self.tracking_history.get(bytetrack_id_str, []))