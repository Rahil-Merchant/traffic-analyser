# project/tracking/path_manager.py
from collections import deque, defaultdict
from config import (
    VEHICLE_TIMEOUTS, TRACK_HISTORY_LENGTH,
    MAX_CONSECUTIVE_MISSES_FOR_PATH_MANAGER
)
from core.utils import debug_print, is_valid_movement

class VehiclePathManager:
    def __init__(self):
        self.active_paths = {}  # {bytetrack_id: {data about the path}}
        self.completed_paths_log = []
        self.tracking_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LENGTH)) # Visual trail
        self.vehicle_type_tracker = defaultdict(lambda: defaultdict(int)) # {bt_id: {type_name: count}}
        self.last_seen_update_time_dt = {} # {bytetrack_id: datetime_object_of_last_update}
        self.consecutive_misses = defaultdict(int) # For paths known to PathManager

    def get_consistent_type(self, bytetrack_id, current_type_name):
        self.vehicle_type_tracker[bytetrack_id][current_type_name] += 1
        # Return the type with the highest count for this ID
        return max(self.vehicle_type_tracker[bytetrack_id], key=self.vehicle_type_tracker[bytetrack_id].get)

    def update_track_sighting(self, bytetrack_id, current_center_point_display, vehicle_type_name_raw,
                               current_timestamp_dt, current_frame_time_video_sec): # current_frame_time_video_sec not used here directly but good for consistency
        prev_pos = self.tracking_history[bytetrack_id][-1] if self.tracking_history[bytetrack_id] else None
        self.tracking_history[bytetrack_id].append(current_center_point_display)

        consistent_type = self.get_consistent_type(bytetrack_id, vehicle_type_name_raw)
        if bytetrack_id in self.active_paths: # If it's an active path, update its type
            self.active_paths[bytetrack_id]['type'] = consistent_type
        
        self.last_seen_update_time_dt[bytetrack_id] = current_timestamp_dt
        self.consecutive_misses[bytetrack_id] = 0 # Reset misses as it's seen
        return prev_pos

    def register_entry_event(self, bytetrack_id, entry_direction, entry_time_dt, entry_point_coords_display):
        if bytetrack_id in self.active_paths and self.active_paths[bytetrack_id]['status'] == 'active':
            debug_print(f"PathMgr: V_BT:{bytetrack_id} already active. New entry from {entry_direction} ignored or handled as re-entry if needed.")
            return False, None

        # Determine vehicle type (use the one most consistently seen so far for this ID)
        path_vehicle_type = "Unknown"
        if self.vehicle_type_tracker[bytetrack_id]:
             path_vehicle_type = max(self.vehicle_type_tracker[bytetrack_id], key=self.vehicle_type_tracker[bytetrack_id].get)
        
        self.active_paths[bytetrack_id] = {
            'id': bytetrack_id, 'type': path_vehicle_type,
            'entry_direction': entry_direction, 'entry_time': entry_time_dt,
            'entry_point': entry_point_coords_display,
            'exit_direction': None, 'exit_time': None, 'exit_point': None,
            'status': 'active'
        }
        self.last_seen_update_time_dt[bytetrack_id] = entry_time_dt # Critical for timeout
        debug_print(f"PathMgr: V_BT:{bytetrack_id} ({path_vehicle_type}) ENTRY from {entry_direction}")
        log_data = {
            'timestamp_dt': entry_time_dt, 'vehicle_id': bytetrack_id,
            'vehicle_type': path_vehicle_type, 'event_type': 'ENTRY',
            'direction_from': entry_direction, 'direction_to': None, 'status': 'entry'
        }
        return True, log_data

    def register_exit_event(self, bytetrack_id, exit_direction, exit_time_dt, exit_point_coords_display, active_zone_names_list):
        if bytetrack_id not in self.active_paths or self.active_paths[bytetrack_id]['status'] != 'active':
            debug_print(f"PathMgr: V_BT:{bytetrack_id} EXIT ignored (not active or already completed).")
            return False, None

        path_data = self.active_paths[bytetrack_id]
        if not is_valid_movement(path_data['entry_direction'], exit_direction, active_zone_names_list):
            debug_print(f"PathMgr: V_BT:{bytetrack_id} INVALID EXIT from {path_data['entry_direction']} to {exit_direction}. Ignored.")
            return False, None

        final_type = path_data['type'] # Type should be the consistent one
        path_data.update({
            'exit_direction': exit_direction, 'exit_time': exit_time_dt,
            'exit_point': exit_point_coords_display, 'status': 'exited'
        })
        time_in_int = (exit_time_dt - path_data['entry_time']).total_seconds()
        path_data['time_in_intersection'] = round(time_in_int, 2)

        self.completed_paths_log.append(path_data.copy())
        # self.active_paths.pop(bytetrack_id) # Don't remove yet, let check_timeouts handle full cleanup
        self.last_seen_update_time_dt[bytetrack_id] = exit_time_dt
        debug_print(f"PathMgr: V_BT:{bytetrack_id} ({final_type}) EXIT to {exit_direction}. Time: {time_in_int:.2f}s.")
        log_data = {
            'timestamp_dt': exit_time_dt, 'vehicle_id': bytetrack_id,
            'vehicle_type': final_type, 'event_type': 'EXIT',
            'direction_from': path_data['entry_direction'], 'direction_to': exit_direction,
            'status': 'exit', 'time_in_intersection': path_data['time_in_intersection']
        }
        return True, log_data

    def check_timeouts(self, current_time_dt, current_frame_active_byte_ids_from_wrapper):
        timed_out_event_logs = []
        ids_to_fully_remove_from_pm = []

        # Increment misses for paths in PathManager that were NOT reported by ByteTrackWrapper this frame
        for path_id_pm in list(self.active_paths.keys()): # Iterate known paths in PM
            if path_id_pm not in current_frame_active_byte_ids_from_wrapper:
                # This path_id_pm was either 'active' or 'exited' in PathManager,
                # but ByteTrackWrapper did not report it as an active track *this specific frame*.
                self.consecutive_misses[path_id_pm] += 1
            # If it WAS in current_frame_active_byte_ids_from_wrapper, update_track_sighting already reset misses to 0.

        for path_id, data in list(self.active_paths.items()):
            # If path already marked 'exited', check if it's been unseen long enough to fully clean up
            if data['status'] == 'exited':
                if self.consecutive_misses[path_id] > MAX_CONSECUTIVE_MISSES_FOR_PATH_MANAGER: # e.g., 5 frames after exit
                    ids_to_fully_remove_from_pm.append(path_id)
                continue # Don't process for timeout if already exited

            # If path is 'active', check for timeout by duration OR by consecutive misses
            if data['status'] == 'active':
                time_in_int_so_far = (current_time_dt - data['entry_time']).total_seconds()
                v_type_current = data.get('type', 'default')
                timeout_duration_config = VEHICLE_TIMEOUTS.get(v_type_current, VEHICLE_TIMEOUTS['default'])

                debug_print(f"PathMgr CheckTimeout: V_BT:{path_id}, Misses: {self.consecutive_misses[path_id]}, "
                    f"TimeInInt: {time_in_int_so_far:.1f}s, Type: {v_type_current}, "
                    f"TimeoutDurCfg: {timeout_duration_config}s, "
                    f"ActiveInWrapperThisFrame: {path_id in current_frame_active_byte_ids_from_wrapper}")

                has_timed_out_by_duration = time_in_int_so_far > timeout_duration_config
                has_timed_out_by_misses = self.consecutive_misses[path_id] > MAX_CONSECUTIVE_MISSES_FOR_PATH_MANAGER

                if has_timed_out_by_duration or has_timed_out_by_misses:
                    reason = "duration" if has_timed_out_by_duration else "misses"
                    # Use last_seen_update_time_dt for exit_time if available, otherwise current_time_dt
                    actual_exit_time_for_timeout = self.last_seen_update_time_dt.get(path_id, current_time_dt)
                    time_in_intersection_final = (actual_exit_time_for_timeout - data['entry_time']).total_seconds()

                    debug_print(f"PathMgr: V_BT:{path_id} ({data['type']}) TIMEOUT ({reason}) after {time_in_intersection_final:.1f}s. Misses: {self.consecutive_misses[path_id]}.")
                    data.update({
                        'status': 'timed_out',
                        'exit_time': actual_exit_time_for_timeout,
                        'exit_direction': 'TIMEOUT',
                        'time_in_intersection': round(time_in_intersection_final, 2)
                    })
                    self.completed_paths_log.append(data.copy())
                    ids_to_fully_remove_from_pm.append(path_id) # Mark for full cleanup
                    
                    log_event = {
                        'timestamp_dt': data['exit_time'], 'vehicle_id': path_id,
                        'vehicle_type': data['type'], 'event_type': 'TIMEOUT',
                        'direction_from': data['entry_direction'], 'direction_to': 'TIMEOUT',
                        'status': 'timeout', 'time_in_intersection': data['time_in_intersection']
                    }
                    timed_out_event_logs.append(log_event)

        # Perform full cleanup for paths marked for removal
        for path_id_to_remove in ids_to_fully_remove_from_pm:
            self._cleanup_path_data(path_id_to_remove)
            
        return timed_out_event_logs
    
    def _cleanup_path_data(self, bytetrack_id):
        """Removes all internal tracking data for a given bytetrack_id."""
        self.active_paths.pop(bytetrack_id, None)
        self.tracking_history.pop(bytetrack_id, None)
        self.vehicle_type_tracker.pop(bytetrack_id, None)
        self.last_seen_update_time_dt.pop(bytetrack_id, None)
        self.consecutive_misses.pop(bytetrack_id, None)
        debug_print(f"PathMgr: Fully cleaned up data for V_BT:{bytetrack_id}")

    def force_exit_all_remaining(self, final_timestamp_dt):
        forced_exit_logs = []
        for path_id, data in list(self.active_paths.items()): # list() for safe removal
            if data['status'] == 'active':
                actual_exit_time_for_force = self.last_seen_update_time_dt.get(path_id, final_timestamp_dt)
                time_in_intersection_final = (actual_exit_time_for_force - data['entry_time']).total_seconds()
                
                debug_print(f"PathMgr: Forcing exit for V_BT:{path_id} ({data['type']}) at end of video.")
                data.update({
                    'status': 'forced_exit', 'exit_time': actual_exit_time_for_force,
                    'exit_direction': 'FORCED',
                    'time_in_intersection': round(time_in_intersection_final, 2)
                })
                self.completed_paths_log.append(data.copy())
                log_event = {
                    'timestamp_dt': data['exit_time'], 'vehicle_id': path_id,
                    'vehicle_type': data['type'], 'event_type': 'FORCED_EXIT', # Differentiate from timeout
                    'direction_from': data['entry_direction'], 'direction_to': 'FORCED',
                    'status': 'forced_exit', 'time_in_intersection': data['time_in_intersection']
                }
                forced_exit_logs.append(log_event)
                self._cleanup_path_data(path_id) # Clean up after forcing
        # self.active_paths.clear() # Should be empty due to _cleanup_path_data
        return forced_exit_logs

    def get_path_data(self, bytetrack_id): # Helper to get path data if needed
        return self.active_paths.get(bytetrack_id)

    def get_tracking_trail(self, bytetrack_id):
        return list(self.tracking_history.get(bytetrack_id, []))

    def get_active_vehicle_count(self):
        return len([pid for pid, data in self.active_paths.items() if data.get('status') == 'active'])

    def get_completed_paths_for_report(self):
        return self.completed_paths_log