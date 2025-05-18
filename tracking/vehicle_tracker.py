# project/tracking/vehicle_tracker.py
import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timedelta

from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.utils.checks import check_requirements

from config import (
    TRACKER_CONFIG_ARGS, VEHICLE_TIMEOUTS, TRACK_HISTORY_LENGTH,
    MAX_APP_CONSECUTIVE_MISSES
)
from core.utils import debug_print

# Ensure tracker dependencies are met (especially for deepsparse if CPU)
check_requirements('lap') # for linear_assignment

class ByteTrackWrapper:
    """
    Wraps Ultralytics' BYTETracker and adds application-specific logic for
    vehicle lifecycle management (entry, exit, timeouts, consistent type).
    """
    def __init__(self, model_names_dict):
        self.args = TRACKER_CONFIG_ARGS # From config.py
        self.byte_tracker = BYTETracker(args=type('Args', (object,), self.args)(), frame_rate=30) # frame_rate is placeholder
        
        self.model_names = model_names_dict # {class_id: class_name} from YOLO model

        # Application-level state for tracks managed by ByteTrack IDs
        self.active_vehicles = {}          # {track_id: {app_data}}
        self.vehicle_type_history = {}     # {track_id: {type_name: count}} for consistent type
        self.tracking_history = {}         # {track_id: deque([(x,y), ...])} for trails
        self.last_seen_time_video_sec = {} # {track_id: video_time_seconds}
        self.app_consecutive_misses = {}   # {track_id: count} application-level miss counter

        self.completed_paths = []          # Stores data of exited/timed_out vehicles
        self.completed_vehicle_ids = set() # ByteTrack IDs that have finished their journey

        debug_print(f"ByteTrackWrapper initialized with args: {self.args}")

    def _get_center(self, tlbr):
        return (int((tlbr[0] + tlbr[2]) / 2), int((tlbr[1] + tlbr[3]) / 2))

    def process_detections(self, detections_np, frame_img_shape_hw, current_video_time_sec):
        """
        Processes raw detections using BYTETracker and updates internal states.
        Args:
            detections_np (np.ndarray): Detections in format [x1, y1, x2, y2, conf, cls_id]
                                         Coordinates are expected to be in frame_img_shape_hw space.
            frame_img_shape_hw (tuple): (height, width) of the frame these detections belong to.
            current_video_time_sec (float): Timestamp of the current frame in video seconds.
        Returns:
            list: A list of dictionaries, each representing an active track:
                  {'id': track_id, 'box_coords_orig_scale': (x1,y1,x2,y2),
                   'center_orig_scale': (cx,cy), 'type_raw_name': str, 'conf': float}
        """
        active_online_tracks = []
        
        if detections_np is not None and len(detections_np) > 0:
            # BYTETracker.update expects detections and original image shape (h, w)
            # It returns np.array of [x1, y1, x2, y2, track_id, conf, cls, idx]
            online_targets_np = self.byte_tracker.update(detections_np, frame_img_shape_hw)
        else:
            # If no detections, still call update with an empty array to allow ByteTrack to manage lost tracks
            online_targets_np = self.byte_tracker.update(np.empty((0, 6)), frame_img_shape_hw)

        current_track_ids_from_bytetrack = set()

        for t in online_targets_np:
            tlbr = t[:4]
            track_id = int(t[4])
            conf = float(t[5])
            cls_id = int(t[6])
            
            current_track_ids_from_bytetrack.add(track_id)

            if track_id in self.completed_vehicle_ids: # Should not happen if ByteTrack reuses IDs correctly
                debug_print(f"WARN: ByteTrack returned completed ID {track_id}. Ignoring.")
                continue

            # Update core tracking info
            self.last_seen_time_video_sec[track_id] = current_video_time_sec
            center_pt = self._get_center(tlbr)
            self.last_positions[track_id] = center_pt # Keep self.last_positions for get_last_position
            
            if track_id not in self.tracking_history:
                self.tracking_history[track_id] = deque(maxlen=TRACK_HISTORY_LENGTH)
            self.tracking_history[track_id].append(center_pt)
            
            self.app_consecutive_misses[track_id] = 0 # Reset app-level miss counter

            raw_type_name = self.model_names.get(cls_id, f"CLS_{cls_id}")

            active_online_tracks.append({
                'id': track_id, # This is the integer ID from ByteTrack
                'box_coords_orig_scale': tuple(tlbr.astype(int)),
                'center_orig_scale': center_pt,
                'type_raw_name': raw_type_name,
                'conf': conf
            })
        
        # Increment app-level misses for tracks not seen by ByteTrack in this frame
        # This operates on tracks previously known to our wrapper.
        # ByteTrack handles its own lost track buffer.
        for prev_track_id in list(self.last_positions.keys()): # Iterate copy
            if prev_track_id not in current_track_ids_from_bytetrack and \
               prev_track_id not in self.completed_vehicle_ids:
                self.app_consecutive_misses[prev_track_id] = self.app_consecutive_misses.get(prev_track_id, 0) + 1

        return active_online_tracks

    def get_consistent_type(self, track_id, current_type_name):
        if track_id not in self.vehicle_type_history:
            self.vehicle_type_history[track_id] = {current_type_name: 1}
            return current_type_name
        
        counts = self.vehicle_type_history[track_id]
        counts[current_type_name] = counts.get(current_type_name, 0) + 1
        # Return the type with the highest count for this track_id
        return max(counts, key=counts.get)

    def register_entry(self, track_id, entry_direction, entry_time_dt: datetime, entry_point_xy, consistent_vehicle_type):
        if track_id in self.completed_vehicle_ids:
            debug_print(f"WARN: V_ID:{track_id} already completed. Ignoring re-entry attempt.")
            return False
        if track_id in self.active_vehicles:
            # This might happen if a vehicle re-enters a zone it was already in,
            # or if zone logic is complex. For now, update if entry direction changes, else ignore.
            if self.active_vehicles[track_id]['entry_direction'] != entry_direction:
                 debug_print(f"INFO: V_ID:{track_id} re-registering entry from new direction {entry_direction} (was {self.active_vehicles[track_id]['entry_direction']}).")
                 # Potentially handle as an inferred exit from old direction / new entry.
                 # For now, we'll just update the entry details. ByteTrack should maintain the ID.
            else:
                 # debug_print(f"INFO: V_ID:{track_id} already active in {entry_direction}. Ignoring redundant entry registration.")
                 return False # No change in state
        
        self.active_vehicles[track_id] = {
            'id': track_id,
            'type': consistent_vehicle_type, # Store the consistent type
            'entry_direction': entry_direction,
            'entry_time': entry_time_dt,
            'entry_point': entry_point_xy,
            'exit_direction': None,
            'exit_time': None,
            'exit_point': None,
            'status': 'active'
        }
        # Ensure other tracking states are initialized if this is the first time we see this ID at app-level
        if track_id not in self.last_positions: self.last_positions[track_id] = entry_point_xy
        if track_id not in self.tracking_history: self.tracking_history[track_id] = deque([entry_point_xy], maxlen=TRACK_HISTORY_LENGTH)
        if track_id not in self.vehicle_type_history: self.vehicle_type_history[track_id] = {consistent_vehicle_type: 1}
        self.app_consecutive_misses[track_id] = 0 # Reset on new entry
        
        debug_print(f"ENTRY (App): V_ID:{track_id} ({consistent_vehicle_type}) from {entry_direction}")
        return True

    def _finalize_and_complete_path(self, track_id, data_to_complete):
        """Helper to add to completed_paths and mark ID as spent at app-level."""
        if track_id in self.completed_vehicle_ids:
            debug_print(f"ERROR: Attempting to finalize already completed V_ID:{track_id}")
            # Still try to remove data if it's somehow active
            if track_id in self.active_vehicles: self.active_vehicles.pop(track_id, None)
            return

        self.completed_paths.append(data_to_complete.copy())
        self.completed_vehicle_ids.add(track_id) # Mark as completed
        
        # Clean up app-level data for this ID
        self.active_vehicles.pop(track_id, None)
        # Keep ByteTrack specific data (like self.last_positions, self.tracking_history for a short while if ByteTrack itself might reuse,
        # but generally, once application decides it's 'completed', it shouldn't be reused by app logic.)
        # For now, let's remove most app-level state for completed tracks.
        self.vehicle_type_history.pop(track_id, None)
        self.app_consecutive_misses.pop(track_id, None)
        # ByteTrack might keep the STrack object for its buffer duration. We don't interfere with that.

    def register_exit(self, track_id, exit_direction, exit_time_dt: datetime, exit_point_xy):
        if track_id in self.active_vehicles and self.active_vehicles[track_id]['status'] == 'active':
            data = self.active_vehicles[track_id]
            
            # Ensure final type is the most consistent one observed
            final_type = data.get('type', 'UnknownType') # Default to current if no history
            if track_id in self.vehicle_type_history and self.vehicle_type_history[track_id]:
                final_type = max(self.vehicle_type_history[track_id], key=self.vehicle_type_history[track_id].get)
            data['type'] = final_type

            data.update({
                'exit_direction': exit_direction,
                'exit_time': exit_time_dt,
                'exit_point': exit_point_xy,
                'status': 'exited'
            })
            time_in_int = (exit_time_dt - data['entry_time']).total_seconds()
            data['time_in_intersection'] = round(time_in_int, 2)
            
            self._finalize_and_complete_path(track_id, data)
            debug_print(f"EXIT (App): V_ID:{track_id} ({final_type}) to {exit_direction}. Time: {time_in_int:.2f}s. ID retired from app.")
            return True, time_in_int
        
        # debug_print(f"WARN: Exit ignored for inactive/already exited V_ID:{track_id} at app-level.")
        return False, None

    def check_timeouts(self, current_time_dt: datetime):
        timed_out_ids_this_pass = []
        for track_id, data in list(self.active_vehicles.items()): # Iterate copy
            if data['status'] == 'active':
                if track_id in self.completed_vehicle_ids: # Should not happen
                    debug_print(f"ERROR: V_ID:{track_id} is 'active' but also 'completed' during timeout check.")
                    self.active_vehicles.pop(track_id, None) # Clean up inconsistent state
                    continue

                time_in_int_seconds = (current_time_dt - data['entry_time']).total_seconds()
                vehicle_type_for_timeout = data.get('type', 'default')
                timeout_duration_seconds = VEHICLE_TIMEOUTS.get(vehicle_type_for_timeout, VEHICLE_TIMEOUTS['default'])

                if time_in_int_seconds > timeout_duration_seconds:
                    final_type = data.get('type', 'UnknownType')
                    if track_id in self.vehicle_type_history and self.vehicle_type_history[track_id]:
                        final_type = max(self.vehicle_type_history[track_id], key=self.vehicle_type_history[track_id].get)
                    data['type'] = final_type
                    
                    miss_count = self.app_consecutive_misses.get(track_id, 0)
                    debug_print(f"TIMEOUT (App): V_ID:{track_id} ({final_type}) after {time_in_int_seconds:.1f}s (Limit: {timeout_duration_seconds}s). App Misses: {miss_count}. ID retired from app.")
                    data.update({
                        'status': 'timed_out',
                        'exit_time': current_time_dt,
                        'exit_direction': 'TIMEOUT',
                        'time_in_intersection': round(time_in_int_seconds, 2)
                    })
                    self._finalize_and_complete_path(track_id, data)
                    timed_out_ids_this_pass.append(track_id)
        return timed_out_ids_this_pass

    def force_exit_vehicle(self, track_id, exit_time_dt: datetime):
        if track_id in self.active_vehicles and self.active_vehicles[track_id]['status'] == 'active':
            if track_id in self.completed_vehicle_ids: # Defensive check
                debug_print(f"ERROR: V_ID:{track_id} is 'active' but also 'completed' during force_exit.")
                self.active_vehicles.pop(track_id, None)
                return False

            data = self.active_vehicles[track_id]
            final_type = data.get('type', 'UnknownType')
            if track_id in self.vehicle_type_history and self.vehicle_type_history[track_id]:
                final_type = max(self.vehicle_type_history[track_id], key=self.vehicle_type_history[track_id].get)
            data['type'] = final_type
            
            time_in_int_seconds = (exit_time_dt - data['entry_time']).total_seconds()
            data.update({
                'status': 'forced_exit',
                'exit_time': exit_time_dt,
                'exit_direction': 'FORCED',
                'time_in_intersection': round(time_in_int_seconds, 2)
            })
            self._finalize_and_complete_path(track_id, data)
            debug_print(f"FORCED EXIT (App): V_ID:{track_id} ({final_type}). ID retired from app.")
            return True
        return False

    def get_last_position(self, track_id):
        """Returns the last known center position for a track_id."""
        return self.last_positions.get(track_id)

    def get_tracking_trail(self, track_id):
        return list(self.tracking_history.get(track_id, []))

    def get_active_vehicle_count(self):
        return len(self.active_vehicles) # active_vehicles only stores non-completed ones

    def get_completed_paths(self):
        return self.completed_paths
        
    def remove_vehicle_data(self, track_id):
        """
        This function is primarily for explicit cleanup if needed, 
        but _finalize_and_complete_path handles most cleanup internally.
        It should ensure data is removed from all relevant application-level dictionaries.
        """
        removed_count = 0
        if self.active_vehicles.pop(track_id, None) is not None: removed_count +=1
        if self.last_positions.pop(track_id, None) is not None: removed_count +=1
        if self.last_seen_time_video_sec.pop(track_id, None) is not None: removed_count +=1
        if self.tracking_history.pop(track_id, None) is not None: removed_count +=1
        if self.vehicle_type_history.pop(track_id, None) is not None: removed_count +=1
        if self.app_consecutive_misses.pop(track_id, None) is not None: removed_count +=1
        
        # Ensure it's marked as completed if being removed this way
        self.completed_vehicle_ids.add(track_id)
        
        if removed_count > 0:
            debug_print(f"Explicitly cleared app-level tracking data for V_ID:{track_id}.")
        return removed_count > 0

    # Add a property for `last_positions` to avoid direct dict access if preferred by VideoProcessor
    @property
    def current_track_positions(self):
        return self.last_positions