# project/tracking/cleanup.py
from core.utils import debug_print

# This timeout is for app-level state if an ID from ByteTrack disappears
# and doesn't get a proper exit/timeout event from VehicleTracker.
APP_LEVEL_STALE_TRACK_TIMEOUT_SECONDS = 60 # e.g., 1 minute

def cleanup_tracking_data(vehicle_tracker, zone_tracker,
                          current_video_time_sec, # Video time (float seconds) for zone cooldowns
                          current_frame_datetime, # datetime object for current frame processing
                          active_bytetrack_ids_this_frame_set): # set of str(bytetrack_id)
    """
    Cleans up stale application-level tracks in VehicleTracker and internal zone cooldowns.
    active_bytetrack_ids_this_frame_set: A set of string track_ids currently output by ByteTracker.
    """
    app_tracks_to_fully_remove = set()

    # 1. Check VehicleTracker's `active_vehicles`:
    # If a vehicle is "active" in our app, but ByteTrack is no longer outputting it,
    # it might be truly gone or lost for too long by ByteTrack.
    for v_id_str, app_data in list(vehicle_tracker.active_vehicles.items()): # Iterate on copy
        if v_id_str not in active_bytetrack_ids_this_frame_set:
            # This app-active vehicle is NOT in the current ByteTrack output.
            # Check how long since the app last saw an update for it.
            last_seen_dt = vehicle_tracker.vehicle_last_seen_app_time.get(v_id_str)
            if last_seen_dt:
                # Use a shorter timeout here as ByteTrack already has its own buffer
                if (current_frame_datetime - last_seen_dt).total_seconds() > (APP_LEVEL_STALE_TRACK_TIMEOUT_SECONDS / 2.0):
                    debug_print(f"Cleanup: App-active ID:{v_id_str} not in current BT output & app_last_seen timeout. Forcing app exit.")
                    vehicle_tracker.force_exit_vehicle(v_id_str, current_frame_datetime)
                    # force_exit moves to completed_paths and clears from active_vehicles.
                    # Mark for full history cleanup.
                    app_tracks_to_fully_remove.add(v_id_str)
            else:
                # Should not happen if it's in active_vehicles, but as a safeguard:
                debug_print(f"Cleanup: App-active ID:{v_id_str} has no last_seen_app_time. Forcing app exit.")
                vehicle_tracker.force_exit_vehicle(v_id_str, current_frame_datetime)
                app_tracks_to_fully_remove.add(v_id_str)


    # 2. Check `vehicle_last_seen_app_time` for general history pruning:
    # These are IDs that ByteTrack might have tracked at some point, but are not currently
    # active in the app's `active_vehicles` list (e.g., they exited normally).
    # We want to remove their history if they haven't been seen at all for a while.
    all_known_app_ids = list(vehicle_tracker.vehicle_last_seen_app_time.keys())
    for v_id_str in all_known_app_ids:
        if v_id_str not in vehicle_tracker.active_vehicles and \
           v_id_str not in active_bytetrack_ids_this_frame_set: # Not app-active AND not in current BT output
            
            last_seen_dt = vehicle_tracker.vehicle_last_seen_app_time.get(v_id_str)
            if last_seen_dt and \
               (current_frame_datetime - last_seen_dt).total_seconds() > APP_LEVEL_STALE_TRACK_TIMEOUT_SECONDS:
                app_tracks_to_fully_remove.add(v_id_str) # Mark for full history cleanup

    count_fully_removed = 0
    for v_id_str in app_tracks_to_fully_remove:
        vehicle_tracker.remove_all_data_for_id(v_id_str) # Clears history, last_seen_app_time etc.
        if zone_tracker: zone_tracker.remove_vehicle_data(v_id_str) # Also clear from zone tracker cooldowns
        count_fully_removed +=1
    
    if count_fully_removed > 0:
        debug_print(f"Cleanup: Fully removed app data for {count_fully_removed} stale/exited IDs.")

    # 3. ZoneTracker internal cooldowns (uses video time float)
    if zone_tracker and hasattr(zone_tracker, 'cleanup_cooldowns'):
        zone_tracker.cleanup_cooldowns(current_video_time_sec)

    return count_fully_removed