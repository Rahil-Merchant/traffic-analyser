# project/tracking/cleanup.py
from loguru import logger

def cleanup_tracking_data(video_processor_instance, current_frame_time_seconds):
    """Removes data for vehicles that are no longer tracked or very old."""
    if not hasattr(video_processor_instance, 'active_track_data'):
        return 0

    stale_ids_app_level = []
    # Example: if a track in active_track_data hasn't been updated (via last_seen_frame)
    # for a very long time, it's an error or an edge case.
    app_level_timeout_frames = video_processor_instance.bytetracker.max_time_lost * 5 # Much longer than bytetrack
    
    current_video_frame_num = video_processor_instance.current_bytetrack_frame_id # Approximate current frame

    for track_id, data in list(video_processor_instance.active_track_data.items()):
        if (current_video_frame_num - data.get('last_seen_frame', current_video_frame_num)) > app_level_timeout_frames:
            stale_ids_app_level.append(track_id)

    count = 0
    for v_id in stale_ids_app_level:
        if v_id in video_processor_instance.active_track_data:
            del video_processor_instance.active_track_data[v_id]
            count += 1
        if video_processor_instance.zone_tracker:
            video_processor_instance.zone_tracker.remove_vehicle_data(v_id)
    
    if count > 0:
        logger.debug(f"AppLevelCleanup: Removed data for {count} very stale tracks from active_track_data.")

    # Cleanup zone cooldowns (this is still relevant)
    if video_processor_instance.zone_tracker and hasattr(video_processor_instance.zone_tracker, 'cleanup_cooldowns'):
        video_processor_instance.zone_tracker.cleanup_cooldowns(current_frame_time_seconds)

    return count