# project/pipelines/pass2_path_analyzer.py
import pandas as pd
import numpy as np
from collections import Counter

# Ensure config is accessible, e.g., by passing the config module or specific dicts
# For simplicity in this example, we'll assume 'config' is an object passed in.

def get_dominant_zones_for_frame(poi_zone_insideness_dict, poi_quorum_method, poi_quorum_value):
    """
    Determines the dominant zone(s) a vehicle is in for a single frame based on its PoIs.
    
    Args:
        poi_zone_insideness_dict (dict): {'poi_name': ['zone_A', ...], ...}
        poi_quorum_method (str): 'min_count' or 'percentage'
        poi_quorum_value (float or int): The threshold for the quorum.
        
    Returns:
        set: A set of dominant zone names for this frame. Can be empty.
    """
    if not poi_zone_insideness_dict:
        return set()

    all_poi_zones = []
    for poi_name, zones_for_poi in poi_zone_insideness_dict.items():
        all_poi_zones.extend(zones_for_poi)
    
    if not all_poi_zones:
        return set()

    zone_counts = Counter(all_poi_zones)
    dominant_zones = set()
    num_pois_with_any_zone_data = len(poi_zone_insideness_dict) # Number of PoIs that had data

    for zone, count in zone_counts.items():
        if poi_quorum_method == 'min_count':
            if count >= poi_quorum_value:
                dominant_zones.add(zone)
        elif poi_quorum_method == 'percentage':
            # Percentage of PoIs that reported being in this zone
            if num_pois_with_any_zone_data > 0 and (count / num_pois_with_any_zone_data) >= poi_quorum_value:
                dominant_zones.add(zone)
        # Add other quorum methods if needed
            
    return dominant_zones


def smooth_zone_occupancy(zone_occupancy_sequence, window_size):
    """
    Applies a mode filter to smooth the zone occupancy sequence.
    Each element in zone_occupancy_sequence is a frozenset of dominant zones for a frame.
    """
    if not zone_occupancy_sequence or window_size <= 1:
        return zone_occupancy_sequence

    smoothed_sequence = []
    n = len(zone_occupancy_sequence)
    for i in range(n):
        window_start = max(0, i - window_size // 2)
        window_end = min(n, i + window_size // 2 + 1)
        current_window = zone_occupancy_sequence[window_start:window_end]
        
        if not current_window:
            smoothed_sequence.append(frozenset()) # Or previous value
            continue
            
        # Count occurrences of each zone *set* in the window
        # frozenset is hashable and can be a key in Counter
        set_counts = Counter(current_window)
        if not set_counts:
             smoothed_sequence.append(frozenset())
             continue
        
        # Find the most common set of zones
        most_common_zone_set, _ = set_counts.most_common(1)[0]
        smoothed_sequence.append(most_common_zone_set)
        
    return smoothed_sequence

def segment_into_zone_stays(track_df, smoothed_zone_col='smoothed_dominant_zones', min_frames_for_stay=3):
    """
    Segments a track's smoothed zone occupancy into stable zone stays.
    Assumes track_df is sorted by 'frame_num' and has a 0-based integer index.

    Args:
        track_df (pd.DataFrame): DataFrame for a single track.
                                 Must contain 'frame_num', 'timestamp_dt_str', 
                                 'timestamp_sec', and smoothed_zone_col.
        smoothed_zone_col (str): Column name with the frozenset of smoothed dominant zones.
        min_frames_for_stay (int): Minimum consecutive frames to consider a stable stay.

    Returns:
        list: A list of dictionaries, each representing a zone stay:
              {'zone_set': frozenset_of_zones, 
               'entry_frame': int, 'exit_frame': int,
               'entry_time_dt': datetime, 'exit_time_dt': datetime,
               'entry_timestamp_sec': float, 'exit_timestamp_sec': float,
               'duration_frames': int, 'duration_sec': float
              }
    """
    stays = []
    if track_df.empty or smoothed_zone_col not in track_df.columns or len(track_df) < min_frames_for_stay:
        return stays

    # Variables to keep track of the current potential stay
    current_stay_zone_set = None
    current_stay_start_idx = -1  # Use DataFrame index for start

    # Iterate through the DataFrame rows (idx will be 0, 1, 2, ...)
    for idx, row in track_df.iterrows():
        dominant_zone_set_now = row[smoothed_zone_col]

        if current_stay_start_idx == -1: # Start of processing or after a stay was finalized
            if dominant_zone_set_now: # Start a new potential stay if we are in some zone(s)
                current_stay_zone_set = dominant_zone_set_now
                current_stay_start_idx = idx
            # If dominant_zone_set_now is empty, we are not in a zone, so just continue
            continue

        # If the zone set changes OR if it's the last row of the DataFrame
        if dominant_zone_set_now != current_stay_zone_set or idx == len(track_df) - 1:
            # Determine the end index of the just-finished stay
            # If zone changed, the previous row (idx-1) was the end of the stay.
            # If it's the last row and zones are the same, this current row (idx) is the end.
            end_idx_of_stay = idx -1 # Default: previous row was the end
            if dominant_zone_set_now == current_stay_zone_set and idx == len(track_df) - 1:
                end_idx_of_stay = idx # Current (last) row is the end

            # Check if the potential stay was long enough
            # Number of frames in the stay is (end_idx_of_stay - current_stay_start_idx + 1)
            num_frames_in_stay = (end_idx_of_stay - current_stay_start_idx + 1)

            if current_stay_zone_set and num_frames_in_stay >= min_frames_for_stay:
                start_row_data = track_df.iloc[current_stay_start_idx]
                end_row_data = track_df.iloc[end_idx_of_stay]

                entry_time_dt = pd.to_datetime(start_row_data['timestamp_dt_str'])
                exit_time_dt = pd.to_datetime(end_row_data['timestamp_dt_str'])
                duration_sec = (exit_time_dt - entry_time_dt).total_seconds()
                # For frame duration, if using timestamp_sec directly:
                # duration_sec_alt = end_row_data['timestamp_sec'] - start_row_data['timestamp_sec']
                # Ensure consistency or pick one method. Using datetime difference is safer.
                # However, if timestamps are not perfectly monotonic with frame rate, frame count is better.
                # For segment duration in seconds, actual timestamps are better.

                stays.append({
                    'zone_set': current_stay_zone_set, # The set of zones for this stay
                    'entry_frame': int(start_row_data['frame_num']),
                    'exit_frame': int(end_row_data['frame_num']),
                    'entry_time_dt': entry_time_dt,
                    'exit_time_dt': exit_time_dt,
                    'entry_timestamp_sec': float(start_row_data['timestamp_sec']),
                    'exit_timestamp_sec': float(end_row_data['timestamp_sec']),
                    'duration_frames': num_frames_in_stay,
                    'duration_sec': round(duration_sec, 3)
                })

            # Reset for the next potential stay:
            # If the current row (dominant_zone_set_now) is part of a new potential stay
            if dominant_zone_set_now: # If current frame is in some zone(s)
                current_stay_zone_set = dominant_zone_set_now
                current_stay_start_idx = idx
            else: # Current frame is not in any zone, so reset
                current_stay_zone_set = None
                current_stay_start_idx = -1
                
    return stays

def derive_paths_from_stays(track_id, class_name, zone_stays_list, valid_movements_config):
    """
    Derives journey paths (entry_zone -> exit_zone) from a sequence of zone stays.
    This version simplifies by considering only stays in a *single* primary zone.
    Overlaps (zone_set with >1 zone) are treated as transitional and ignored for path start/end.
    """
    paths = []
    if len(zone_stays_list) < 2: # Need at least two stays to form a path
        return paths

    for i in range(len(zone_stays_list) - 1):
        current_stay = zone_stays_list[i]
        next_stay = zone_stays_list[i+1]

        current_zone_set = current_stay['zone_set']
        next_zone_set = next_stay['zone_set']

        # For simplicity, only consider paths between single, primary zones
        # If a stay is in an overlap (len(zone_set) > 1) or empty, it can't be a start/end of a simple path.
        if len(current_zone_set) == 1 and len(next_zone_set) == 1:
            entry_direction = list(current_zone_set)[0]
            exit_direction = list(next_zone_set)[0]

            if entry_direction == exit_direction: # Not a movement to a different zone
                continue

            # Check against VALID_MOVEMENTS
            if entry_direction in valid_movements_config and \
               exit_direction in valid_movements_config.get(entry_direction, []):
                
                entry_time = current_stay['entry_time_dt']
                # Exit time from the entry_direction is the end of the current_stay.
                # Entry time into the exit_direction is the start of the next_stay.
                # For a path segment A->B, the "exit" time is when it *enters B*.
                exit_time = next_stay['entry_time_dt'] 
                time_in_segment = (exit_time - entry_time).total_seconds()

                paths.append({
                    'track_id': track_id,
                    'class_name': class_name,
                    'entry_direction': entry_direction,
                    'exit_direction': exit_direction,
                    'entry_time_dt': entry_time,
                    'exit_time_dt': exit_time,
                    'time_in_segment_sec': time_in_segment,
                    'status': 'completed_segment' # This is an initial, unstitched segment
                })
    return paths


def analyze_initial_paths(raw_tracks_parquet_path, 
                          initial_paths_output_path, 
                          fragmented_tracks_output_path, 
                          app_config): # Pass the main config module/object
    """
    Main function for Pass 2.
    """
    print(f"Pass 2: Reading raw track data from {raw_tracks_parquet_path}...")
    try:
        df_raw_tracks = pd.read_parquet(raw_tracks_parquet_path)
    except Exception as e:
        print(f"Error reading Parquet file {raw_tracks_parquet_path}: {e}")
        return

    if df_raw_tracks.empty:
        print("Pass 2: Raw track data is empty. Nothing to process.")
        # Create empty output files if they are expected by subsequent passes
        pd.DataFrame().to_parquet(initial_paths_output_path, index=False)
        pd.DataFrame().to_parquet(fragmented_tracks_output_path, index=False)
        return

    all_complete_paths = []
    all_fragment_data = [] # Info about tracks that are candidates for stitching

    # Config parameters for Pass 2
    poi_quorum_method = app_config.PATH_ANALYSIS_PARAMS.get('poi_quorum_method', 'min_count')
    poi_quorum_value = app_config.PATH_ANALYSIS_PARAMS.get('poi_quorum_value_min_count', 1) # Default to 1 PoI if not specified
    if poi_quorum_method == 'percentage':
         poi_quorum_value = app_config.PATH_ANALYSIS_PARAMS.get('poi_quorum_value_percentage', 0.51)

    smoothing_window = app_config.PATH_ANALYSIS_PARAMS.get('zone_occupancy_smoothing_window_frames', 3)
    min_stay_frames = app_config.PATH_ANALYSIS_PARAMS.get('min_frames_for_stable_zone_stay', 3)
    valid_movements = app_config.VALID_MOVEMENTS

    # Group by track_id
    grouped_tracks = df_raw_tracks.groupby('track_id')
    total_tracks = len(grouped_tracks)
    print(f"Pass 2: Processing {total_tracks} unique track_ids...")

    processed_count = 0
    for track_id, track_df in grouped_tracks:
        if processed_count % 100 == 0:
            print(f"Pass 2: Processed {processed_count}/{total_tracks} track_ids...")
        processed_count += 1

        track_df = track_df.sort_values(by='frame_num').reset_index(drop=True) # Ensure order
        
        # 1. Determine dominant zone(s) per frame
        track_df['dominant_zones_set'] = track_df['poi_zone_insideness'].apply(
            lambda x: get_dominant_zones_for_frame(x, poi_quorum_method, poi_quorum_value)
        )
        # Convert to frozenset for hashing/counting in smoothing
        track_df['dominant_zones_set_frozen'] = track_df['dominant_zones_set'].apply(frozenset)

        # 2. Smooth zone occupancy
        zone_sequence = track_df['dominant_zones_set_frozen'].tolist()
        smoothed_zone_sequence = smooth_zone_occupancy(zone_sequence, smoothing_window)
        track_df['smoothed_dominant_zones'] = smoothed_zone_sequence
        
        # 3. Segment into zone stays
        zone_stays = segment_into_zone_stays(track_df, 
                                             smoothed_zone_col='smoothed_dominant_zones', 
                                             min_frames_for_stay=min_stay_frames)
        
        # 4. Derive paths from stays
        class_name_series = track_df['class_name'].mode() # Get the most frequent class name
        class_name = class_name_series[0] if not class_name_series.empty else "Unknown"
        
        derived_paths_for_track = derive_paths_from_stays(track_id, class_name, zone_stays, valid_movements)
        
        # 5. Classify track and collect data
        if derived_paths_for_track:
            all_complete_paths.extend(derived_paths_for_track)
            # If a track has at least one derived path, we consider it "handled" for now.
            # More complex logic could decide if parts of it are still fragments.
            # For this iteration, if it forms any path, it's not primarily a fragment for stitching.
        else:
            # This track did not form any valid zone-to-zone paths. It's a fragment.
            if not track_df.empty:
                first_frame_data = track_df.iloc[0]
                last_frame_data = track_df.iloc[-1]
                
                # Collect necessary info for stitching pass
                # Need image patch references for start and end of fragment
                all_fragment_data.append({
                    'track_id': track_id,
                    'class_name': class_name,
                    'start_frame': first_frame_data['frame_num'],
                    'end_frame': last_frame_data['frame_num'],
                    'start_time_dt_str': first_frame_data['timestamp_dt_str'],
                    'end_time_dt_str': last_frame_data['timestamp_dt_str'],
                    'start_timestamp_sec': first_frame_data['timestamp_sec'],
                    'end_timestamp_sec': last_frame_data['timestamp_sec'],
                    'start_bbox_zone_space': first_frame_data['bbox_zone_space_xyxy'],
                    'end_bbox_zone_space': last_frame_data['bbox_zone_space_xyxy'],
                    'start_image_patch_ref': first_frame_data.get('image_patch_ref'), # Use .get for safety
                    'end_image_patch_ref': last_frame_data.get('image_patch_ref'),
                    'num_frames_in_fragment': len(track_df)
                })

    print(f"Pass 2: Processing complete. Found {len(all_complete_paths)} initial complete path segments.")
    print(f"Pass 2: Identified {len(all_fragment_data)} fragmented tracks for potential stitching.")

    # Save outputs
    df_complete_paths = pd.DataFrame(all_complete_paths)
    df_fragment_data = pd.DataFrame(all_fragment_data)

    try:
        df_complete_paths.to_parquet(initial_paths_output_path, index=False, engine='pyarrow')
        print(f"Pass 2: Initial complete paths saved to {initial_paths_output_path}")
    except Exception as e:
        print(f"Error saving complete paths to {initial_paths_output_path}: {e}")

    try:
        df_fragment_data.to_parquet(fragmented_tracks_output_path, index=False, engine='pyarrow')
        print(f"Pass 2: Fragmented track data saved to {fragmented_tracks_output_path}")
    except Exception as e:
        print(f"Error saving fragmented tracks to {fragmented_tracks_output_path}: {e}")