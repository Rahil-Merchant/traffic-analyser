# project/pipelines/pass3_fragment_stitcher.py
import pandas as pd
import numpy as np
import os
import cv2 # For loading image patches
from sklearn.metrics.pairwise import cosine_similarity
import torch # For Re-ID model inference
from PIL import Image # If your Re-ID model needs PIL images
import torchvision.transforms as T # Common for Re-ID preprocessing

# Assuming config and model loader are accessible
# from config import REID_MODEL_SETTINGS, FRAGMENT_STITCHING_PARAMS, FILE_PATHS, ... (Pass app_config)
from models.model_loader import load_reid_model
# We'll need the path analysis functions from Pass 2 to re-analyze stitched tracks
from pipelines.pass2_path_analyzer import (
    get_dominant_zones_for_frame, 
    smooth_zone_occupancy,
    segment_into_zone_stays,
    derive_paths_from_stays
)


def get_reid_features(image_patches_batch, reid_model, reid_config, device):
    """
    Extracts Re-ID features for a batch of image patches.
    This function needs to be tailored to your Re-ID model's input requirements.
    """
    if not image_patches_batch:
        return []

    # Example preprocessing (common for many Re-ID models)
    # You MUST adapt this to your Re-ID model's specific preprocessing.
    reid_input_size = tuple(reid_config.get('input_size', (256, 128))) # H, W
    
    preprocess = T.Compose([
        T.Resize(reid_input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet defaults
    ])

    batch_tensors = []
    for img_patch_np in image_patches_batch:
        if img_patch_np is None or img_patch_np.size == 0: 
            # Handle cases where a patch might be missing or invalid
            # Append a dummy tensor or skip, then handle in results
            # For now, let's assume we need to maintain batch order
            # This could be a zero tensor of the expected feature_dim if an error occurs.
             batch_tensors.append(torch.zeros((3, reid_input_size[0], reid_input_size[1]))) # Dummy
             continue
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img_patch_np, cv2.COLOR_BGR2RGB))
            tensor = preprocess(img_pil)
            batch_tensors.append(tensor)
        except Exception as e:
            print(f"Warning: Could not preprocess image patch for ReID: {e}")
            batch_tensors.append(torch.zeros((3, reid_input_size[0], reid_input_size[1]))) # Dummy on error


    if not batch_tensors:
        return []
        
    input_batch = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        features = reid_model(input_batch) # Output shape: (batch_size, feature_dim)
    
    return features.cpu().numpy()


def aggregate_features(features_list, method='average_last_k', k=5):
    """Aggregates a list of features (e.g., from last k frames of a tracklet)."""
    if not features_list:
        return None
    
    valid_features = [f for f in features_list if f is not None and not np.all(f == 0)] # Filter out dummy/zero features
    if not valid_features:
        return None

    if method == 'average_last_k':
        selected_features = valid_features[-k:]
        if not selected_features: return None
        return np.mean(selected_features, axis=0)
    # Add other aggregation methods if needed (e.g., 'use_best_quality_last_k')
    else:
        return valid_features[-1] # Default to last valid feature


def calculate_motion_consistency(frag1_end_data, frag2_start_data, max_time_gap, max_spatial_dist, assume_max_speed):
    """
    Calculates a simple motion consistency score.
    fragX_data: dict with 'end_time_dt_str'/'start_time_dt_str', 'end_bbox_zone_space'/'start_bbox_zone_space'
    Returns a score between 0 and 1 (1 is best), or 0 if inconsistent.
    """
    time_gap = (pd.to_datetime(frag2_start_data['start_time_dt_str']) - 
                pd.to_datetime(frag1_end_data['end_time_dt_str'])).total_seconds()

    if time_gap < 0 or time_gap > max_time_gap:
        return 0.0 # Invalid temporal order or too large a gap

    f1_x1, f1_y1, f1_x2, f1_y2 = frag1_end_data['end_bbox_zone_space']
    f2_x1, f2_y1, f2_x2, f2_y2 = frag2_start_data['start_bbox_zone_space']

    f1_center_x, f1_center_y = (f1_x1 + f1_x2) / 2, (f1_y1 + f1_y2) / 2
    f2_center_x, f2_center_y = (f2_x1 + f2_x2) / 2, (f2_y1 + f2_y2) / 2
    
    spatial_dist = np.sqrt((f2_center_x - f1_center_x)**2 + (f2_center_y - f1_center_y)**2)

    # Simple gating: if too far based on max assumed speed and time gap
    max_plausible_dist = assume_max_speed * time_gap 
    # Add a buffer to max_plausible_dist because speed is an estimate
    if spatial_dist > max_plausible_dist + (max_spatial_dist * 0.5) and spatial_dist > max_spatial_dist : # If way beyond reasonable travel or fixed max
         return 0.0

    # Normalize spatial distance (closer is better)
    # This score is 1 if dist=0, decreases to 0 as dist approaches max_spatial_dist
    spatial_score = max(0, 1 - (spatial_dist / max_spatial_dist))
    
    # Can add size similarity, velocity consistency later if needed.
    # For now, a simple spatial score.
    return spatial_score


def stitch_fragments(fragmented_tracks_parquet_path, 
                     raw_tracks_parquet_path, # Needed to get all frames of a fragment
                     stitched_paths_output_path, 
                     app_config): # Main config module
    """
    Main function for Pass 3.
    """
    print(f"Pass 3: Reading fragmented track data from {fragmented_tracks_parquet_path}...")
    try:
        df_fragments = pd.read_parquet(fragmented_tracks_parquet_path)
        if df_fragments.empty:
            print("Pass 3: No fragments to stitch. Saving empty output.")
            pd.DataFrame().to_parquet(stitched_paths_output_path, index=False)
            return
    except Exception as e:
        print(f"Error reading fragments Parquet file {fragmented_tracks_parquet_path}: {e}")
        pd.DataFrame().to_parquet(stitched_paths_output_path, index=False) # Save empty
        return

    print(f"Pass 3: Reading full raw track data from {raw_tracks_parquet_path} for stitch details...")
    try:
        df_all_raw_tracks = pd.read_parquet(raw_tracks_parquet_path)
        if df_all_raw_tracks.empty:
            print("Pass 3: Full raw track data is empty. Cannot proceed with stitching.")
            pd.DataFrame().to_parquet(stitched_paths_output_path, index=False)
            return
    except Exception as e:
        print(f"Error reading raw tracks Parquet file {raw_tracks_parquet_path}: {e}")
        pd.DataFrame().to_parquet(stitched_paths_output_path, index=False)
        return

    # Load Re-ID Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        reid_model = load_reid_model(app_config.REID_MODEL_SETTINGS, device)
    except Exception as e:
        print(f"Pass 3: Could not load Re-ID model. Skipping stitching. Error: {e}")
        pd.DataFrame().to_parquet(stitched_paths_output_path, index=False)
        return

    # --- Prepare fragment data for easier lookup ---
    # Sort fragments by their end times (for F1) and start times (for F2)
    df_fragments_ends = df_fragments.sort_values(by='end_timestamp_sec').copy()
    df_fragments_starts = df_fragments.sort_values(by='start_timestamp_sec').copy()

    # Store image patch references from df_all_raw_tracks for quick lookup
    # This assumes 'image_patch_ref' is relative to FILE_PATHS['pass1_image_patch_output_dir']
    # And 'track_id' and 'frame_num' can uniquely identify a patch reference.
    patch_ref_map = {}
    if app_config.PASS1_SAVE_IMAGE_PATCHES:
        for _, row in df_all_raw_tracks[['track_id', 'frame_num', 'image_patch_ref']].dropna().iterrows():
            patch_ref_map[(row['track_id'], row['frame_num'])] = row['image_patch_ref']
    
    image_patch_base_dir = app_config.FILE_PATHS['pass1_image_patch_output_dir']

    # --- Main Stitching Loop ---
    stitched_track_ids_map = {} # To map old fragment IDs to new stitched track IDs
    next_new_stitched_track_id_num = 0
    
    # To avoid re-stitching already stitched fragments
    # When F1 is stitched with F2, F2 should not be considered as an F1 later.
    consumed_fragment_ids = set() 
    
    newly_stitched_raw_data_dfs = [] # List to hold DataFrames of newly formed stitched tracks

    print(f"Pass 3: Attempting to stitch {len(df_fragments_ends)} fragments...")
    for idx1, frag1_data in df_fragments_ends.iterrows():
        frag1_id = frag1_data['track_id']
        if frag1_id in consumed_fragment_ids:
            continue

        # Get Re-ID features for end of frag1
        frag1_end_patches_np = []
        frag1_raw_df = df_all_raw_tracks[df_all_raw_tracks['track_id'] == frag1_id].sort_values(by='frame_num')
        k_frames = app_config.REID_MODEL_SETTINGS.get('k_frames_for_reid_patch_aggregation', 5)
        
        for _, frame_row in frag1_raw_df.tail(k_frames).iterrows():
            patch_ref = patch_ref_map.get((frag1_id, frame_row['frame_num']))
            if patch_ref:
                try:
                    patch_img = cv2.imread(os.path.join(image_patch_base_dir, patch_ref))
                    if patch_img is not None: frag1_end_patches_np.append(patch_img)
                except Exception as e_img: print(f"Warn: Failed to load patch {patch_ref}: {e_img}")
        
        if not frag1_end_patches_np: continue # No patches for Re-ID
        frag1_features_list = get_reid_features(frag1_end_patches_np, reid_model, app_config.REID_MODEL_SETTINGS, device)
        frag1_agg_feature = aggregate_features(frag1_features_list, 
                                             app_config.REID_MODEL_SETTINGS.get('patch_aggregation_method_reid', 'average_last_k'),
                                             k_frames)
        if frag1_agg_feature is None: continue

        best_candidate_frag2_id = None
        best_candidate_score = -1

        # Find candidate F2s
        max_gap = app_config.FRAGMENT_STITCHING_PARAMS['max_temporal_gap_seconds_for_reid']
        min_start_time_f2 = frag1_data['end_timestamp_sec'] # F2 must start after F1 ends
        max_start_time_f2 = frag1_data['end_timestamp_sec'] + max_gap
        
        # Efficiently find candidate F2s that start in the time window
        potential_f2s = df_fragments_starts[
            (df_fragments_starts['start_timestamp_sec'] > min_start_time_f2) &
            (df_fragments_starts['start_timestamp_sec'] <= max_start_time_f2) &
            (df_fragments_starts['track_id'] != frag1_id) & # Not itself
            (~df_fragments_starts['track_id'].isin(consumed_fragment_ids)) # Not already consumed
        ]

        for idx2, frag2_data in potential_f2s.iterrows():
            frag2_id = frag2_data['track_id']
            
            # Motion Consistency (basic check for now)
            motion_score = calculate_motion_consistency(
                frag1_data, frag2_data, 
                max_gap,
                app_config.FRAGMENT_STITCHING_PARAMS['max_spatial_distance_pixels_for_reid'],
                app_config.FRAGMENT_STITCHING_PARAMS.get('assume_max_avg_speed_pixels_per_sec', 100)
            )
            if motion_score < app_config.FRAGMENT_STITCHING_PARAMS.get('min_motion_consistency_score_stitch', 0.01) and \
               app_config.FRAGMENT_STITCHING_PARAMS.get('min_motion_consistency_score_stitch', 0.01) > 0: # Allow disabling motion check
                continue

            # Re-ID for frag2_start
            frag2_start_patches_np = []
            frag2_raw_df = df_all_raw_tracks[df_all_raw_tracks['track_id'] == frag2_id].sort_values(by='frame_num')
            for _, frame_row in frag2_raw_df.head(k_frames).iterrows():
                patch_ref = patch_ref_map.get((frag2_id, frame_row['frame_num']))
                if patch_ref:
                    try:
                        patch_img = cv2.imread(os.path.join(image_patch_base_dir, patch_ref))
                        if patch_img is not None: frag2_start_patches_np.append(patch_img)
                    except: pass # Ignore if patch fails
            
            if not frag2_start_patches_np: continue
            frag2_features_list = get_reid_features(frag2_start_patches_np, reid_model, app_config.REID_MODEL_SETTINGS, device)
            frag2_agg_feature = aggregate_features(frag2_features_list,
                                                 app_config.REID_MODEL_SETTINGS.get('patch_aggregation_method_reid', 'average_last_k'),
                                                 k_frames)
            if frag2_agg_feature is None: continue

            # Calculate Re-ID Similarity
            reid_sim = cosine_similarity(frag1_agg_feature.reshape(1, -1), frag2_agg_feature.reshape(1, -1))[0, 0]

            if reid_sim < app_config.FRAGMENT_STITCHING_PARAMS['min_reid_similarity_stitch']:
                continue

            # Combine scores (example)
            final_score = (reid_sim * app_config.FRAGMENT_STITCHING_PARAMS['stitch_score_weights']['reid'] +
                           motion_score * app_config.FRAGMENT_STITCHING_PARAMS['stitch_score_weights']['motion'])
            
            if final_score > best_candidate_score and final_score >= app_config.FRAGMENT_STITCHING_PARAMS['final_stitch_score_threshold']:
                best_candidate_score = final_score
                best_candidate_frag2_id = frag2_id
        
        # If a best candidate is found, perform the stitch
        if best_candidate_frag2_id:
            print(f"Stitching: {frag1_id} --> {best_candidate_frag2_id} (Score: {best_candidate_score:.2f})")
            
            # Create a new stitched track ID
            new_stitched_id = f"stitched_{next_new_stitched_track_id_num}"
            next_new_stitched_track_id_num += 1
            
            # Get all raw data for frag1 and frag2
            df1_full = df_all_raw_tracks[df_all_raw_tracks['track_id'] == frag1_id].copy()
            df2_full = df_all_raw_tracks[df_all_raw_tracks['track_id'] == best_candidate_frag2_id].copy()
            
            # Assign the new track_id to both
            df1_full['track_id'] = new_stitched_id
            df2_full['track_id'] = new_stitched_id 
            # Could also add original_track_ids column if needed for tracing
            
            # Concatenate them
            df_stitched_full_raw = pd.concat([df1_full, df2_full]).sort_values(by=['frame_num', 'timestamp_sec']).reset_index(drop=True)
            newly_stitched_raw_data_dfs.append(df_stitched_full_raw)

            stitched_track_ids_map[frag1_id] = new_stitched_id
            stitched_track_ids_map[best_candidate_frag2_id] = new_stitched_id
            consumed_fragment_ids.add(frag1_id)
            consumed_fragment_ids.add(best_candidate_frag2_id)


    print(f"Pass 3: Stitching attempts complete. {len(newly_stitched_raw_data_dfs)} new stitched tracks formed.")

    # --- Re-analyze newly stitched tracks to get their path segments ---
    stitched_paths_segments = []
    if newly_stitched_raw_data_dfs:
        print("Pass 3: Re-analyzing zone interactions for newly stitched tracks...")
        # Config params for path analysis (same as Pass 2)
        poi_quorum_method = app_config.PATH_ANALYSIS_PARAMS.get('poi_quorum_method', 'min_count')
        poi_quorum_value = app_config.PATH_ANALYSIS_PARAMS.get('poi_quorum_value_min_count', 1)
        if poi_quorum_method == 'percentage':
            poi_quorum_value = app_config.PATH_ANALYSIS_PARAMS.get('poi_quorum_value_percentage', 0.51)
        smoothing_window = app_config.PATH_ANALYSIS_PARAMS.get('zone_occupancy_smoothing_window_frames', 3)
        min_stay_frames = app_config.PATH_ANALYSIS_PARAMS.get('min_frames_for_stable_zone_stay', 3)
        valid_movements = app_config.VALID_MOVEMENTS

        for df_stitched_track in newly_stitched_raw_data_dfs:
            if df_stitched_track.empty: continue
            
            current_stitched_track_id = df_stitched_track['track_id'].iloc[0] # They all share the new ID
            
            # Re-do Pass 2 logic for this stitched track
            df_stitched_track['dominant_zones_set'] = df_stitched_track['poi_zone_insideness'].apply(
                lambda x: get_dominant_zones_for_frame(x, poi_quorum_method, poi_quorum_value)
            )
            df_stitched_track['dominant_zones_set_frozen'] = df_stitched_track['dominant_zones_set'].apply(frozenset)
            zone_sequence_stitched = df_stitched_track['dominant_zones_set_frozen'].tolist()
            smoothed_zone_sequence_stitched = smooth_zone_occupancy(zone_sequence_stitched, smoothing_window)
            df_stitched_track['smoothed_dominant_zones'] = smoothed_zone_sequence_stitched
            
            zone_stays_stitched = segment_into_zone_stays(df_stitched_track, 
                                                        smoothed_zone_col='smoothed_dominant_zones', 
                                                        min_frames_for_stay=min_stay_frames)
            
            class_name_series_stitched = df_stitched_track['class_name'].mode()
            class_name_stitched = class_name_series_stitched[0] if not class_name_series_stitched.empty else "Unknown"
            
            derived_paths_stitched = derive_paths_from_stays(current_stitched_track_id, 
                                                             class_name_stitched, 
                                                             zone_stays_stitched, 
                                                             valid_movements)
            if derived_paths_stitched:
                stitched_paths_segments.extend(derived_paths_stitched)
    
    df_stitched_final_paths = pd.DataFrame(stitched_paths_segments)
    try:
        df_stitched_final_paths.to_parquet(stitched_paths_output_path, index=False, engine='pyarrow')
        print(f"Pass 3: Path segments from stitched tracks saved to {stitched_paths_output_path}")
    except Exception as e:
        print(f"Error saving stitched paths to {stitched_paths_output_path}: {e}")