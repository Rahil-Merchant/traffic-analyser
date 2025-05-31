# project/main.py
import os
import sys
# Assuming your config object 'app_config' will be properly loaded or imported
import config as app_config # Import the config module directly

from core.raw_data_extractor import RawDataExtractor
# NEW: Import Pass 2 function
from pipelines.pass2_path_analyzer import analyze_initial_paths
from pipelines.pass3_fragment_stitcher import stitch_fragments
# from reporting.excel_report import create_final_report 

from core.utils import setup_torch_global_settings, setup_thread_affinity # If used

def run_full_pipeline():
    print("--- Starting Emaan Traffic Application Pipeline ---")
    # print(f"LINE_MODE: {app_config.LINE_MODE.upper()}") # If LINE_MODE is still relevant

    if hasattr(app_config, 'SETUP_TORCH_GLOBAL_SETTINGS') and app_config.SETUP_TORCH_GLOBAL_SETTINGS: # Example if you add such a flag
        setup_torch_global_settings()
    if hasattr(app_config, 'SETUP_THREAD_AFFINITY') and app_config.SETUP_THREAD_AFFINITY:
        setup_thread_affinity()

    # --- Pass 1 ---
    if app_config.PIPELINE_CONTROL['run_pass1_extraction']:
        print("\n--- Running Pass 1: Raw Data Extraction ---")
        if not os.path.exists(app_config.INPUT_VIDEO_PATH): # Assuming INPUT_VIDEO_PATH is used
            print(f"ERROR: Input video not found: {app_config.INPUT_VIDEO_PATH}", file=sys.stderr)
            sys.exit(1)
        try:
            extractor = RawDataExtractor(app_config) # Pass the config module
            result_message = extractor.extract_raw_data(
                video_path=app_config.INPUT_VIDEO_PATH,
                start_date_str=app_config.START_DATE,
                start_time_str=app_config.START_TIME
            )
            print("\n--- Pass 1 Result ---")
            print(result_message)
            print("---------------------")
        except Exception as e:
            handle_pipeline_error("Pass 1", e) # Create a helper for error handling
    else:
        print("Pass 1 (Raw Data Extraction) is disabled in config.")

    # --- Pass 2 ---
    if app_config.PIPELINE_CONTROL['run_pass2_path_analysis']:
        print("\n--- Running Pass 2: Initial Path Analysis & Fragment Identification ---")
        if not app_config.PIPELINE_CONTROL['run_pass1_extraction'] and \
           not os.path.exists(app_config.FILE_PATHS['pass1_raw_track_output']):
            print(f"ERROR: Pass 1 output not found ({app_config.FILE_PATHS['pass1_raw_track_output']}) and Pass 1 was not run. Cannot proceed with Pass 2.", file=sys.stderr)
            sys.exit(1)
        try:
            analyze_initial_paths(
                raw_tracks_parquet_path=app_config.FILE_PATHS['pass1_raw_track_output'],
                initial_paths_output_path=app_config.FILE_PATHS['pass2_initial_paths_output'],
                fragmented_tracks_output_path=app_config.FILE_PATHS['pass2_fragmented_tracks_output'],
                app_config=app_config # Pass the whole config module
            )
            print("--- Pass 2 Completed ---")
        except Exception as e:
            handle_pipeline_error("Pass 2", e)
    else:
        print("Pass 2 (Initial Path Analysis) is disabled in config.")

    # --- Pass 3 ---
    if app_config.PIPELINE_CONTROL['run_pass3_fragment_stitching']:
        print("\n--- Running Pass 3: Fragment Stitching with Re-ID ---")
        # Check dependencies
        pass1_output_exists = os.path.exists(app_config.FILE_PATHS['pass1_raw_track_output'])
        pass2_fragments_exist = os.path.exists(app_config.FILE_PATHS['pass2_fragmented_tracks_output'])

        if not (app_config.PIPELINE_CONTROL['run_pass1_extraction'] or pass1_output_exists):
            print(f"ERROR: Pass 1 output not found and Pass 1 was not run. Cannot proceed with Pass 3.", file=sys.stderr)
            sys.exit(1)
        if not (app_config.PIPELINE_CONTROL['run_pass2_path_analysis'] or pass2_fragments_exist):
            print(f"ERROR: Pass 2 fragmented tracks output not found and Pass 2 was not run. Cannot proceed with Pass 3.", file=sys.stderr)
            sys.exit(1)
            
        try:
            stitch_fragments(
                fragmented_tracks_parquet_path=app_config.FILE_PATHS['pass2_fragmented_tracks_output'],
                raw_tracks_parquet_path=app_config.FILE_PATHS['pass1_raw_track_output'],
                stitched_paths_output_path=app_config.FILE_PATHS['pass3_stitched_paths_output'],
                app_config=app_config
            )
            print("--- Pass 3 Completed ---")
        except Exception as e:
            handle_pipeline_error("Pass 3", e)
    else:
        print("Pass 3 (Fragment Stitching) is disabled in config.")
    
    # --- Pass 4 (Placeholder) ---
    if app_config.PIPELINE_CONTROL['run_pass4_final_report']:
        print("\n--- Skipping Pass 4 (Final Report) - Not yet implemented ---")
        # Check dependencies: Pass 2 complete paths + Pass 3 stitched paths

def handle_pipeline_error(pass_name, error):
    print(f"\n--- ERROR during {pass_name} ---", file=sys.stderr)
    print(f"Error: {error}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    run_full_pipeline()