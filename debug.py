import os
import sys
import traceback

# --- Project-specific Imports ---
from config import INPUT_VIDEO_PATH, START_DATE, START_TIME, DEFAULT_PRIMARY_DIRECTION
from core.utils import setup_torch_global_settings, setup_thread_affinity
from core.video_processor import VideoProcessor

def run_local_processing():
    """
    Runs the video processing in a non-interactive, "hardcoded" mode
    based on settings in config.py.
    """
    print("--- Starting EMAAN Urban Reconnaissance (Hardcoded Mode) ---")
    
    # Perform the same setup as the main app
    setup_torch_global_settings()
    setup_thread_affinity()

    # --- Validate required config values ---
    if not INPUT_VIDEO_PATH or not isinstance(INPUT_VIDEO_PATH, list) or not os.path.exists(INPUT_VIDEO_PATH[0]):
        print(f"ERROR: Input video file not found or invalid in config.py.", file=sys.stderr)
        print(f"Searched for: {INPUT_VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Using Input Video(s): {INPUT_VIDEO_PATH}")
    print(f"Using Start Date:  {START_DATE}")
    print(f"Using Start Time:  {START_TIME}")

    try:
        # Initialize the processor and run the analysis.
        # Note: This mode will use the hardcoded LINE_POINTS from config.py
        # because the processor's gradio_polygons will be empty.
        processor = VideoProcessor()
        result_message = processor.process_video(
            video_paths=INPUT_VIDEO_PATH,
            start_date_str=START_DATE,
            start_time_str=START_TIME,
            primary_direction=DEFAULT_PRIMARY_DIRECTION
        )
        print("\n--- Processing Complete ---")
        print(result_message)
        print("---------------------------")
        
    except Exception as e:
        print(f"\n--- FATAL ERROR during hardcoded processing ---", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # To run this mode, execute `python project/local.py` from your terminal.
    run_local_processing()