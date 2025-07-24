import os
import sys
import traceback
import cProfile
import pstats

# --- Project-specific Imports ---
from config import INPUT_VIDEO_PATH, START_DATE, START_TIME, DEFAULT_PRIMARY_DIRECTION
from core.utils import setup_torch_global_settings, setup_thread_affinity
from core.video_processor import VideoProcessor
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="DEBUG")

def run_local_processing():
    print("--- Starting Hardcoded Mode ---")
    setup_torch_global_settings()
    setup_thread_affinity()

    print(f"Using Input Video(s): {INPUT_VIDEO_PATH}")
    print(f"Using Start Date:  {START_DATE}")
    print(f"Using Start Time:  {START_TIME}")

    try:
        processor = VideoProcessor('hardcoded')
        result_message = processor.process_video(
            video_paths=INPUT_VIDEO_PATH,
            start_date_str=START_DATE,
            start_time_str=START_TIME,
            primary_direction=DEFAULT_PRIMARY_DIRECTION,
            station_number_str='',
            generate_link_count_report_flag=False
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
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_local_processing()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(30)
