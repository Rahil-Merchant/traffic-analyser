import os
import sys
import traceback
import threading
import queue
from datetime import datetime

from core.deepstream_pipeline import DeepStreamTrafficPipeline
from core.utils import setup_torch_global_settings, setup_thread_affinity
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")

def run_deepstream_local_processing():
    print("--- Starting DeepStream Interactive Mode (Local Debug) ---")
    setup_torch_global_settings()
    setup_thread_affinity()

    # --- Hardcoded test configuration (edit as needed) ---
    video_paths = [
        "sample_videos/5 Minute Test - 2 Way ii.mp4"
    ]
    start_date = "250728"   # YYMMDD
    start_time = "050000000"  # HHMMSSmmm
    primary_direction = "south"
    station_number = ""
    generate_bidirectional_report = False

    # Example interactive polygons: (replace as needed)
    polygons = {
        'north': [(282, 339), (422, 312), (371, 240), (272, 264)],
        'south': [(290, 446), (605, 335), (619, 441), (470, 466), (361, 476)],
        'east': [(512, 306), (596, 322), (628, 214), (525, 175)],
        'west': [(252, 378), (244, 442), (57, 466), (53, 397)]
    }

    processor = DeepStreamTrafficPipeline(line_mode="interactive")
    processor.set_zone_polygons(polygons)

    result_queue = queue.Queue()

    def target():
        try:
            msg = processor.process_video(
                video_paths=video_paths,
                start_date_str=start_date,
                start_time_str=start_time,
                primary_direction=primary_direction,
                station_number_str=station_number,
                generate_link_count_report_flag=generate_bidirectional_report,
                progress_queue=None,
                preview_image_width=640,
                preview_image_height=480
            )
            result_queue.put(msg)
        except Exception as e:
            traceback.print_exc()
            result_queue.put(e)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    result = result_queue.get()
    if isinstance(result, Exception):
        logger.error(f"Processing failed with exception: {result}")
    else:
        print("\n--- Processing Complete ---")
        print(result)
        print("---------------------------")

if __name__ == "__main__":
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    run_deepstream_local_processing()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(30)
