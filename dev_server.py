import os
import uvicorn
import cv2
from typing import List, Dict
from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from config import SERVER_PORT
from core.utils import setup_torch_global_settings, setup_thread_affinity, compute_speed_from_trajectory
from core.video_processor import VideoProcessor
import signal
import atexit
from datetime import datetime
from reporting.excel_report import create_excel_report, create_bidirectional_link_count_report
from typing import Optional
from typing import Optional

print("--- Initializing EMAAN DEVELOPMENT Server (File System Access API) ---")

app = FastAPI(title="EMAAN API", version="6.0.0")
processor = VideoProcessor('interactive')
job_queues: Dict[str, asyncio.Queue] = {}

# Global cleanup flag
cleanup_in_progress = False

def cleanup_on_exit():
    """Cleanup function called on exit"""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True
    
    print("\n--- EMAAN Backend Shutting Down ---")
    
    # Stop any active video processing
    if hasattr(processor, 'stop_event') and processor.stop_event:
        processor.abort()
    
    # Clean up any active threads
    if hasattr(processor, 'reader_thread') and processor.reader_thread:
        try:
            processor.reader_thread.join(timeout=2)
        except:
            pass
    
    if hasattr(processor, 'writer_thread') and processor.writer_thread:
        try:
            processor.writer_thread.join(timeout=2)
        except:
            pass
    
    # Clear job queues
    global job_queues
    job_queues.clear()
    
    print("--- Backend Cleanup Complete ---")

def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals"""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    cleanup_on_exit()
    exit(0)

# Register cleanup handlers
atexit.register(cleanup_on_exit)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@app.on_event("startup")
async def startup_event():
    """
    This function will be executed ONCE when the FastAPI application starts up.
    This is the CORRECT place to run initialization code.
    """
    print("--- Server is starting up. Running initial configurations... ---")
    setup_torch_global_settings()
    setup_thread_affinity()
    print("--- Configurations applied. Server is ready. ---")


app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SCRIPT_DIR, "output", "static_previews")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/previews", StaticFiles(directory=STATIC_DIR), name="previews")

# --- Define Pydantic Models for structured JSON data from frontend ---
class FilePathPayload(BaseModel):
    path: str

class AnalysisConfigPayload(BaseModel):
    video_paths: List[str]
    start_date: str
    start_time: str
    station_number: str
    primary_direction: str
    polygons: dict
    preview_image_width: int
    preview_image_height: int
    perspective_corners: Optional[list] = None
    road_width_meters: Optional[float] = None
    road_length_meters: Optional[float] = None
    length_skew_factor: Optional[float] = 1.0
    width_skew_factor: Optional[float] = 1.0
    speed_limit_kmh: Optional[float] = None
    night_hours_start: Optional[str] = None
    night_hours_end: Optional[str] = None

class RecalibrateConfigPayload(BaseModel):
    video_paths: List[str]
    start_date: str
    start_time: str
    station_number: str
    primary_direction: str
    polygons: dict
    preview_image_width: int
    preview_image_height: int
    perspective_corners: Optional[list] = None
    road_width_meters: Optional[float] = None
    road_length_meters: Optional[float] = None
    length_skew_factor: Optional[float] = 1.0
    width_skew_factor: Optional[float] = 1.0
    speed_limit_kmh: Optional[float] = None
    night_hours_start: Optional[str] = None
    night_hours_end: Optional[str] = None
    video_duration_formatted: str

# --- API Endpoints ---
@app.get("/api/v1/health", tags=["System"])
async def health_check(): return {"status": "ok"}

@app.post("/api/v1/video-preview", tags=["Setup"])
async def get_video_preview(payload: FilePathPayload):
    """
    Reads a video file directly from disk using the provided path.
    This works with real filesystem paths from Tauri.
    """
    video_path = payload.path
    try:
        print(f"[DEBUG] Attempting to read video from: {video_path}")
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="OpenCV could not open video file.")
        
        # Get actual video dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        success, frame = cap.read()
        cap.release()
        if not success:
            raise HTTPException(status_code=400, detail="Could not read the first frame.")
        
        preview_image_name = f"preview_{os.urandom(8).hex()}.jpg"
        preview_image_path = os.path.join(STATIC_DIR, preview_image_name)
        cv2.imwrite(preview_image_path, frame)
        
        full_image_url = f"http://127.0.0.1:{SERVER_PORT}/previews/{preview_image_name}"
        print(f"Video dimensions: {frame_width}x{frame_height}")
        print(f"[DEBUG] Preview generated successfully: {full_image_url}")
        
        return {"preview_image_url": full_image_url, 'video_width': frame_width, 'video_height': frame_height}
    except Exception as e:
        print(f"[ERROR] Preview generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing preview: {e}")

@app.post("/api/v1/process", tags=["Processing"])
async def run_analysis(background_tasks: BackgroundTasks, config: AnalysisConfigPayload):
    """Processes video files directly from disk using real filesystem paths."""
    print(f"Received new job. Processing videos: {config.video_paths}")

    # Validate all video files exist
    for video_path in config.video_paths:
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")

    job_id = f"job_{os.urandom(4).hex()}"
    progress_queue = asyncio.Queue()
    job_queues[job_id] = progress_queue

    processor.zone_polygons = config.polygons
    processor.perspective_corners_raw = config.perspective_corners
    processor.road_width_meters = config.road_width_meters
    processor.road_length_meters = config.road_length_meters
    processor.length_skew_factor = config.length_skew_factor
    processor.width_skew_factor = config.width_skew_factor
    processor.speed_limit_kmh = config.speed_limit_kmh
    processor.night_hours_start = config.night_hours_start
    processor.night_hours_end = config.night_hours_end
    
    background_tasks.add_task(
        processor.process_video,
        video_paths=config.video_paths,
        start_date_str=config.start_date,
        start_time_str=config.start_time,
        station_number_str=config.station_number,
        primary_direction=config.primary_direction.lower(),
        generate_link_count_report_flag=(len(config.polygons) == 2),
        progress_queue=progress_queue,
        preview_image_width=config.preview_image_width,
        preview_image_height=config.preview_image_height
    )
    background_tasks.add_task(lambda: job_queues.pop(job_id, None))
    
    return {"status": "processing_started", "job_id": job_id}


@app.post("/api/v1/recalibrate-speed", tags=["Processing"])
async def recalibrate_speed(config: RecalibrateConfigPayload):
    print("Recalibrating speeds with new parameters...")
    try:
        old_length = processor.road_length_meters
        new_length = config.road_length_meters
        old_skew = processor.length_skew_factor if processor.length_skew_factor else 1.0
        new_skew = config.length_skew_factor if config.length_skew_factor else 1.0
        
        old_eff_length = old_length * old_skew if old_length else None
        new_eff_length = new_length * new_skew if new_length else None
        
        scale_factor = (new_eff_length / old_eff_length) if (old_eff_length and old_eff_length > 0 and new_eff_length) else 1.0

        all_paths = processor.finalized_paths_store.get_all()
        for path in all_paths:
            old_speed = path.get('average_speed_kmh')
            if old_speed is not None:
                new_speed = round(old_speed * scale_factor, 1)
                processor.finalized_paths_store.update_speed(path['id'], new_speed)
        
        processor.road_length_meters = config.road_length_meters
        processor.road_width_meters = config.road_width_meters
        processor.length_skew_factor = config.length_skew_factor
        processor.width_skew_factor = config.width_skew_factor
        processor.speed_limit_kmh = config.speed_limit_kmh
        processor.night_hours_start = config.night_hours_start
        processor.night_hours_end = config.night_hours_end

        time_str = str(config.start_time)
        time_part, ms_part = (time_str[:6], time_str[6:]) if len(time_str) > 6 else (time_str, "0")
        start_dt = datetime.strptime(f"{config.start_date}{time_part}", "%y%m%d%H%M%S")
        start_dt = start_dt.replace(microsecond=int(ms_part.ljust(3, '0')) * 1000)

        calib_factors = {
            'Base Road Length (m)': processor.road_length_meters,
            'Base Road Width (m)': processor.road_width_meters,
            'Length Skew Factor': processor.length_skew_factor,
            'Width Skew Factor': processor.width_skew_factor
        }

        excel_path = create_excel_report(
            completed_paths_data=processor.finalized_paths_store.get_all(),
            start_datetime=start_dt,
            primary_direction=config.primary_direction.lower(),
            video_path=config.video_paths[0],
            report_type='standard',
            calibration_factors=calib_factors
        )
        
        excel_tia_path = create_excel_report(
            completed_paths_data=processor.finalized_paths_store.get_all(),
            start_datetime=start_dt,
            primary_direction=config.primary_direction.lower(),
            video_path=config.video_paths[0],
            report_type='tia',
            calibration_factors=calib_factors
        )

        excel_link_count_path = None
        if len(config.polygons) == 2:
            secondary_dirs = [d for d in config.polygons.keys() if d != config.primary_direction.lower()]
            if secondary_dirs:
                excel_link_count_path = create_bidirectional_link_count_report(
                    completed_paths_data=processor.finalized_paths_store.get_all(),
                    start_datetime_video=start_dt,
                    video_path=config.video_paths[0],
                    primary_direction=config.primary_direction.lower(),
                    secondary_direction=secondary_dirs[0],
                    video_duration_formatted=config.video_duration_formatted,
                    station_number_str=config.station_number,
                    speed_limit_kmh=config.speed_limit_kmh,
                    night_hours_start=config.night_hours_start,
                    night_hours_end=config.night_hours_end
                )

        updated_paths = processor.finalized_paths_store.get_all()
        speeds = [p['average_speed_kmh'] for p in updated_paths if p.get('average_speed_kmh') is not None]
        avg_speed = round(float(sum(speeds)) / len(speeds), 1) if speeds else None

        results_dict = {
            "excel_path": excel_path,
            "bidirectional_excel_path": excel_link_count_path,
            "tia_excel_path": excel_tia_path
        }
        stats_dict = {
            "fleet_average_speed_kmh": avg_speed
        }
        return {"status": "success", "results": results_dict, "statistics": stats_dict}
        
    except Exception as e:
        print(f"Recalibration Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/abort", tags=["Processing"])
async def abort_analysis():
    """Abort the currently running analysis by signaling the stop event."""
    print("[ABORT] Abort request received.")
    try:
        if hasattr(processor, 'stop_event') and processor.stop_event:
            processor.abort()
            # Send abort status through all active job queues
            for job_id, queue in list(job_queues.items()):
                try:
                    await queue.put({"status": "error", "message": "Analysis aborted by user.", "progress": 0})
                except Exception:
                    pass
            print("[ABORT] Stop event set. Processing will halt shortly.")
            return {"status": "aborted", "message": "Analysis abort signal sent."}
        else:
            return {"status": "no_active_job", "message": "No active analysis to abort."}
    except Exception as e:
        print(f"[ABORT ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
        

@app.websocket("/ws/progress/{job_id}")
async def websocket_progress_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    queue = job_queues.get(job_id)
    if not queue:
        await websocket.send_json({"error": f"Job ID '{job_id}' not found."}); await websocket.close(); return
    try:
        while True:
            data = await queue.get()
            await websocket.send_json(data)
            if data.get("status") in ["complete", "error", "finished"]: break
    except WebSocketDisconnect: print(f"Frontend for job '{job_id}' disconnected.")
    finally: job_queues.pop(job_id, None); print(f"WebSocket for job '{job_id}' closed.")

if __name__ == "__main__":
    uvicorn.run("dev_server:app", host="127.0.0.1", port=SERVER_PORT, reload=True)