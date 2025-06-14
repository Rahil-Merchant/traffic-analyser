import os
import uvicorn
import cv2
from typing import List, Dict
from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

from config import SERVER_PORT
from core.utils import setup_torch_global_settings, setup_thread_affinity
from core.video_processor import VideoProcessor
import signal
import atexit

print("--- Initializing EMAAN DEVELOPMENT Server (File System Access API) ---")

app = FastAPI(title="EMAAN API", version="5.0.0")
processor = VideoProcessor('interactive')
job_queues: Dict[str, asyncio.Queue] = {}
video_dimensions_cache = {}  # Store dimensions by video path


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
        processor.stop_event.set()
    
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
    primary_direction: str
    report_type: str
    polygons: dict
    preview_image_width: int = 640
    preview_image_height: int = 480

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
        video_dimensions_cache[video_path] = {
            'width': frame_width,
            'height': frame_height
        }

        success, frame = cap.read()
        cap.release()
        if not success:
            raise HTTPException(status_code=400, detail="Could not read the first frame.")
        
        preview_image_name = f"preview_{os.urandom(8).hex()}.jpg"
        preview_image_path = os.path.join(STATIC_DIR, preview_image_name)
        cv2.imwrite(preview_image_path, frame)
        
        full_image_url = f"http://127.0.0.1:{SERVER_PORT}/previews/{preview_image_name}"
        print(f"Video dimensions cached: {frame_width}x{frame_height}")
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
    
    # Get dimensions from cache for the first video
    first_video_path = config.video_paths[0]
    cached_dimensions = video_dimensions_cache.get(first_video_path)
    
    if cached_dimensions:
        preview_width = cached_dimensions['width']
        preview_height = cached_dimensions['height']
        print(f"✅ Using cached dimensions: {preview_width}x{preview_height}")
    else:
        print(f"⚠️ No cached dimensions for {first_video_path}, falling back to defaults")
        preview_width = 640
        preview_height = 480

    processor.zone_polygons = config.polygons
    
    background_tasks.add_task(
        processor.process_video,
        video_paths=config.video_paths,
        start_date_str=config.start_date,
        start_time_str=config.start_time,
        station_number_str=config.station_number,
        primary_direction=config.primary_direction.lower(),
        generate_link_count_report_flag=(config.report_type == "Standard + Bi-directional Report"),
        progress_queue=progress_queue,
        preview_image_width=preview_width,
        preview_image_height=preview_height
    )
    background_tasks.add_task(lambda: job_queues.pop(job_id, None))
    
    return {"status": "processing_started", "job_id": job_id}


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