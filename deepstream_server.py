# project/core/deepstream_server.py

import os
import uvicorn
import cv2
from typing import List, Dict
from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from configs.deepstream_config import *
from core.utils import setup_torch_global_settings, setup_thread_affinity
from core.deepstream_pipeline import DeepStreamTrafficPipeline
import signal
import atexit

print("=== EMAAN DeepStream 7.1 Server ===")

app = FastAPI(title="EMAAN DeepStream API", version="7.1.0")

# Initialize DeepStream pipeline
processor = DeepStreamTrafficPipeline('interactive')

job_queues: Dict[str, asyncio.Queue] = {}
cleanup_in_progress = False

def cleanup_on_exit():
    """Cleanup function called on exit"""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True
    
    print("\n=== EMAAN DeepStream Server Shutting Down ===")
    
    # Clean up processor
    if processor:
        processor.cleanup()
    
    # Clear job queues
    global job_queues
    job_queues.clear()
    
    print("=== Server Cleanup Complete ===")

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
    """Initialize server components"""
    print("=== DeepStream Server Starting ===")
    setup_torch_global_settings()
    setup_thread_affinity()
    print("=== Server Ready ===")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SCRIPT_DIR, "output", "static_previews")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/previews", StaticFiles(directory=STATIC_DIR), name="previews")

# Pydantic models
class FilePathPayload(BaseModel):
    path: str

class AnalysisConfigPayload(BaseModel):
    video_paths: List[str]
    start_date: str
    start_time: str
    station_number: str
    primary_direction: str
    report_type: str
    polygons: dict
    preview_image_width: int
    preview_image_height: int

# API Endpoints
@app.get("/api/v1/health", tags=["System"])
async def health_check():
    return {"status": "ok", "version": "DeepStream 7.1"}

@app.post("/api/v1/video-preview", tags=["Setup"])
async def get_video_preview(payload: FilePathPayload):
    """Generate video preview using OpenCV (unchanged from original)"""
    video_path = payload.path
    
    try:
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="OpenCV could not open video file.")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        success, frame = cap.read()
        cap.release()
        
        if not success:
            raise HTTPException(status_code=400, detail="Could not read the first frame.")
        
        preview_image_name = f"preview_{os.urandom(8).hex()}.jpg"
        preview_image_path = os.path.join(STATIC_DIR, preview_image_name)
        cv2.imwrite(preview_image_path, frame)
        
        full_image_url = f"http://127.0.0.1:7862/previews/{preview_image_name}"
        
        return {
            "preview_image_url": full_image_url,
            'video_width': frame_width,
            'video_height': frame_height
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing preview: {e}")

@app.post("/api/v1/process", tags=["Processing"])
async def run_analysis(background_tasks: BackgroundTasks, config: AnalysisConfigPayload):
    """Process video files using DeepStream pipeline"""
    print(f"Received DeepStream job. Processing videos: {config.video_paths}")
    
    # Validate video files
    for video_path in config.video_paths:
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    job_id = f"deepstream_job_{os.urandom(4).hex()}"
    progress_queue = asyncio.Queue()
    job_queues[job_id] = progress_queue
    
    # Set zone polygons for interactive mode
    processor.set_zone_polygons(config.polygons)
    
    # Start processing
    background_tasks.add_task(
        processor.process_video,
        video_paths=config.video_paths,
        start_date_str=config.start_date,
        start_time_str=config.start_time,
        station_number_str=config.station_number,
        primary_direction=config.primary_direction.lower(),
        generate_link_count_report_flag=(config.report_type == "Standard + Bi-directional Report"),
        progress_queue=progress_queue,
        preview_image_width=config.preview_image_width,
        preview_image_height=config.preview_image_height
    )
    
    # Clean up job queue after completion
    background_tasks.add_task(lambda: job_queues.pop(job_id, None))
    
    return {"status": "deepstream_processing_started", "job_id": job_id}

@app.websocket("/ws/progress/{job_id}")
async def websocket_progress_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for progress updates"""
    await websocket.accept()
    queue = job_queues.get(job_id)
    
    if not queue:
        await websocket.send_json({"error": f"Job ID '{job_id}' not found."})
        await websocket.close()
        return
    
    try:
        while True:
            data = await queue.get()
            await websocket.send_json(data)
            if data.get("status") in ["complete", "error", "finished"]:
                break
    except WebSocketDisconnect:
        print(f"Frontend for DeepStream job '{job_id}' disconnected.")
    finally:
        job_queues.pop(job_id, None)
        print(f"WebSocket for DeepStream job '{job_id}' closed.")

if __name__ == "__main__":
    uvicorn.run("deepstream_server:app", host="0.0.0.0", port=7862, reload=False)


