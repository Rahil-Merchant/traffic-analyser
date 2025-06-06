# project/interface/gradio_app.py
# This file is only loaded and used if LINE_MODE == 'interactive'
import gradio as gr
import os
import re
from config import (
    START_DATE, START_TIME, VIDEO_OUTPUT_DIR
)
from core.video_processor import VideoProcessor
from core.utils import debug_print

processor = VideoProcessor() # Assumes VideoProcessor is updated for polygons

# --- Gradio Event Handlers ---

# Handle video upload - gets first frame for drawing
def handle_upload(video_files_list):
    debug_print(f"Gradio: Handling video upload for preview/drawing: {video_files_list}")
    video_path_for_preview = None
    if video_files_list and len(video_files_list) > 0:
        first_file_obj = video_files_list[0]
        # Check if the first element is a valid file object with a .name
        if first_file_obj and hasattr(first_file_obj, 'name'):
            video_path_for_preview = first_file_obj.name
            debug_print(f"Gradio handle_upload: Using preview path: {video_path_for_preview}")
        else:
            debug_print(f"Gradio handle_upload: First file object is invalid or has no name: {first_file_obj}")
            return None, "Error: First uploaded file is invalid for preview."
    else:
        debug_print("Gradio handle_upload: No files provided in video_files_list for preview.")
        # Return current state if no video is uploaded, or an appropriate message
        return processor.last_drawn_polygons_frame, "Please upload a video to define zones."

    # Processor method should now reset polygon state and prepare frame
    return processor.process_video_upload_for_gradio(video_path_for_preview)

# Set the direction for drawing the *next* polygon
def handle_set_direction(direction):
    debug_print(f"Gradio: Setting direction for polygon: {direction}")
    # Processor method should update current_gradio_direction and clear temp points
    frame_with_polys, status = processor.set_gradio_polygon_direction(direction)
    return frame_with_polys, status

# Handle clicks on the image to add polygon vertices
def handle_click(evt: gr.SelectData):
    debug_print(f"Gradio: Handling click event at index: {evt.index} for polygon vertex")
    # Processor method should add point to temp points and redraw the frame
    frame_with_polys, status = processor.handle_gradio_polygon_click(evt)
    return frame_with_polys, status

# Finalize the currently drawn polygon points
def handle_finish_polygon():
    debug_print(f"Gradio: Finishing current polygon")
    # Processor method should validate temp points, add to main polygon dict, clear temp points, redraw
    frame_with_polys, status = processor.finalize_current_polygon()
    return frame_with_polys, status

# Undo the last point added
def handle_undo_point():
    debug_print(f"Gradio: Undoing last point")
    # Call the method in VideoProcessor to handle the undo logic
    frame_with_polys, status = processor.undo_last_polygon_point()
    return frame_with_polys, status

# Clear all drawn polygons (optional reset)
def handle_clear_polygons():
    debug_print(f"Gradio: Clearing all polygons")
    frame_with_polys, status = processor.reset_gradio_polygons()
    return frame_with_polys, status

# Process the video (uses finalized polygons)
def handle_process(video_files_list, start_date, start_time, primary_direction_str, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Initializing...") 
    debug_print(f"Gradio handle_process: Received video object: {video_files_list}")
    debug_print(f"Gradio handle_process: Primary Direction selected: {primary_direction_str}")

    if not video_files_list:
        return "Please upload at least one video file.", None, "N/A", "N/A", "N/A", "N/A"
    
    if not primary_direction_str:
         return "Please select a Primary Direction.", None, "N/A", "N/A", "N/A", "N/A"

    video_paths_from_gradio = []
    for video_file_obj_item in video_files_list: # Iterate through the list
        if video_file_obj_item is not None and hasattr(video_file_obj_item, 'name'):
            video_paths_from_gradio.append(video_file_obj_item.name)
        else:
            debug_print(f"Gradio handle_process: Skipped an invalid item in video_files_list: {video_file_obj_item}")

    # MODIFIED: Check if any valid paths were extracted
    if not video_paths_from_gradio:
        return "No valid video files found in the upload. Please ensure files are uploaded correctly.", None, "N/A", "N/A", "N/A", "N/A"

    debug_print(f"Gradio handle_process: Video paths from Gradio objects: {video_paths_from_gradio}")

    result_message_raw = ""

    # Run the processing - processor.process_video uses self.gradio_polygons
    # Add a try-except block here to catch potential errors during processing setup (like zone validation)
    # and report them back to the Gradio UI gracefully.
    
    try:
        progress(0.05, desc="Starting video processing engine...")
        result_message_raw = processor.process_video(video_paths_from_gradio, start_date, start_time, primary_direction=primary_direction_str.lower())
        debug_print(f"Gradio: Processing finished.")
    except RuntimeError as rte: 
         # Catch specific errors VideoProcessor might raise (e.g., for DAV/FFmpeg issues)
         print(f"Gradio handle_process: RuntimeError during processing: {rte}")
         progress(0, desc="Error!") 
         result_message_raw = f"❌ Error: {rte}" 
    except ValueError as ve:
         # Catch specific validation errors (like not enough zones)
         print(f"Gradio: Validation Error during processing setup: {ve}")
         progress(0, desc="Error!") # Update progress bar on error
         return f"Error: {ve}", None, "N/A", "N/A", "N/A", "N/A"
    except Exception as e:
         # Catch any other unexpected errors during processing
         print(f"Gradio: Unexpected Error during processing: {e}")
         import traceback; traceback.print_exc()
         progress(0, desc="Error!")
         return f"Unexpected Error: {e}", None, "N/A", "N/A", "N/A", "N/A"


    # --- Stat Parsing ---
    frames_processed = "N/A"; time_seconds = "N/A"; fps = "N/A"; completed_paths = "N/A"
    output_log = result_message_raw
    if "✅ Processing completed" in result_message_raw:
        progress(1.0, desc="Processing Complete!") # Progress bar should complete here
        match_frames = re.search(r"STAT_FRAMES_PROCESSED=(\d+)", result_message_raw)
        match_time = re.search(r"STAT_TIME_SECONDS=([\d\.]+)", result_message_raw)
        match_fps = re.search(r"STAT_FPS=([\d\.]+)", result_message_raw)
        match_paths = re.search(r"STAT_COMPLETED_PATHS=(\d+)", result_message_raw)
        if match_frames: frames_processed = match_frames.group(1)
        if match_time: time_seconds = match_time.group(1)
        if match_fps: fps = match_fps.group(1)
        if match_paths: completed_paths = match_paths.group(1)
        # Clean up the log message for display
        output_log = re.sub(r"--- Summary Stats ---.*", "", result_message_raw, flags=re.DOTALL).strip()
        output_log = re.sub(r"STAT_.*", "", output_log, flags=re.MULTILINE).strip() # Remove individual STAT lines too
    elif isinstance(result_message_raw, str) and (result_message_raw.startswith("❌ Error:") or "Error:" in result_message_raw):
         output_log = result_message_raw # Already an error message
         if progress.total is None or progress.current < progress.total:
            try:
                current_progress_val = 0.0 
                if hasattr(progress, 'value'): current_progress_val = progress.value # Older Gradio
                elif hasattr(progress, 'index') and hasattr(progress, 'max_index'): # Newer structure
                     if progress.max_index is not None and progress.max_index > 0 :
                         current_progress_val = (progress.index or 0) / progress.max_index
                     elif progress.index is not None : current_progress_val = progress.index # If used as simple counter

                if current_progress_val < 1.0 : # Only update if not already at 100% (e.g. from success path)
                    progress(0, desc="Error during processing!")
            except Exception as e_prog:
                debug_print(f"Gradio: Could not update progress bar on error: {e_prog}")
    else: # Should not happen if VideoProcessor always returns one of the above formats
        output_log = f"Processing finished, but result format unexpected: {str(result_message_raw)[:300]}"
        progress(1.0, desc="Finished (Unknown Result State)")


    # --- Find Output Video ---
    output_video_path = None
    if "✅ Processing completed" in result_message_raw:
        try:
            output_dir = VIDEO_OUTPUT_DIR
            files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("output_") and (f.endswith(".mp4") or f.endswith(".avi"))]
            if files: output_video_path = max(files, key=os.path.getmtime)
        except Exception as e: print(f"Error finding output video: {e}")

    # Return values for all output components
    return output_log, output_video_path, frames_processed, time_seconds, fps, completed_paths


# --- Gradio UI Definition ---
def create_interface():
    print("Creating Gradio interface (Polygon Mode)...")

    # Use a slightly more vibrant theme
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

    # CSS for minor tweaks (optional)
    css = """
    .gradio-container { max-width: 1280px !important; margin: auto; }
    /* Add other CSS tweaks if desired */
    footer { display: none !important; } /* Hide default Gradio footer */
    """

    with gr.Blocks(title="Emaan Traffic Vehicle Detector", theme=theme, css=css) as interface:
        # --- Header ---
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="font-size: 2.5em; font-weight: 600; color: #10b981;">🚦 Emaan Traffic Vehicle Detector</h1>
                <p style="font-size: 1.1em; color: #4b5563;">Upload a video, configure settings, and process to count vehicles.</p>
            </div>
            """
        )

        # --- Main Layout with Tabs ---
        with gr.Tabs() as tabs:
            # --- Tab 1: Setup ---
            with gr.TabItem("1️⃣ Setup", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Video Source")
                        video_input = gr.File(
                            label="Video File (.mp4, .avi, .mkv, .mov, .dav etc.)",
                            file_types=['video', '.dav'], # Suggest accepted types,
                            file_count="multiple",
                            height = 360
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### Set Initial Timestamp")
                        gr.Markdown("Enter the date and time corresponding to the *start* of the video.")
                        with gr.Group(): # Group timestamp inputs
                             with gr.Row():
                                 start_date = gr.Textbox(
                                     label="Start Date", value=START_DATE, placeholder="YYMMDD"
                                 )
                                 start_time = gr.Textbox(
                                     label="Start Time", value=START_TIME, placeholder="HHMMSSmmm"
                                 )
                        primary_direction_input = gr.Dropdown(
                            label="Select Primary Direction",
                            choices=["North", "South", "East", "West"],
                            value="South", # Default value
                            info="Select the main approach direction for numbering."
                        )
                        gr.Markdown("---") # Separator
                        status_text_draw = gr.Textbox(
                            label="✏️ Zone Drawing Helper",
                            value="Upload video and go to 'Draw Zones' tab.",
                            interactive=False,
                            lines=3, # Keep slightly larger
                            visible=True
                        )
                        gr.Markdown(
                            "➡️ Proceed to **Draw Zones** tab (if interactive) or **Process** tab.",
                            visible=True
                        )


            # --- Tab 2: Draw Zones (Conditional) ---
            with gr.TabItem("✍️ Draw Zones", id=1, visible=True):
                 gr.Markdown("### Define Detection Zones (Polygons)")
                 # Updated Instructions
                 gr.Markdown(
                     "Select a direction, then **click the desired vertices** (minimum 3) on the image below to define the polygon zone. "
                     "You **do not** need to click the starting point again to close it. "
                     "Use 'Undo Last Point' to correct mistakes. Click **'Finish Current Polygon'** when all vertices for the current zone are placed."
                 )
                 with gr.Row(equal_height=False):
                     with gr.Column(scale=1, min_width=150): # Controls column
                         gr.Markdown("**Select Direction:**")
                         north_btn = gr.Button("⬆️ North Zone", variant="secondary")
                         south_btn = gr.Button("⬇️ South Zone", variant="secondary")
                         east_btn = gr.Button("➡️ East Zone", variant="secondary")
                         west_btn = gr.Button("⬅️ West Zone", variant="secondary")
                         gr.Markdown("---")
                         finish_poly_btn = gr.Button("✅ Finish Current Polygon", variant="primary")
                         undo_point_btn = gr.Button("↩️ Undo Last Point") # Added button
                         clear_all_btn = gr.Button("❌ Clear All Polygons", variant="stop")
                         # Reuse the status text from Setup tab for feedback
                     with gr.Column(scale=3): # Image column
                         image_display = gr.Image(
                             label="Click Image to Define Polygon Vertices",
                             type="numpy",
                             interactive=True, # Enable clicks
                             height=640 ,# Make drawing area prominent,
                             width=800
                         )


            # --- Tab 3: Process & Results ---
            with gr.TabItem("🚀 Process & View Results", id=2):
                gr.Markdown("### Start Processing")
                with gr.Row():
                    process_btn = gr.Button("📊 Process Video Now", variant="primary", scale=1)
                gr.Markdown("---")
                gr.Markdown("### Processing Output")
                with gr.Accordion("📊 Summary Statistics", open=False): # Collapsible stats
                     with gr.Row(equal_height=True):
                         stat_frames = gr.Textbox(label="Frames Processed", value="N/A", interactive=False, text_align="center")
                         stat_time = gr.Textbox(label="Processing Time (s)", value="N/A", interactive=False, text_align="center")
                         stat_fps = gr.Textbox(label="Overall FPS", value="N/A", interactive=False, text_align="center")
                         stat_completed = gr.Textbox(label="Valid Paths", value="N/A", interactive=False, text_align="center")

                with gr.Group(): # Group log and video output
                    completion_text = gr.Textbox(label="📋 Processing Log", lines=6, interactive=False, show_copy_button=True)
                    gr.Markdown("") # Spacer
                    video_output = gr.Video(label="🎬 Processed Video", interactive=False, height=480)

        # --- Event Listeners (Connecting UI to Handlers) ---
        drawing_outputs = [image_display, status_text_draw]
        video_input.change( # Or .upload if you prefer that trigger
            fn=handle_upload,
            inputs=[video_input], # Pass the gr.File object
            outputs=drawing_outputs, 
            show_progress="minimal"
        )
        
        # Direction Button Actions: Update image display and drawing status text
        north_btn.click(lambda: handle_set_direction("north"), outputs=drawing_outputs, show_progress="minimal")
        south_btn.click(lambda: handle_set_direction("south"), outputs=drawing_outputs, show_progress="minimal")
        east_btn.click(lambda: handle_set_direction("east"), outputs=drawing_outputs, show_progress="minimal")
        west_btn.click(lambda: handle_set_direction("west"), outputs=drawing_outputs, show_progress="minimal")

        # Image Click Action: Updates image display and drawing status text
        image_display.select(
            fn=handle_click,
            outputs=drawing_outputs,
            show_progress="minimal"
        )

        # Finish Polygon Button: Finalize points and update display/status
        finish_poly_btn.click(fn=handle_finish_polygon, outputs=drawing_outputs, show_progress="minimal")

        # Undo Point Button: Remove last point and update display/status
        undo_point_btn.click(fn=handle_undo_point, outputs=drawing_outputs, show_progress="minimal")

        # Clear All Button: Reset polygons and update display/status
        clear_all_btn.click(fn=handle_clear_polygons, outputs=drawing_outputs, show_progress="minimal")


        # Process Button Action: Runs processing and updates all result components
        process_outputs = [
            completion_text, video_output,
            stat_frames, stat_time, stat_fps, stat_completed
        ]
        # Assign the process_btn click handler
        process_btn.click(
            fn=handle_process,
            inputs=[video_input, start_date, start_time, primary_direction_input],
            outputs=process_outputs # All result components
        )
        
        # process_btn.click(
        #     fn=processor.run_processing_on_all_videos,
        #     inputs=[ start_date, start_time, primary_direction_input],
        #     outputs=process_outputs # All result components
        # )
    print("Gradio interface created (Polygon Mode).")
    return interface
