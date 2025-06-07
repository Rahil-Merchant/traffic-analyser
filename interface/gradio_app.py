# project/interface/gradio_app.py
import gradio as gr
import os
import re
from config import (
    START_DATE, START_TIME, VIDEO_OUTPUT_DIR
)
from core.video_processor import VideoProcessor
from core.utils import debug_print

# Initialize the backend processor
processor = VideoProcessor()

# --- Gradio Event Handlers (Backend logic remains the same, no changes needed here) ---

def update_zone_dependent_controls():
    defined_zones = list(processor.gradio_polygons.keys())
    defined_zones_display = [z.capitalize() for z in defined_zones]
    primary_dir_update = gr.update(choices=defined_zones_display)
    if len(defined_zones) == 2:
        report_type_update = gr.update(interactive=True, value="Standard Report")
    else:
        report_type_update = gr.update(interactive=False, value="Standard Report")
    return primary_dir_update, report_type_update

def handle_upload(video_files_list):
    if not (video_files_list and hasattr(video_files_list[0], 'name')):
        primary_dir_update, report_type_update = update_zone_dependent_controls()
        return None, "Please upload a video to begin.", primary_dir_update, report_type_update
    video_path_for_preview = video_files_list[0].name
    frame, status = processor.process_video_upload_for_gradio(video_path_for_preview)
    primary_dir_update, report_type_update = update_zone_dependent_controls()
    return frame, status, primary_dir_update, report_type_update

def handle_set_direction(direction):
    frame_with_polys, status = processor.set_gradio_polygon_direction(direction)
    primary_dir_update, report_type_update = update_zone_dependent_controls()
    return frame_with_polys, status, primary_dir_update, report_type_update

def handle_click(evt: gr.SelectData):
    frame_with_polys, status = processor.handle_gradio_polygon_click(evt)
    return frame_with_polys, status

def handle_finish_polygon():
    frame_with_polys, status = processor.finalize_current_polygon()
    primary_dir_update, report_type_update = update_zone_dependent_controls()
    return frame_with_polys, status, primary_dir_update, report_type_update

def handle_undo_point():
    frame_with_polys, status = processor.undo_last_polygon_point()
    return frame_with_polys, status

def handle_clear_polygons():
    frame_with_polys, status = processor.reset_gradio_polygons()
    primary_dir_update, report_type_update = update_zone_dependent_controls()
    return frame_with_polys, status, primary_dir_update, report_type_update

def handle_process(video_files_list, start_date, start_time, primary_direction_str, report_type, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Initializing...")
    generate_bidirectional_report = (report_type == "Standard + Bi-directional Report")
    if not video_files_list:
        return "Please upload at least one video file.", None, "N/A", "N/A", "N/A", "N/A"
    if not primary_direction_str:
         return "Please define at least one zone and select a Primary Direction.", None, "N/A", "N/A", "N/A", "N/A"
    if generate_bidirectional_report and len(processor.gradio_polygons) != 2:
        return "Error: Bi-directional report was requested, but exactly 2 zones are not defined.", None, "N/A", "N/A", "N/A", "N/A"
    video_paths_from_gradio = [f.name for f in video_files_list if f and hasattr(f, 'name')]
    if not video_paths_from_gradio:
        return "No valid video files found.", None, "N/A", "N/A", "N/A", "N/A"
    try:
        progress(0.05, desc="Starting Video Processing Engine...")
        result_message_raw = processor.process_video(
            video_paths=video_paths_from_gradio, start_date_str=start_date, start_time_str=start_time,
            primary_direction=primary_direction_str.lower(), generate_link_count_report_flag=generate_bidirectional_report
        )
    except Exception as e:
         return f"❌ Error: {e}", None, "N/A", "N/A", "N/A", "N/A"
    frames_processed, time_seconds, fps, completed_paths = "N/A", "N/A", "N/A", "N/A"
    output_log = result_message_raw
    if "✅ Processing completed" in result_message_raw:
        progress(1.0, desc="Processing Complete!")
        match_frames = re.search(r"STAT_FRAMES_PROCESSED=(\d+)", result_message_raw)
        match_time = re.search(r"STAT_TIME_SECONDS=([\d\.]+)", result_message_raw)
        match_fps = re.search(r"STAT_FPS=([\d\.]+)", result_message_raw)
        match_paths = re.search(r"STAT_COMPLETED_PATHS=(\d+)", result_message_raw)
        if match_frames: frames_processed = match_frames.group(1)
        if match_time: time_seconds = match_time.group(1)
        if match_fps: fps = match_fps.group(1)
        if match_paths: completed_paths = match_paths.group(1)
        output_log = re.sub(r"--- Summary Stats ---.*", "", result_message_raw, flags=re.DOTALL).strip()
    output_video_path = None
    if "✅ Processing completed" in result_message_raw:
        try:
            files = [os.path.join(VIDEO_OUTPUT_DIR, f) for f in os.listdir(VIDEO_OUTPUT_DIR) if f.startswith("output_") and f.endswith(".mp4")]
            if files: output_video_path = max(files, key=os.path.getmtime)
        except Exception as e: print(f"Error finding output video: {e}")
    return output_log, output_video_path, frames_processed, time_seconds, fps, completed_paths


# --- Gradio UI Definition ---
def create_interface():
    print("Creating AURA Gradio interface...")
    
    # A premium, dark theme as the foundation
    theme = gr.themes.Monochrome(
        primary_hue="blue",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="*neutral_950",
        block_background_fill="*neutral_950",
        block_border_width="0px"
    )
    
    # Advanced CSS for the premium "glassmorphism" look and feel
    css = """
    /* --- Main Container & Overall Layout --- */
    .gradio-container { max-width: 1800px !important; margin: auto; padding-top: 20px; }
    footer { display: none !important; }

    /* --- Custom Title & Header --- */
    #aura_title {
        text-align: center; color: white; font-size: 3em; font-weight: 800; margin-bottom: 5px;
        letter-spacing: -2px;
    }
    #aura_subtitle {
        text-align: center; color: #888; font-size: 1.2em; font-weight: 300;
        margin-bottom: 40px;
    }

    /* --- Core UI Panels ("Glassmorphism") --- */
    .control-panel {
        background: rgba(30, 30, 45, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(12px) saturate(180%);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* --- Panel Headers --- */
    .panel-header { color: #fff; font-size: 1.4em; font-weight: 500; margin: 15px 0px 10px 5px; }
    .panel-subheader { color: #aaa; font-weight: 300; margin: 0px 0px 15px 5px; }

    /* --- Button Styling --- */
    .gr-button { border-radius: 8px !important; }
    #process_button {
        font-size: 1.2em !important; height: 60px !important;
        background: linear-gradient(45deg, #0d6efd, #0d6efd) !important;
        color: white !important; font-weight: bold !important;
        box-shadow: 0 0 15px #0d6efd, inset 0 0 5px rgba(255,255,255,0.4) !important;
    }

    /* --- Status Text Box --- */
    #status_textbox > label > span { display: none !important; }
    #status_textbox > label > textarea {
        background: rgba(0,0,0,0.3) !important; border-color: rgba(255,255,255,0.1) !important;
        color: #0dcaf0 !important; font-family: 'monospace' !important; font-size: 1.1em !important;
    }

    /* --- Make Accordion labels white --- */
    .gr-accordion > .gr-block > .label-wrap { color: #fff !important; }
    """

    with gr.Blocks(title="AURA Intersection Analyzer", theme=theme, css=css) as interface:
        gr.Markdown("# AURA", elem_id="aura_title")
        gr.Markdown("Advanced Urban Reconnaissance & Analysis", elem_id="aura_subtitle")

        with gr.Row(equal_height=False, variant="panel"):
            # --- COLUMN 1: SETUP (Left-most) ---
            with gr.Column(scale=3, min_width=380):
                with gr.Group(elem_classes=["control-panel"]):
                    gr.Markdown("## 📂 Project Setup", elem_classes=["panel-header"])
                    video_input = gr.File(label="Upload Video File(s)", file_types=['video', '.dav'], file_count="multiple")
                    with gr.Accordion("Timestamp Configuration", open=True):
                        gr.Markdown("Enter the date and time for the **start** of the first video.")
                        with gr.Row():
                            start_date = gr.Textbox(label="Start Date", value=START_DATE, placeholder="YYMMDD")
                            start_time = gr.Textbox(label="Start Time", value=START_TIME, placeholder="HHMMSSmmm")
                    
                    with gr.Accordion("Execution & Results", open=True):
                        completion_text = gr.Textbox(label="📋 Processing Log", lines=8, interactive=False, show_copy_button=True)
                        video_output = gr.Video(label="🎬 Processed Video", interactive=False, height=250)
                        with gr.Accordion("📊 Summary Statistics", open=False):
                            with gr.Row():
                                stat_frames = gr.Textbox(label="Frames Processed", interactive=False, text_align="center")
                                stat_time = gr.Textbox(label="Time (s)", interactive=False, text_align="center")
                            with gr.Row():
                                stat_fps = gr.Textbox(label="Overall FPS", interactive=False, text_align="center")
                                stat_completed = gr.Textbox(label="Valid Paths", interactive=False, text_align="center")


            # --- COLUMN 2: INTERACTIVE CANVAS (Middle) ---
            with gr.Column(scale=8, min_width=800):
                image_display = gr.Image(
                    label="Interactive Canvas", type="numpy", interactive=True,
                    height=800, show_label=True
                )
                status_text_draw = gr.Textbox(
                    value="[STATUS] Please upload a video to begin analysis.",
                    interactive=False, lines=1, elem_id="status_textbox"
                )

            # --- COLUMN 3: CONTROLS (Right-most) ---
            with gr.Column(scale=3, min_width=380):
                with gr.Group(elem_classes=["control-panel"]):
                    gr.Markdown("## 🕹️ Controls", elem_classes=["panel-header"])
                    gr.Markdown("Define zones and configure reports here.", elem_classes=["panel-subheader"])
                    
                    with gr.Group():
                        gr.Markdown("**Zone Definition**")
                        with gr.Row():
                            north_btn = gr.Button("⬆️ North", variant="secondary")
                            south_btn = gr.Button("⬇️ South", variant="secondary")
                            east_btn = gr.Button("➡️ East", variant="secondary")
                            west_btn = gr.Button("⬅️ West", variant="secondary")
                        with gr.Row():
                            finish_poly_btn = gr.Button("✅ Finish Zone", variant="primary")
                            undo_point_btn = gr.Button("↩️ Undo Point")
                            clear_all_btn = gr.Button("❌ Clear All", variant="stop")

                    with gr.Group():
                        gr.Markdown("**Report Configuration**")
                        primary_direction_input = gr.Dropdown(
                            label="Primary Direction", choices=[], interactive=True,
                            info="Select main approach from your defined zones."
                        )
                        report_type_radio = gr.Radio(
                            label="Report Type", choices=["Standard Report", "Standard + Bi-directional Report"],
                            value="Standard Report", interactive=False,
                            info="Bi-directional option requires exactly 2 zones."
                        )

                    process_btn = gr.Button("🚀 LAUNCH ANALYSIS", variant="primary", elem_id="process_button")

        # --- Event Listeners ---
        drawing_outputs = [image_display, status_text_draw]
        zone_dependent_outputs = [primary_direction_input, report_type_radio]
        
        video_input.upload(fn=handle_upload, inputs=video_input, outputs=drawing_outputs + zone_dependent_outputs, show_progress="full")
        
        btn_outputs = drawing_outputs + zone_dependent_outputs
        north_btn.click(lambda: handle_set_direction("north"), outputs=btn_outputs, show_progress="minimal")
        south_btn.click(lambda: handle_set_direction("south"), outputs=btn_outputs, show_progress="minimal")
        east_btn.click(lambda: handle_set_direction("east"), outputs=btn_outputs, show_progress="minimal")
        west_btn.click(lambda: handle_set_direction("west"), outputs=btn_outputs, show_progress="minimal")
        finish_poly_btn.click(fn=handle_finish_polygon, outputs=btn_outputs, show_progress="minimal")
        clear_all_btn.click(fn=handle_clear_polygons, outputs=btn_outputs, show_progress="minimal")

        image_display.select(fn=handle_click, outputs=drawing_outputs, show_progress="minimal")
        undo_point_btn.click(fn=handle_undo_point, outputs=drawing_outputs, show_progress="minimal")

        process_outputs = [completion_text, video_output, stat_frames, stat_time, stat_fps, stat_completed]
        process_btn.click(
            fn=handle_process,
            inputs=[video_input, start_date, start_time, primary_direction_input, report_type_radio],
            outputs=process_outputs
        )

    return interface