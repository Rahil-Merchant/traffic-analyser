import sys
import os
from datetime import datetime, timedelta
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import yaml

# --- Import Your Existing Logic ---
# This allows the script to find your other modules like 'tracking'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from tracking.zone_tracker import ZoneTracker
from config import VEHICLE_TIMEOUTS, DATA_YAML_PATH, FRAME_WIDTH, FRAME_HEIGHT, LINE_POINTS
from excel_report import create_excel_report # We'll call this at the end

# --- Global State Management ---
# Replaces class members (e.g., self.active_track_data) from your VideoProcessor
g_active_track_data = {}
g_finalized_paths = []
g_zone_tracker = None
g_class_names = []
g_start_datetime = datetime.now()
g_video_path = ""
g_primary_direction = "south" # Can be passed as an argument later

def load_class_names(yaml_path):
    """Loads class names from your .yaml file."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])
    except Exception as e:
        print(f"Error loading class names from {yaml_path}: {e}")
        return []

def probe_logic_callback(pad, info, u_data):
    """
    This is the heart of your custom logic. It's called for every batch of
    frames after tracking is complete.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        # Use the frame number to calculate a more accurate timestamp
        current_frame_time_sec = frame_meta.buf_pts / 1e9
        current_global_timestamp = g_start_datetime + timedelta(seconds=current_frame_time_sec)
        
        # --- Process current objects in the frame ---
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            # Use the object_id from the tracker
            track_id = str(obj_meta.object_id)
            vehicle_type = g_class_names[obj_meta.class_id]
            
            # Bounding box coordinates are relative to the streammux output resolution
            rect = obj_meta.rect_params
            bottom_center = (rect.left + rect.width / 2, rect.top + rect.height)

            if track_id not in g_active_track_data:
                # --- New Vehicle Detected: Check for ENTRY ---
                for zone_name, zone_poly in g_zone_tracker.zones.items():
                    if g_zone_tracker._check_point_in_zone(bottom_center, zone_poly):
                        g_active_track_data[track_id] = {
                            'id': track_id, 'entry_direction': zone_name,
                            'entry_time': current_global_timestamp,
                            'type_history': {vehicle_type: 1}, 'current_type': vehicle_type,
                            'last_pos': bottom_center, 'status': 'active'
                        }
                        # Add text to display on screen (optional, but useful for debugging)
                        obj_meta.text_params.display_text = f"ID:{track_id} En:{zone_name}"
                        break
            else:
                # --- Existing Vehicle: Check for EXIT or RE-ENTRY ---
                track_data = g_active_track_data[track_id]
                track_data['last_pos'] = bottom_center
                track_data['type_history'][vehicle_type] = track_data['type_history'].get(vehicle_type, 0) + 1
                track_data['current_type'] = max(track_data['type_history'], key=track_data['type_history'].get)
                
                # check_zone_transition logic needs to be adapted slightly
                # For now, we'll assume a simple exit check
                event_type, event_dir = g_zone_tracker.check_zone_transition(
                    track_data['last_pos'], (rect.left, rect.top, rect.left + rect.width, rect.top + rect.height), 
                    track_id, current_frame_time_sec
                )
                
                if event_type == "EXIT":
                    duration = (current_global_timestamp - track_data['entry_time']).total_seconds()
                    g_finalized_paths.append({
                        'id': track_id, 'type': track_data['current_type'],
                        'entry_direction': track_data['entry_direction'], 'exit_direction': event_dir,
                        'entry_time': track_data['entry_time'], 'exit_time': current_global_timestamp,
                        'time_in_intersection': duration, 'status': 'exited'
                    })
                    # Remove from active tracks
                    del g_active_track_data[track_id]
                
                # Update display text
                obj_meta.text_params.display_text = f"ID:{track_id} T:{track_data['current_type']}"

            l_obj = l_obj.next
        
        # --- Handle Timeouts (replaces cleanup.py logic) ---
        # Do this after processing frame objects to get the latest timestamp
        for track_id, data in list(g_active_track_data.items()):
            duration = (current_global_timestamp - data['entry_time']).total_seconds()
            timeout_limit = VEHICLE_TIMEOUTS.get(data['current_type'], 400)
            if duration > timeout_limit:
                g_finalized_paths.append({
                    'id': track_id, 'type': data['current_type'],
                    'entry_direction': data['entry_direction'], 'exit_direction': 'TIMEOUT',
                    'entry_time': data['entry_time'], 'exit_time': current_global_timestamp,
                    'time_in_intersection': duration, 'status': 'timed_out'
                })
                del g_active_track_data[track_id]

        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK

def main(args):
    if len(args) != 2:
        sys.stderr.write("Usage: %s <H264 video file>\n" % args[0])
        sys.exit(1)

    global g_zone_tracker, g_class_names, g_video_path
    g_video_path = args[1]
    g_class_names = load_class_names(DATA_YAML_PATH)
    
    # IMPORTANT: The polygons must be scaled to the pipeline's processing resolution (FRAME_WIDTH x FRAME_HEIGHT)
    # Your config.py already does this, so we can reuse it.
    g_zone_tracker = ZoneTracker(LINE_POINTS)

    Gst.init(None)
    pipeline = Gst.Pipeline.new("traffic-analysis-pipeline")

    # --- Create Pipeline Elements ---
    source = Gst.ElementFactory.make("filesrc", "file-source")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference-engine")
    tracker = Gst.ElementFactory.make("nvtracker", "object-tracker")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "video-converter")
    osd = Gst.ElementFactory.make("nvdsosd", "on-screen-display")
    # To save the output video with overlays
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
    codeparser = Gst.ElementFactory.make("h264parse", "h264-parser-for-encoding")
    muxer = Gst.ElementFactory.make("mp4mux", "mp4-muxer")
    sink = Gst.ElementFactory.make("filesink", "file-sink")

    # --- Configure Elements ---
    source.set_property('location', g_video_path)
    streammux.set_property('width', FRAME_WIDTH)
    streammux.set_property('height', FRAME_HEIGHT)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)
    pgie.set_property('config-file-path', "configs/ds_config_triton.txt")
    tracker.set_property('ll-config-file', "configs/tracker_config_nvdcf.yml")
    tracker.set_property('ll-lib-file', "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvdeepstream_tracker.so")
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    sink.set_property('location', './output_deepstream.mp4')
    sink.set_property('sync', 0)

    # --- Add All Elements to the Pipeline ---
    for element in [source, h264parser, decoder, streammux, pgie, tracker, nvvidconv, osd, encoder, codeparser, muxer, sink]:
        pipeline.add(element)

    # --- Link Elements Together ---
    source.link(h264parser)
    h264parser.link(decoder)
    
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = decoder.get_static_pad("src")
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(osd)
    osd.link(encoder)
    encoder.link(codeparser)
    codeparser.link(muxer)
    muxer.link(sink)

    # --- Add Probe for Custom Logic ---
    # We add the probe on the SINK pad of the OSD element. This means we get the
    # buffer AFTER the tracker has added its metadata to the objects.
    osd_sink_pad = osd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, probe_logic_callback, 0)

    # --- Define Bus Callback and Run Pipeline ---
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def bus_call(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream reached.")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"GStreamer error: {err}, {debug}")
            loop.quit()
        return True

    bus.connect("message", bus_call, loop)
    
    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    
    print("Pipeline has stopped. Finalizing...")
    pipeline.set_state(Gst.State.NULL)
    
    # --- After Pipeline Finishes, Generate Report ---
    print(f"Total vehicle paths finalized: {len(g_finalized_paths)}")
    excel_path = create_excel_report(
        completed_paths_data=g_finalized_paths,
        start_datetime=g_start_datetime,
        primary_direction=g_primary_direction,
        video_path=g_video_path
    )
    if excel_path:
        print(f"Successfully generated Excel report: {excel_path}")
    else:
        print("Failed to generate Excel report.")

if __name__ == '__main__':
    sys.exit(main(sys.argv))