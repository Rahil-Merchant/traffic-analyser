import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds  # DeepStream Python bindings

from config import MODEL_INPUT_SIZE, TARGET_FPS, LINE_POINTS

def bus_call(bus, message, loop):
    # Handle messages (errors, EOS)
    t = message.type
    if t == Gst.MessageType.EOS: loop.quit()
    elif t == Gst.MessageType.ERROR: loop.quit()
    return True

def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        vehicle_counts = {}  # Per-zone counts
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            # Extract metadata (class, zone, etc.) for reporting
            zone_id = obj_meta.unique_component_id  # From nvdsanalytics
            vehicle_type = obj_meta.class_id  # Map to your types
            vehicle_counts.setdefault(zone_id, {}).setdefault(vehicle_type, 0)
            vehicle_counts[zone_id][vehicle_type] += 1
            l_obj = l_obj.next
        # Feed to your reporting logic (e.g., queue for excel_report.py)
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK

def main(video_paths):
    Gst.init(None)
    loop = GLib.MainLoop()
    pipeline = Gst.Pipeline()

    # Source (multi-file)
    source_bin = create_multi_source_bin(video_paths)  # Custom function for multi-URI
    pipeline.add(source_bin)

    # StreamMux for batching
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', MODEL_INPUT_SIZE)
    streammux.set_property('height', MODEL_INPUT_SIZE)
    streammux.set_property('batch-size', 128)  # Your OPTIMAL_BATCH_SIZE
    streammux.set_property('batched-push-timeout', 40000)
    pipeline.add(streammux)

    # PGIE (Triton Inference)
    pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    pgie.set_property('config-file-path', 'triton_config.pgie')  # Your Triton config
    pipeline.add(pgie)

    # Tracker (NvDCF)
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file', 'config_tracker_NvDCF_perf.yml')  # Tune for accuracy
    pipeline.add(tracker)

    # Analytics (Zones)
    analytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    analytics.set_property('config-file', 'config_analytics.txt')  # Define zones/lines
    pipeline.add(analytics)

    # OSD for visualization (if enabled)
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    pipeline.add(nvosd)
    nvosd_sink_pad = nvosd.get_static_pad("sink")
    nvosd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # Sink (file or RTSP)
    sink = Gst.ElementFactory.make("filesink", "filesink")
    sink.set_property('location', 'output.mp4')
    pipeline.add(sink)

    # Link elements
    source_bin.link(streammux)
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = source_bin.get_static_pad("src")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(analytics)
    analytics.link(nvosd)
    nvosd.link(sink)

    # Bus for messages
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)
    loop.run()
    pipeline.set_state(Gst.State.NULL)

# Custom multi-source bin function (implement based on deepstream-test5)
def create_multi_source_bin(video_paths):
    # Create bin with uridecodebin for each path, etc.
    pass  # See DeepStream docs for details[1]

if __name__ == '__main__':
    main(["video1.mp4", "video2.mp4"])
