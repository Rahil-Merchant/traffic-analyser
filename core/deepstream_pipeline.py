# project/core/deepstream_pipeline.py

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import pyds
import sys
import threading
import queue
from datetime import datetime, timedelta
import numpy as np
import tempfile
import os

from configs.deepstream_config import *
from reporting.excel_report import create_excel_report, create_bidirectional_link_count_report
from core.utils import is_valid_movement, format_timestamp
from core.performance import PerformanceTracker
from loguru import logger

class DeepStreamTrafficPipeline:
    """
    DeepStream 7.1 Traffic Analytics Pipeline
    Replaces the custom video_processor.py with hardware-accelerated components
    """
    
    def __init__(self, line_mode='interactive'):
        self.line_mode = line_mode
        self.pipeline = None
        self.loop = None
        self.bus = None
        self.parsers = {}   # Store parsers keyed by stream index 
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Performance tracking
        self.perf = PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS)
        
        # Traffic data storage (replaces your active_track_data)
        self.active_vehicles = {}
        self.completed_paths = []
        self.zone_polygons = {}
        
        # Analytics configuration
        self.analytics_config_file = None
        
        # Threading
        self.processing_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        logger.info("DeepStream Traffic Pipeline initialized")

    def create_analytics_config(self, zones_dict, frame_width, frame_height):
        """
        Create nvdsanalytics configuration file from zone definitions
        Replaces your ZoneTracker functionality
        """
        roi_sections = ""
        line_sections = ""
        
        for idx, (zone_name, points) in enumerate(zones_dict.items()):
            # Convert points to semicolon-separated format for nvdsanalytics
            points_str = ";".join([f"{int(x)};{int(y)}" for x, y in points])
            
            # ROI filtering section
            roi_sections += f"""
[roi-filtering-stream-0]
enable=1
roi-{zone_name}={points_str}
inverse-roi=0
class-id=0
"""
            
            # Line crossing section for zone boundaries  
            line_sections += f"""
[line-crossing-stream-0]
enable=1
line-crossing-{zone_name}={points_str}
class-id=0
extended=0
mode=balanced
"""
        
        config_content = ANALYTICS_CONFIG_TEMPLATE.format(
            width=frame_width,
            height=frame_height,
            roi_sections=roi_sections,
            line_sections=line_sections
        )
        
        # Write to temporary file
        self.analytics_config_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False
        )
        self.analytics_config_file.write(config_content)
        self.analytics_config_file.close()
        
        logger.info(f"Analytics config created: {self.analytics_config_file.name}")
        return self.analytics_config_file.name

    def create_tracker_config(self):
        """Create NvDCF tracker configuration for traffic scenarios"""
        tracker_config = f"""
%YAML:1.0
################################################################################
# DeepStream 7.1 NvDCF Tracker Configuration for Traffic Analytics
# Optimized for accuracy in vehicle tracking scenarios
################################################################################

BaseConfig:
  minDetectorConfidence: 0.05    # Lower threshold for traffic detection
  
NvDCF:
  useUniqueID: 1
  maxTargetsPerStream: 150       # Higher for dense traffic
  
  # Feature extraction settings optimized for vehicles
  useHog: 1
  useColorNames: 1
  useHighPrecisionFeature: 1
  
  # Tracking parameters tuned for vehicle movement patterns  
  filterLr: 0.15
  gaussianSigma: 0.75
  peakCorrLr: 0.1
  lambda: 0.05
  
  # Vehicle-specific feature settings
  featureImgSize: 72
  SearchRegionPaddingScale: 1.2
  modelMaskingWindowFactor: 2
  
  # Traffic-specific temporal settings
  maxTimeToLive: 30              # Longer for vehicles crossing intersection
  probationAge: 3
  maxShadowTrackingAge: 30       # Important for occlusions
  
  # Performance optimization
  enableBatchProcess: 1
  useAsyncData: 1
"""
        
        tracker_config_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yml', delete=False
        )
        tracker_config_file.write(tracker_config)
        tracker_config_file.close()
        
        return tracker_config_file.name
    
    def create_pipeline(self, video_paths, output_path=None):
        """
        Create the complete DeepStream pipeline with proper asynchronous linking.
        """
        logger.info("Creating DeepStream pipeline...")

        self.pipeline = Gst.Pipeline.new("pipeline0")
        if not self.pipeline:
            logger.error("Unable to create Pipeline")
            return False

        # === STREAM MULTIPLEXER (Create first) ===
        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        streammux.set_property('width', MODEL_INPUT_SIZE)
        streammux.set_property('height', MODEL_INPUT_SIZE)
        streammux.set_property('batch-size', len(video_paths))
        streammux.set_property('batched-push-timeout', 40000)
        self.pipeline.add(streammux)

        # === SOURCE CONFIGURATION ===
        # Pass streammux to the source bin so it can link itself in the callback
        source_bin = self.create_multi_file_source(video_paths, streammux)
        self.pipeline.add(source_bin)

        # NOTE: The linking of source_bin to streammux now happens inside the
        # on_pad_added callback, not here.

        # === REMAINDER OF THE PIPELINE ===
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
        pgie_config = self.create_triton_config()
        pgie.set_property('config-file-path', pgie_config)
        self.pipeline.add(pgie)

        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        tracker_config_file = self.create_tracker_config()
        tracker.set_property('ll-lib-file', TRACKER_CONFIG["ll-lib-file"])
        tracker.set_property('ll-config-file', tracker_config_file)
        tracker.set_property('tracker-width', TRACKER_CONFIG["tracker-width"])
        tracker.set_property('tracker-height', TRACKER_CONFIG["tracker-height"])
        tracker.set_property('gpu_id', TRACKER_CONFIG["gpu-id"])
        self.pipeline.add(tracker)

        analytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
        analytics_config = self.create_analytics_config(
            self.zone_polygons, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE
        )
        analytics.set_property('config-file', analytics_config)
        self.pipeline.add(analytics)

        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        self.pipeline.add(nvvidconv)

        if ENABLE_VISUALIZATION:
            nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
            self.pipeline.add(nvosd)

        if output_path and ENABLE_VISUALIZATION:
            sink = self.create_file_sink(output_path)
        else:
            sink = Gst.ElementFactory.make("fakesink", "fakesink")
            sink.set_property('sync', False)
        self.pipeline.add(sink)

        # === LINK DOWNSTREAM ELEMENTS ===
        streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(analytics)
        analytics.link(nvvidconv)

        if ENABLE_VISUALIZATION:
            nvvidconv.link(nvosd)
            nvosd.link(sink)
        else:
            nvvidconv.link(sink)

        analytics_src_pad = analytics.get_static_pad("src")
        analytics_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self.analytics_src_pad_buffer_probe,
            0
        )

        logger.info("Pipeline created successfully")
        return True


    def create_multi_file_source(self, video_paths, streammux):
        """
        Creates a source Gst.Bin that dynamically links to the provided streammux.
        """
        source_bin = Gst.Bin.new("source-bin")

        for i, video_path in enumerate(video_paths):
            filesrc = Gst.ElementFactory.make("filesrc", f"filesrc_{i}")
            filesrc.set_property('location', video_path)

            qtdemux = Gst.ElementFactory.make("qtdemux", f"qtdemux_{i}")

            source_bin.add(filesrc)
            source_bin.add(qtdemux)
            filesrc.link(qtdemux)

            # Pass streammux in the context dictionary
            context = {
                "source_bin": source_bin,
                "stream_id": i,
                "streammux": streammux
            }
            qtdemux.connect("pad-added", self.on_pad_added, context)

        return source_bin

    
    def on_pad_added(self, demux, new_pad, ctx):
        """
        Handles a new pad from qtdemux, builds the decode chain, and links to streammux.
        """
        source_bin = ctx["source_bin"]
        stream_id = ctx["stream_id"]
        streammux = ctx["streammux"]

        caps = new_pad.query_caps(None)
        if not caps:
            logger.warning(f"[Stream {stream_id}] Pad has no caps, ignoring.")
            return

        caps_str = caps.to_string().lower()
        logger.debug(f"[Stream {stream_id}] Received new pad with caps: {caps_str}")

        if "video/x-h264" in caps_str:
            parser = Gst.ElementFactory.make("h264parse", f"h264parse_{stream_id}")
        elif "video/x-h265" in caps_str:
            parser = Gst.ElementFactory.make("h265parse", f"h265parse_{stream_id}")
        else:
            logger.info(f"[Stream {stream_id}] Skipping non-H264/H265 pad.")
            return

        decoder = Gst.ElementFactory.make("nvv4l2decoder", f"nvdec_{stream_id}")

        source_bin.add(parser)
        source_bin.add(decoder)
        parser.sync_state_with_parent()
        decoder.sync_state_with_parent()

        # Link demux -> parser -> decoder
        if new_pad.link(parser.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            logger.error(f"[Stream {stream_id}] Failed to link demuxer to parser.")
            return
        if not parser.link(decoder):
            logger.error(f"[Stream {stream_id}] Failed to link parser to decoder.")
            return

        # Create a ghost pad and link it to the streammux
        ghost_pad_name = f"src_{stream_id}"
        ghost_pad = Gst.GhostPad.new(ghost_pad_name, decoder.get_static_pad("src"))
        if not source_bin.add_pad(ghost_pad):
            logger.error(f"[Stream {stream_id}] Failed to add ghost pad to source bin.")
            return

        mux_sink_pad = streammux.get_request_pad(f"sink_{stream_id}")
        if ghost_pad.link(mux_sink_pad) == Gst.PadLinkReturn.OK:
            logger.info(f"✅ [Stream {stream_id}] Successfully linked source to streammux.")
        else:
            logger.error(f"❌ [Stream {stream_id}] Failed to link ghost pad to streammux.")


    def create_triton_config(self):
        """
        Create nvinferserver configuration for Triton inference
        Replaces direct YOLO model loading
        """
        triton_config = f"""
                        PluginControl {{
                        infer_config {{
                            unique_id: {GPU_ID}
                            gpu_ids: [{GPU_ID}]
                            max_batch_size: {MAX_BATCH_SIZE}

                            backend {{
                            triton {{
                                model_name: "{TRITON_MODEL_NAME}"
                                version: {TRITON_MODEL_VERSION}
                                model_repo {{ root: "/models" }}
                            }}
                            }}

                            preprocess {{
                            network_format: IMAGE_FORMAT_RGB
                            tensor_order: TENSOR_ORDER_LINEAR
                            tensor_name: "images"
                            maintain_aspect_ratio: false
                            frame_scaling_hw: FRAME_SCALING_HW_DEFAULT
                            frame_scaling_filter: 1
                            normalize {{ scale_factor: 0.0039216 channel_offsets: [0,0,0] }}
                            }}

                            postprocess {{
                            labelfile_path: "{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'labels.txt'))}"
                            other {{
                                num_detected_classes: 9
                                confidence_threshold: 0.05
                                iou_threshold: 0.6
                                topk: 300
                            }}
                            }}

                            custom_lib {{
                            path: "{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'libnvdsinfer_custom_impl_Yolo.so'))}"
                            }}
                        }}

                        input_control {{
                            process_mode: PROCESS_MODE_FULL_FRAME
                            interval: 0
                        }}

                        extra_control {{
                            copy_input_to_host_buffers: false
                        }}
                        }}
                        """

        
        triton_config_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False
        )
        triton_config_file.write(triton_config)
        triton_config_file.close()
        
        return triton_config_file.name

    def create_file_sink(self, output_path):
        """Create hardware-accelerated file output"""
        # Video converter
        nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
        
        # Caps filter
        caps = Gst.ElementFactory.make("capsfilter", "filter")
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
        
        # Hardware encoder (NVENC)[122]
        nvv4l2h264enc = Gst.ElementFactory.make("nvv4l2h264enc", "nvv4l2h264enc")
        nvv4l2h264enc.set_property('bitrate', 4000000)
        
        # Parser
        h264parse = Gst.ElementFactory.make("h264parse", "h264parse")
        
        # Muxer
        qtmux = Gst.ElementFactory.make("qtmux", "qtmux")
        
        # File sink
        filesink = Gst.ElementFactory.make("filesink", "filesink")
        filesink.set_property('location', output_path)
        filesink.set_property('sync', False)
        
        # Create sink bin
        sink_bin = Gst.Bin("sink-bin")
        sink_bin.add(nvvidconv_postosd)
        sink_bin.add(caps)
        sink_bin.add(nvv4l2h264enc)
        sink_bin.add(h264parse)
        sink_bin.add(qtmux)
        sink_bin.add(filesink)
        
        # Link elements
        nvvidconv_postosd.link(caps)
        caps.link(nvv4l2h264enc)
        nvv4l2h264enc.link(h264parse)
        h264parse.link(qtmux)
        qtmux.link(filesink)
        
        # Ghost pad
        ghost_pad = Gst.GhostPad.new("sink", nvvidconv_postosd.get_static_pad("sink"))
        sink_bin.add_pad(ghost_pad)
        
        return sink_bin

    def analytics_src_pad_buffer_probe(self, pad, info, u_data):
        """
        Main metadata processing function
        Replaces your _process_inference_results method
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
            
        # Get batch metadata[132][134]
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
                
            # Process frame timestamp
            frame_timestamp = datetime.now()  # You can get actual timestamp from metadata
            
            # Process objects in frame
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                    
                # Extract tracking information
                track_id = str(obj_meta.object_id)
                class_id = obj_meta.class_id
                confidence = obj_meta.confidence
                
                # Get vehicle type (map from your class_id_to_name_map)
                vehicle_type = self.get_vehicle_type_from_class_id(class_id)
                
                # Bounding box coordinates
                bbox = [
                    obj_meta.rect_params.left,
                    obj_meta.rect_params.top,
                    obj_meta.rect_params.width,
                    obj_meta.rect_params.height
                ]
                
                # Bottom center point for zone detection
                bottom_center = (
                    int(bbox[0] + bbox[2] / 2),
                    int(bbox[1] + bbox[3])
                )
                
                # Process vehicle tracking and zone transitions
                self.process_vehicle_tracking(
                    track_id, vehicle_type, bottom_center, 
                    frame_timestamp, obj_meta
                )
                
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            # Process analytics metadata for zone information[139]
            self.process_analytics_metadata(frame_meta)
            
            try:
                l_frame = l_frame.next  
            except StopIteration:
                break
        
        # Update performance metrics
        if self.perf:
            self.perf.sample_system_metrics()
            
        return Gst.PadProbeReturn.OK

    def process_vehicle_tracking(self, track_id, vehicle_type, bottom_center, timestamp, obj_meta):
        """
        Process vehicle tracking and zone transitions
        Replaces your tracking logic in _process_inference_results
        """
        current_zone = self.get_zone_from_point(bottom_center)
        
        if track_id not in self.active_vehicles:
            # New vehicle entry
            if current_zone:
                self.active_vehicles[track_id] = {
                    'id': track_id,
                    'type': vehicle_type,
                    'entry_direction': current_zone,
                    'entry_time': timestamp,
                    'last_zone': current_zone,
                    'last_seen': timestamp,
                    'status': 'active'
                }
                
                if self.perf:
                    self.perf.record_vehicle_entry()
                    
                logger.debug(f"New vehicle {track_id} entered {current_zone}")
        else:
            # Update existing vehicle
            vehicle_data = self.active_vehicles[track_id]
            vehicle_data['last_seen'] = timestamp
            
            previous_zone = vehicle_data.get('last_zone')
            
            if current_zone and current_zone != previous_zone:
                # Zone transition detected
                if previous_zone and is_valid_movement(
                    previous_zone, current_zone, 
                    list(self.zone_polygons.keys())
                ):
                    # Valid exit
                    duration = (timestamp - vehicle_data['entry_time']).total_seconds()
                    self.finalize_vehicle_path(
                        track_id, vehicle_data, 'exited', 
                        current_zone, timestamp, duration
                    )
                    
                    # Re-entry to new zone
                    vehicle_data.update({
                        'entry_direction': current_zone,
                        'entry_time': timestamp,
                        'last_zone': current_zone,
                        'status': 'active'
                    })
                    
                    if self.perf:
                        self.perf.record_vehicle_exit('exited', duration)
                        self.perf.record_vehicle_entry()
                        
                    logger.debug(f"Vehicle {track_id} moved from {previous_zone} to {current_zone}")
                else:
                    vehicle_data['last_zone'] = current_zone

    def process_analytics_metadata(self, frame_meta):
        """
        Process nvdsanalytics metadata for additional insights
        Extracts zone crossing events, occupancy, etc.
        """
        # Access analytics metadata[139]
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                
                if user_meta.base_meta.meta_type == pyds.NVDS_EVENT_MSG_META:
                    # Event metadata (line crossing, etc.)
                    event_msg_meta = pyds.NvDsEventMsgMeta.cast(user_meta.user_meta_data)
                    if event_msg_meta:
                        self.handle_analytics_event(event_msg_meta)
                        
            except StopIteration:
                break
                
            try:
                l_user = l_user.next
            except StopIteration:
                break

    def handle_analytics_event(self, event_msg_meta):
        """Handle analytics events from nvdsanalytics"""
        if event_msg_meta.type == pyds.NVDS_EVENT_ENTRY:
            logger.debug(f"Analytics: Entry event for object {event_msg_meta.objId}")
        elif event_msg_meta.type == pyds.NVDS_EVENT_EXIT:
            logger.debug(f"Analytics: Exit event for object {event_msg_meta.objId}")

    def get_zone_from_point(self, point):
        """
        Determine which zone contains the given point
        Uses your existing zone polygon definitions
        """
        import cv2
        
        for zone_name, polygon in self.zone_polygons.items():
            if isinstance(polygon, list):
                polygon_array = np.array(polygon, dtype=np.int32)
                result = cv2.pointPolygonTest(polygon_array, point, False)
                if result >= 0:
                    return zone_name
        return None

    def get_vehicle_type_from_class_id(self, class_id):
        """Map class ID to vehicle type using your existing mapping"""
        # This should use your class_id_to_name_map logic
        type_mapping = {
            0: 'Light Vehicle',
            1: 'Motorcycle', 
            2: 'Short Truck',
            3: 'Bus',
            # Add other mappings based on your model
        }
        return type_mapping.get(class_id, 'Light Vehicle')

    def finalize_vehicle_path(self, track_id, vehicle_data, status, exit_direction, exit_time, duration):
        """
        Finalize completed vehicle path
        Replaces your _add_to_finalized_paths method
        """
        path_entry = {
            'id': track_id,
            'type': vehicle_data['type'],
            'entry_direction': vehicle_data['entry_direction'],
            'exit_direction': exit_direction,
            'entry_time': vehicle_data['entry_time'],
            'exit_time': exit_time,
            'time_in_intersection': duration,
            'status': status
        }
        
        self.completed_paths.append(path_entry)
        logger.info(f"Vehicle {track_id} completed: {vehicle_data['entry_direction']} → {exit_direction} ({duration:.1f}s)")

    def cleanup_stale_vehicles(self, current_time):
        """
        Clean up vehicles that haven't been seen for timeout period
        Replaces cleanup_tracking_data functionality
        """
        stale_vehicles = []
        
        for track_id, vehicle_data in self.active_vehicles.items():
            if vehicle_data['status'] != 'active':
                continue
                
            time_since_seen = (current_time - vehicle_data['last_seen']).total_seconds()
            timeout_limit = VEHICLE_TIMEOUTS.get(
                vehicle_data['type'], 
                VEHICLE_TIMEOUTS['default']
            )
            
            if time_since_seen > timeout_limit:
                # Timeout vehicle
                duration = (current_time - vehicle_data['entry_time']).total_seconds()
                self.finalize_vehicle_path(
                    track_id, vehicle_data, 'timed_out', 
                    'TIMEOUT', current_time, duration
                )
                stale_vehicles.append(track_id)
                
                if self.perf:
                    self.perf.record_vehicle_exit('timed_out')
        
        # Remove stale vehicles
        for track_id in stale_vehicles:
            del self.active_vehicles[track_id]
            
        if stale_vehicles:
            logger.debug(f"Cleaned up {len(stale_vehicles)} stale vehicles")

    def bus_call(self, bus, message, loop):
        """Handle GStreamer bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning(f"Warning: {err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Error: {err}: {debug}")
            loop.quit()
        return True

    def process_video(self, video_paths, start_date_str, start_time_str, 
                     station_number_str, primary_direction, 
                     generate_link_count_report_flag=False,
                     progress_queue=None, preview_image_width=640, 
                     preview_image_height=480):
        """
        Main processing function - replaces your existing process_video method
        """
        
        if not self.processing_lock.acquire(blocking=False):
            logger.warning("Processing already in progress")
            return "Processing is already in progress."
        
        try:
            logger.info("Starting DeepStream video processing...")
            
            # Initialize zone polygons
            if self.line_mode == 'hardcoded':
                self.zone_polygons = ZONE_DEFINITIONS
            elif self.line_mode == 'interactive' and hasattr(self, 'zone_polygons'):
                # Scale polygons from preview to processing resolution
                self.zone_polygons = self._scale_polygons_to_processing_coordinates(
                    self.zone_polygons, preview_image_width, preview_image_height
                )
            else:
                raise ValueError("No valid zone polygons configured")
            
            # Parse start time
            try:
                time_part = start_time_str[:6] if len(start_time_str) > 6 else start_time_str
                ms_part = start_time_str[6:] if len(start_time_str) > 6 else "0"
                start_datetime = datetime.strptime(f"{start_date_str}{time_part}", "%y%m%d%H%M%S")
                start_datetime = start_datetime.replace(microsecond=int(ms_part.ljust(3, '0')) * 1000)
            except ValueError as e:
                raise ValueError(f"Invalid date/time format: {e}")
            
            # Create pipeline
            output_path = None
            if ENABLE_VISUALIZATION:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(video_paths[0]))[0]
                output_path = os.path.join(VIDEO_OUTPUT_DIR, f"output_{base_name}_{timestamp_str}.mp4")
                os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
            
            if not self.create_pipeline(video_paths, output_path):
                return "Failed to create pipeline"
            
            # Start performance tracking
            if self.perf:
                self.perf.start_processing()
            
            # Set up progress reporting
            if progress_queue:
                progress_queue.put_nowait({"progress": 10, "status": "Pipeline created"})
            
            # Create event loop
            self.loop = GLib.MainLoop()
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.bus_call, self.loop)
            
            # Start pipeline
            logger.info("Starting pipeline...")
            ret = self.pipeline.set_state(Gst.State.PLAYING) 
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("Unable to set the pipeline to playing state")
                return "Failed to start pipeline"
            
            if progress_queue:
                progress_queue.put_nowait({"progress": 20, "status": "Processing..."})
            
            # Start cleanup timer
            cleanup_timer = threading.Timer(30.0, self._periodic_cleanup)
            cleanup_timer.daemon = True
            cleanup_timer.start()
            
            # Run event loop
            try:
                self.loop.run()
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            
            # Stop pipeline
            self.pipeline.set_state(Gst.State.NULL)
            cleanup_timer.cancel()
            
            if progress_queue:
                progress_queue.put_nowait({"progress": 90, "status": "Finalizing..."})
            
            # Finalize remaining vehicles
            current_time = datetime.now()
            for track_id, vehicle_data in list(self.active_vehicles.items()):
                if vehicle_data['status'] == 'active':
                    duration = (current_time - vehicle_data['entry_time']).total_seconds()
                    self.finalize_vehicle_path(
                        track_id, vehicle_data, 'forced_exit',
                        'FORCED_EXIT', current_time, duration
                    )
            
            # Generate reports
            excel_path = None
            excel_link_count_path = None
            
            if self.completed_paths:
                excel_path = create_excel_report(
                    completed_paths_data=self.completed_paths,
                    start_datetime=start_datetime,
                    primary_direction=primary_direction,
                    video_path=video_paths[0]
                )
                
                if generate_link_count_report_flag:
                    secondary_dirs = [d for d in self.zone_polygons.keys() if d != primary_direction]
                    if secondary_dirs:
                        excel_link_count_path = create_bidirectional_link_count_report(
                            completed_paths_data=self.completed_paths,
                            start_datetime_video=start_datetime,
                            video_path=video_paths[0],
                            primary_direction=primary_direction,
                            secondary_direction=secondary_dirs[0],
                            video_duration_formatted="Processing Complete",
                            station_number_str=station_number_str
                        )
            
            # Performance summary
            if self.perf:
                self.perf.end_processing()
                self.perf.print_summary()
            
            # Prepare results
            completed_count = len([p for p in self.completed_paths if p['status'] == 'exited'])
            
            result_msg = "✅ DeepStream processing completed."
            if excel_path:
                result_msg += f"\nExcel report: {excel_path}"
            if excel_link_count_path:
                result_msg += f"\nBi-directional report: {excel_link_count_path}"
            if output_path:
                result_msg += f"\nOutput video: {output_path}"
            
            results_dict = {
                "video_path": output_path,
                "excel_path": excel_path,
                "bidirectional_excel_path": excel_link_count_path
            }
            
            stats_dict = {
                "total_valid_paths": completed_count,
                "processing_time_seconds": self.perf.total_time if self.perf else 0,
                "processing_fps": completed_count / self.perf.total_time if self.perf and self.perf.total_time > 0 else 0,
                "total_frames_processed": len(self.completed_paths)
            }
            
            if progress_queue:
                progress_queue.put_nowait({
                    "progress": 100, 
                    "status": "complete",
                    "results": results_dict,
                    "statistics": stats_dict
                })
            
            return result_msg
            
        except Exception as e:
            logger.exception("Error during DeepStream processing")
            if progress_queue:
                progress_queue.put_nowait({
                    "progress": 100, 
                    "status": "error", 
                    "message": str(e)
                })
            return f"❌ Error: {e}"
            
        finally:
            self.cleanup()
            self.processing_lock.release()

    def _periodic_cleanup(self):
        """Periodic cleanup of stale vehicles"""
        if not self.stop_event.is_set():
            self.cleanup_stale_vehicles(datetime.now())
            # Schedule next cleanup
            cleanup_timer = threading.Timer(30.0, self._periodic_cleanup)
            cleanup_timer.daemon = True
            cleanup_timer.start()

    def _scale_polygons_to_processing_coordinates(self, polygons_dict, original_w, original_h):
        """Scale polygon coordinates from preview to processing resolution"""
        if not polygons_dict:
            return {}
            
        target_w, target_h = MODEL_INPUT_SIZE, MODEL_INPUT_SIZE
        scale_x = target_w / original_w
        scale_y = target_h / original_h
        
        scaled_polygons = {}
        for direction, points in polygons_dict.items():
            scaled_polygons[direction] = [
                (int(x * scale_x), int(y * scale_y)) for x, y in points
            ]
        
        logger.info(f"Scaled polygons from {original_w}x{original_h} to {target_w}x{target_h}")
        return scaled_polygons

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up DeepStream pipeline...")
        
        self.stop_event.set()
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            
        if self.loop:
            if self.loop.is_running():
                self.loop.quit()
            self.loop = None
            
        # Clean up temporary config files
        if self.analytics_config_file:
            try:
                os.unlink(self.analytics_config_file)
            except:
                pass
                
        self.active_vehicles.clear()
        self.completed_paths.clear()
        
        logger.info("Cleanup complete")

    def set_zone_polygons(self, polygons_dict):
        """Set zone polygons for interactive mode"""
        self.zone_polygons = polygons_dict
