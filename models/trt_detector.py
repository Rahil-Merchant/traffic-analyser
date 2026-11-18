"""TensorRT engines for detectors that Ultralytics cannot load (DEIMv2), behind the interface the app uses.

Engine files written by utility_scripts/onnx2tensorrt.py start with the same header Ultralytics uses: a 4-byte
little-endian length, that many bytes of JSON metadata, then the serialized engine. The metadata says what
kind of model it is, its class names and the preprocessing it needs, so one file is all that is deployed.
YOLO engines keep going through Ultralytics (which also runs their NMS); engines whose metadata says
`model_type: deim` are run by `TensorRTDetector` here.

A DEIM engine takes `images` (N, 3, S, S), normalised as models/preprocessing.py describes, and returns
`detections` (N, 300, 6): x1, y1, x2, y2, score, class in input pixels, already final (no NMS needed).
`TensorRTDetector` is called like the Ultralytics model and returns one result per image with `.boxes.data`
(n, 6) and `.names`, which is all the app and the frame selector read.
"""
import json
import os
import threading

DEFAULT_CONF = 0.25      # the score floor Ultralytics applies when the app passes none; kept for DEIM engines


def read_engine_metadata(path):
    """The JSON header of an engine file, or {} if it has none (an engine built by trtexec or trtutils)."""
    with open(path, 'rb') as f:
        return _read_header(f)[0]


def _read_header(f):
    """-> (metadata, offset of the engine bytes). Leaves the file position at that offset."""
    size = os.fstat(f.fileno()).st_size
    n = int.from_bytes(f.read(4), 'little', signed=True)
    if 0 < n <= size - 4:
        try:
            meta = json.loads(f.read(n).decode('utf-8'))
            if isinstance(meta, dict):
                return meta, 4 + n
        except (UnicodeDecodeError, ValueError):
            pass
    f.seek(0)
    return {}, 0


def write_engine(path, metadata, engine_bytes):
    header = json.dumps(metadata).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(len(header).to_bytes(4, 'little', signed=True))
        f.write(header)
        f.write(engine_bytes)


def engine_preprocessing(metadata):
    """The preprocessing mode an engine needs. Engines that do not say are Ultralytics YOLO exports."""
    return metadata.get('preprocessing', 'yolo')


def check_engine_preprocessing(path, configured):
    """Raise if the engine at `path` needs a different preprocessing than the configured one. -> metadata"""
    meta = read_engine_metadata(path)
    needed = engine_preprocessing(meta)
    if configured != needed:
        raise ValueError(
            f"{os.path.basename(path)} needs MODEL_PREPROCESSING = {needed!r}, but config.py has {configured!r}. "
            f"Set 'model_preprocessing' next to this machine's 'model_path' in PC_PATHS.")
    return meta


class _Boxes:
    def __init__(self, data):
        self.data = data            # (n, 6) float32: x1, y1, x2, y2, score, class, in model-input pixels

    def __len__(self):
        return self.data.shape[0]


class Detections:
    """One image's detections, shaped like the part of an Ultralytics result the pipeline reads."""
    def __init__(self, data, names):
        self.boxes = _Boxes(data)
        self.names = names


class TensorRTDetector:
    """Runs a DEIM engine on CUDA tensors without copying them off the GPU."""

    def __init__(self, engine_path, conf=DEFAULT_CONF):
        import tensorrt as trt
        import torch
        self._torch = torch
        self.conf = DEFAULT_CONF if conf is None else conf
        self.path = str(engine_path)
        with open(engine_path, 'rb') as f:
            self.metadata, _ = _read_header(f)
            blob = f.read()
        if self.metadata.get('model_type') != 'deim':
            raise ValueError(f"{engine_path} is not a DEIM engine built by utility_scripts/onnx2tensorrt.py "
                             f"(model_type: {self.metadata.get('model_type')!r})")
        self.names = {int(k): str(v) for k, v in self.metadata['names'].items()}
        self.preprocessing = engine_preprocessing(self.metadata)
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(blob)
        if self._engine is None:
            raise RuntimeError(f"TensorRT {trt.__version__} could not load {engine_path}. An engine only loads with the "
                               f"TensorRT version and GPU model it was built on (built with "
                               f"{self.metadata.get('tensorrt', 'an unknown version')}): rebuild it on this machine.")
        self._context = self._engine.create_execution_context()
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        ins = [n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outs = [n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(ins) != 1 or len(outs) != 1:
            raise ValueError(f"{engine_path}: expected one input and one output, found {ins} -> {outs}")
        self._in, self._out = ins[0], outs[0]
        to_torch = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        self._in_dtype = to_torch[self._engine.get_tensor_dtype(self._in)]
        self._out_dtype = to_torch[self._engine.get_tensor_dtype(self._out)]
        lo, _, hi = self._engine.get_tensor_profile_shape(self._in, 0)
        self.min_batch, self.max_batch = int(lo[0]), int(hi[0])
        self.input_shape = tuple(int(d) for d in hi[1:])          # (3, S, S)
        self._lock = threading.Lock()                             # one execution context: one inference at a time
        self._stream = torch.cuda.Stream()                        # used when the caller is on the default stream

    def __call__(self, images, conf=None, **_ignored):
        """images: (N, 3, S, S) float CUDA tensor, already normalised. Other Ultralytics arguments (imgsz, iou,
        half, augment, device, verbose) are accepted and ignored: the engine fixes them."""
        torch = self._torch
        if not images.is_cuda:
            raise ValueError('TensorRT engines run on the GPU: pass a CUDA tensor')
        if tuple(images.shape[1:]) != self.input_shape:
            raise ValueError(f"{os.path.basename(self.path)} takes {self.input_shape} images, got {tuple(images.shape[1:])}")
        conf = self.conf if conf is None else conf
        # TensorRT reads the tensor's raw memory, so it must be contiguous and of the engine's input type
        images = images.to(self._in_dtype).contiguous()
        results = []
        for start in range(0, images.shape[0], self.max_batch):
            out = self._infer(images[start:start + self.max_batch])
            for det in out:
                results.append(Detections(det[det[:, 4] >= conf], self.names))
        return results

    def _infer(self, x):
        torch = self._torch
        n = x.shape[0]
        if n < self.min_batch:                                    # engines built with a fixed batch
            x = torch.cat([x, x.new_zeros((self.min_batch - n, *x.shape[1:]))])
        with self._lock:
            ctx = self._context
            ctx.set_input_shape(self._in, tuple(x.shape))
            out = torch.empty(tuple(ctx.get_tensor_shape(self._out)), dtype=self._out_dtype, device=x.device)
            ctx.set_tensor_address(self._in, x.data_ptr())
            ctx.set_tensor_address(self._out, out.data_ptr())
            stream = torch.cuda.current_stream(x.device)
            if stream == torch.cuda.default_stream(x.device):     # TensorRT is slower on the default stream:
                stream.synchronize()                              # finish what produced x, then use our own
                stream = self._stream
            if not ctx.execute_async_v3(stream.cuda_stream):
                raise RuntimeError(f"TensorRT inference failed for {os.path.basename(self.path)}")
            stream.synchronize()
        out = out[:n].float()
        if torch.isnan(out[..., 4]).any():
            # An FP16 overflow inside the engine. Without this check the run would finish with no vehicles counted.
            raise RuntimeError(f"{os.path.basename(self.path)} returned NaN scores: a layer overflows FP16. The export "
                               f"(utility_scripts/export_onnx.py) must keep that layer in FP32, or use its --fp32 option.")
        return out
