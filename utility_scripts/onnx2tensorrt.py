"""Build a TensorRT engine from an ONNX file written by utility_scripts/export_onnx.py (DEIMv2 or YOLO).

    python utility_scripts/onnx2tensorrt.py models/weights/deimv2_l_cctv.onnx
    python utility_scripts/onnx2tensorrt.py models/weights/yolo26l_o2m_sq640.onnx --max-batch 64 --no-fp16

Run it with the Python environment that runs the app, on the machine that runs the app: an engine only loads
with the TensorRT version and GPU model it was built on. It needs nothing but `tensorrt` and `onnx`.

The engine is written next to the ONNX file (or to --out) with a JSON header in the Ultralytics layout, filled
from the ONNX file's metadata: class names, input size, and for DEIM the model type and preprocessing. A YOLO
engine built here loads in the app exactly like one exported by Ultralytics; a DEIM engine is run by
models/trt_detector.py. Batch is dynamic from --min-batch to --max-batch and tuned for --opt-batch (the app
always sends OPTIMAL_BATCH_SIZE frames, 64).

Engines are FP16. DEIMv2 has a few layers whose values exceed the FP16 range, so export_onnx.py writes its
ONNX file in mixed precision and it is built here exactly as typed. Always score a new engine before using it
(training/predict_coco.py engine ...): a precision problem shows up as missing detections, not as a build error.
"""
import argparse
import ast
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.trt_detector import write_engine  # noqa: E402


def onnx_metadata(onnx_path):
    """The ONNX file's metadata_props, with values parsed back from the strings ONNX stores them as."""
    import onnx
    model = onnx.load(str(onnx_path), load_external_data=False)
    meta = {}
    for p in model.metadata_props:
        try:
            meta[p.key] = ast.literal_eval(p.value)
        except (ValueError, SyntaxError):
            meta[p.key] = p.value
    return meta


def build(onnx_path, out_path, fp16=True, min_batch=1, opt_batch=64, max_batch=64, imgsz=None, workspace_gb=8,
          timing_cache=None, verbose=False):
    import tensorrt as trt
    meta = onnx_metadata(onnx_path)
    if imgsz is None:
        imgsz = meta.get('imgsz', 640)
    imgsz = [imgsz, imgsz] if isinstance(imgsz, int) else list(imgsz)

    # 'mixed': export_onnx.py already gave every layer its type (FP16, with the layers that overflow it in FP32),
    # and a strongly typed network makes TensorRT keep them. 'fp32': a DEIM model that must not be reduced.
    precision = meta.get('precision')
    if precision == 'fp32' and fp16:
        print('this ONNX file is marked FP32-only (DEIMv2 exported with --fp32): building an FP32 engine')
        fp16 = False
    if precision == 'mixed' and not fp16:
        raise ValueError('this ONNX file is already mixed FP16/FP32; for an FP32 engine export it again with --fp32')

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if precision == 'mixed':
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT {trt.__version__} could not parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16 and precision != 'mixed':
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        dims = list(inp.shape)
        if dims[0] != -1:                      # fixed batch in the ONNX file: nothing to choose
            if -1 in dims:
                raise ValueError(f"input {inp.name} {dims}: dynamic size with a fixed batch is not supported")
            print(f"input {inp.name}: fixed shape {dims}")
            lo = opt = hi = dims
            min_batch = opt_batch = max_batch = dims[0]
        else:
            rest = [d if d != -1 else imgsz[k - 2] for k, d in enumerate(dims) if k > 0]   # dynamic H, W -> imgsz
            lo, opt, hi = ([b, *rest] for b in (min_batch, opt_batch, max_batch))
            print(f"input {inp.name}: batch {min_batch}..{max_batch} (tuned for {opt_batch}), image {rest}")
        profile.set_shape(inp.name, lo, opt, hi)
    config.add_optimization_profile(profile)

    cache = None
    if timing_cache:
        data = Path(timing_cache).read_bytes() if Path(timing_cache).exists() else b''
        cache = config.create_timing_cache(data)
        config.set_timing_cache(cache, ignore_mismatch=True)

    kind = 'mixed FP16/FP32' if precision == 'mixed' else 'FP16' if fp16 else 'FP32'
    print(f"building {kind} engine with TensorRT {trt.__version__} (this takes minutes)...")
    t0 = time.time()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError('TensorRT failed to build the engine (see the messages above)')
    if timing_cache:
        Path(timing_cache).write_bytes(bytes(config.get_timing_cache().serialize()))

    meta.setdefault('preprocessing', 'yolo')   # Ultralytics exports do not say; everything else is stamped by export_onnx.py
    meta.update(batch=max_batch, imgsz=imgsz, dynamic=min_batch != max_batch, fp16=fp16,
                tensorrt=trt.__version__, onnx=Path(onnx_path).name)
    if isinstance(meta.get('args'), dict):     # what Ultralytics reads to decide the engine is dynamic and has no NMS
        meta['args'].update(dynamic=meta['dynamic'], half=fp16, batch=max_batch, nms=False, format='engine')
    write_engine(out_path, meta, bytes(engine))
    print(f"wrote {out_path} ({Path(out_path).stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f} s")
    print(f"  model: {meta.get('model_type', 'yolo')}, preprocessing: {meta['preprocessing']}, "
          f"classes: {len(meta.get('names', {}))}")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('onnx')
    ap.add_argument('--out', help='engine path (default: the ONNX path with .engine)')
    ap.add_argument('--no-fp16', action='store_true', help='build in FP32 (about half the speed)')
    ap.add_argument('--min-batch', type=int, default=1)
    ap.add_argument('--opt-batch', type=int, default=64)
    ap.add_argument('--max-batch', type=int, default=64)
    ap.add_argument('--imgsz', type=int, nargs='+', help='input height and width when the ONNX leaves them dynamic')
    ap.add_argument('--workspace-gb', type=float, default=8)
    ap.add_argument('--timing-cache', help='file to reuse layer timings across builds (faster rebuilds)')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args(argv)
    if not a.min_batch <= a.opt_batch <= a.max_batch:
        ap.error('need --min-batch <= --opt-batch <= --max-batch')
    imgsz = a.imgsz[0] if a.imgsz and len(a.imgsz) == 1 else a.imgsz
    build(a.onnx, a.out or str(Path(a.onnx).with_suffix('.engine')), fp16=not a.no_fp16, min_batch=a.min_batch,
          opt_batch=a.opt_batch, max_batch=a.max_batch, imgsz=imgsz, workspace_gb=a.workspace_gb,
          timing_cache=a.timing_cache, verbose=a.verbose)


if __name__ == '__main__':
    main()
