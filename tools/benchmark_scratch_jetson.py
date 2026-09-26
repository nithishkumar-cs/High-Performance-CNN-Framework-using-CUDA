#!/usr/bin/env python3
'''Benchmark raw-head scratch YOLO via ONNX Runtime or native TensorRT.'''

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from scratch_yolo.numpy_decode import decode_numpy_heads
from scratch_yolo.preprocess import letterbox


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metadata() -> dict:
    details = {'platform': platform.platform()}
    release = Path('/etc/nv_tegra_release')
    if release.is_file():
        details['jetson_release'] = release.read_text(encoding='utf-8').strip()
    for name, command in {'power_mode': ['nvpmodel', '-q'],
                          'clock_settings': ['jetson_clocks', '--show']}.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    check=False, timeout=5)
            if result.returncode == 0:
                details[name] = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return details


def create_session(path: Path, backend: str):
    if backend == 'tensorrt-native':
        from tools.tensorrt_inference import TensorRTSession
        return TensorRTSession(path)
    import onnxruntime as ort
    provider = 'CPUExecutionProvider' if backend == 'cpu' else 'CUDAExecutionProvider'
    if provider not in ort.get_available_providers():
        raise RuntimeError(f'{provider} is unavailable: {ort.get_available_providers()}')
    return ort.InferenceSession(str(path), providers=[provider])


def detect(session, image: Image.Image, side: int, confidence: float,
           nms_iou: float) -> tuple[int, float]:
    tensor, scale, left, top = letterbox(image, side)
    name = session.get_inputs()[0].name
    start = time.perf_counter_ns()
    outputs = session.run(None, {name: tensor})
    inference_ms = (time.perf_counter_ns() - start) / 1e6
    # TensorRT I/O enumeration need not match graph output order.
    heads = sorted(outputs, key=lambda value: value.shape[-1])
    boxes = decode_numpy_heads(heads, confidence, nms_iou)[0]
    width, height = image.size
    xyxy = boxes[:, :4].copy()
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] * side - left) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] * side - top) / scale
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
    valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
    return int(np.count_nonzero(valid)), inference_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'tensorrt-native'], default='cpu')
    parser.add_argument('--side', type=int, default=416)
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--nms-iou', type=float, default=.45)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('runs/scratch_yolo/benchmark.json'))
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.side < 32 or args.side % 32:
        parser.error('invalid runs, warmup, or input side')
    with Image.open(args.image) as opened:
        image = opened.convert('RGB')
    session = create_session(args.model, args.backend)
    expected = (1, 3, args.side, args.side)
    if tuple(session.get_inputs()[0].shape) != expected:
        parser.error(f'model input shape differs from {expected}')
    for _ in range(args.warmup):
        detect(session, image, args.side, args.confidence, args.nms_iou)
    totals, model_times = [], []
    count = 0
    for _ in range(args.runs):
        start = time.perf_counter_ns()
        count, model_ms = detect(session, image, args.side, args.confidence, args.nms_iou)
        totals.append((time.perf_counter_ns() - start) / 1e6)
        model_times.append(model_ms)
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    result = {
        'model': str(args.model.resolve()), 'model_sha256': digest,
        'image': str(args.image.resolve()), 'backend': args.backend,
        'providers': session.get_providers(), 'device': metadata(),
        'side': args.side, 'confidence': args.confidence, 'nms_iou': args.nms_iou,
        'warmup': args.warmup, 'runs': args.runs, 'detections': count,
        'scope': 'letterbox + model with transfers + raw-head decode/NMS + box remap; image load excluded',
        'end_to_end_ms': {'mean': statistics.mean(totals), 'p95': percentile(totals, 95)},
        'inference_ms': {'mean': statistics.mean(model_times), 'p95': percentile(model_times, 95)},
        'end_to_end_fps': 1000 / statistics.mean(totals),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
