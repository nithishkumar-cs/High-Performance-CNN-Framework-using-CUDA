#!/usr/bin/env python3
'''Compare raw ONNX CPU and Jetson TensorRT FP16 heads on the same image.'''

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from scratch_yolo.numpy_decode import decode_numpy_heads
from scratch_yolo.preprocess import letterbox
from tools.tensorrt_inference import TensorRTSession


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--side', type=int, default=416)
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--max-absolute-error', type=float, default=.1)
    parser.add_argument('--min-box-iou', type=float, default=.9)
    parser.add_argument('--max-score-error', type=float, default=.05)
    args = parser.parse_args()
    with Image.open(args.image) as opened:
        tensor, _, _, _ = letterbox(opened, args.side)
    cpu = ort.InferenceSession(str(args.onnx), providers=['CPUExecutionProvider'])
    trt = TensorRTSession(args.engine)
    try:
        expected = sorted(cpu.run(None, {cpu.get_inputs()[0].name: tensor}),
                          key=lambda item: item.shape[-1])
        actual = sorted(trt.run(None, {trt.get_inputs()[0].name: tensor}),
                        key=lambda item: item.shape[-1])
    finally:
        trt.close()
    if len(expected) != 2 or len(actual) != 2:
        raise RuntimeError('expected two heads from each backend')
    heads = []
    for reference, got in zip(expected, actual):
        if reference.shape != got.shape:
            raise RuntimeError(f'output shape mismatch: {reference.shape} vs {got.shape}')
        difference = np.abs(reference - got)
        heads.append({'shape': list(reference.shape), 'max_absolute_error': float(difference.max()),
                      'mean_absolute_error': float(difference.mean())})
    cpu_boxes = decode_numpy_heads(expected, args.confidence)[0]
    trt_boxes = decode_numpy_heads(actual, args.confidence)[0]
    used = np.zeros(len(trt_boxes), dtype=bool)
    minimum_iou = 1.0
    maximum_score_error = 0.0
    matched = 0
    for reference in cpu_boxes:
        choices = np.flatnonzero((trt_boxes[:, 5] == reference[5]) & ~used)
        if not len(choices):
            break
        others = trt_boxes[choices, :4]
        xy1 = np.maximum(reference[:2], others[:, :2])
        xy2 = np.minimum(reference[2:4], others[:, 2:4])
        intersection = np.prod(np.maximum(xy2 - xy1, 0), axis=1)
        first_area = np.prod(np.maximum(reference[2:4] - reference[:2], 0))
        other_area = np.prod(np.maximum(others[:, 2:4] - others[:, :2], 0), axis=1)
        overlaps = intersection / np.maximum(first_area + other_area - intersection, 1e-9)
        best = int(np.argmax(overlaps))
        chosen = choices[best]
        used[chosen] = True
        matched += 1
        minimum_iou = min(minimum_iou, float(overlaps[best]))
        maximum_score_error = max(maximum_score_error,
                                  abs(float(reference[4] - trt_boxes[chosen, 4])))
    result = {'heads': heads, 'cpu_detections': len(cpu_boxes),
              'tensorrt_detections': len(trt_boxes), 'matched_detections': matched,
              'minimum_matched_box_iou': minimum_iou if matched else None,
              'maximum_matched_score_error': maximum_score_error if matched else None,
              'confidence': args.confidence, 'max_absolute_error_allowed': args.max_absolute_error}
    print(json.dumps(result, indent=2))
    if any(head['max_absolute_error'] > args.max_absolute_error for head in heads):
        raise RuntimeError('TensorRT output exceeded absolute-error tolerance')
    if matched != len(cpu_boxes) or matched != len(trt_boxes):
        raise RuntimeError('TensorRT detection count/class parity failed')
    if matched and (minimum_iou < args.min_box_iou or maximum_score_error > args.max_score_error):
        raise RuntimeError('TensorRT detection box/score parity failed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
