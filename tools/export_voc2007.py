#!/usr/bin/env python3
"""Export a trained VOC YOLOv8 checkpoint to a decoded, NMS-free ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Checkpoint is missing: {args.weights}")
    if args.imgsz < 32:
        parser.error("imgsz must be at least 32")

    model = YOLO(str(args.weights))
    exported = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        batch=1,
        dynamic=False,
        nms=False,
        simplify=False,
        device="cpu",
    ))
    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("Expected exactly one ONNX input and output")
    output_shape = outputs[0].shape
    if len(output_shape) != 3 or output_shape[1] != 24:
        raise RuntimeError(f"Expected VOC YOLOv8 output [1, 24, N], got {output_shape}")
    manifest = {
        "weights": str(args.weights.resolve()),
        "onnx": str(exported.resolve()),
        "input_shape": inputs[0].shape,
        "output_shape": output_shape,
        "providers": session.get_providers(),
    }
    exported.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
