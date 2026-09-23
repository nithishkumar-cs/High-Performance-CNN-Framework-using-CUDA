#!/usr/bin/env python3
"""Run decoded YOLOv5 or YOLOv8 ONNX detection models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


def make_session(model: Path, backend: str) -> Any:
    import onnxruntime as ort

    requested = PROVIDERS[backend]
    available = ort.get_available_providers()
    if requested not in available:
        raise RuntimeError(f"{requested} is unavailable; installed providers: {available}")
    providers = [requested]
    if backend == "tensorrt" and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if backend != "cpu":
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(model), providers=providers)
    if requested not in session.get_providers():
        raise RuntimeError(f"{requested} was not activated: {session.get_providers()}")
    return session


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, float, int, int]:
    width, height = image.size
    scale = min(size / width, size / height)
    new_width = min(size, round(width * scale))
    new_height = min(size, round(height * scale))
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((new_width, new_height), Image.Resampling.BILINEAR), (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(tensor), scale, left, top


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xy1 = np.maximum(box[:2], boxes[:, :2])
    xy2 = np.minimum(box[2:], boxes[:, 2:])
    overlap = np.prod(np.maximum(xy2 - xy1, 0), axis=1)
    area_a = np.prod(np.maximum(box[2:] - box[:2], 0))
    area_b = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return overlap / np.maximum(area_a + area_b - overlap, 1e-12)


def decode(
    output: np.ndarray,
    model_format: str,
    original_size: tuple[int, int],
    scale: float,
    left: int,
    top: int,
    confidence: float = 0.25,
    nms_threshold: float = 0.45,
    max_detections: int = 300,
) -> list[dict[str, Any]]:
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Expected a batched output tensor, got {output.shape}")
    if model_format == "v8":
        rows = output[0].T  # [1, 4 + classes, candidates]
        if rows.shape[1] < 5:
            raise ValueError(f"YOLOv8 output has too few channels: {output.shape}")
        class_scores = rows[:, 4:]
        classes = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(rows)), classes]
    elif model_format == "v5":
        rows = output[0]  # [1, candidates, 5 + classes]
        if rows.shape[1] < 6:
            raise ValueError(f"YOLOv5 output has too few channels: {output.shape}")
        class_scores = rows[:, 5:]
        classes = class_scores.argmax(axis=1)
        scores = rows[:, 4] * class_scores[np.arange(len(rows)), classes]
    else:
        raise ValueError(f"Unsupported format: {model_format}")

    selected = np.flatnonzero(np.isfinite(scores) & (scores >= confidence))
    selected = selected[np.argsort(scores[selected])[::-1][:3000]]
    boxes = rows[selected, :4].astype(np.float32, copy=True)
    centers = boxes[:, :2].copy()
    half_sizes = boxes[:, 2:] / 2
    boxes[:, :2] = centers - half_sizes
    boxes[:, 2:] = centers + half_sizes
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_size[0])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_size[1])

    kept: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        if len(kept) >= max_detections:
            break
        box = boxes[index]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        label = int(classes[candidate])
        if any(
            item["class_id"] == label
            and iou(box, np.asarray(item["box"])[None])[0] > nms_threshold
            for item in kept
        ):
            continue
        kept.append({
            "class_id": label,
            "confidence": float(scores[candidate]),
            "box": [float(value) for value in box],
        })
    return kept


class Detector:
    def __init__(
        self,
        session: Any,
        model_format: str,
        size: int = 640,
        confidence: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> None:
        self.session = session
        self.model_format = model_format
        self.size = size
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one model input, found {len(inputs)}")
        self.input_name = inputs[0].name
        shape = inputs[0].shape
        if len(shape) != 4 or (isinstance(shape[1], int) and shape[1] != 3):
            raise ValueError(f"Expected NCHW RGB model input, got {shape}")
        if any(isinstance(dim, int) and dim != size for dim in shape[2:]):
            raise ValueError(f"Model expects {shape[2:]} input; select the matching --size")

    def detect(self, image: Image.Image) -> tuple[list[dict[str, Any]], float]:
        tensor, scale, left, top = letterbox(image, self.size)
        start = time.perf_counter_ns()
        output = self.session.run(None, {self.input_name: tensor})
        model_ms = (time.perf_counter_ns() - start) / 1_000_000
        if len(output) != 1:
            raise ValueError(f"Expected one decoded output tensor, found {len(output)}")
        return decode(output[0], self.model_format, image.size, scale, left, top,
                      self.confidence, self.nms_threshold), model_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--format", choices=("v5", "v8"), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "tensorrt"), default="cpu")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--annotated", type=Path)
    args = parser.parse_args()
    if args.size <= 0 or not 0 <= args.confidence <= 1 or not 0 <= args.nms <= 1:
        parser.error("size must be positive; confidence and nms must be in [0, 1]")

    session = make_session(args.model, args.backend)
    detector = Detector(session, args.format, args.size, args.confidence, args.nms)
    image = load_image(args.image)
    detections, model_ms = detector.detect(image)
    result = {"providers_active": session.get_providers(), "inference_ms": model_ms,
              "detections": detections}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.annotated:
        draw = ImageDraw.Draw(image)
        for item in detections:
            draw.rectangle(item["box"], outline="red", width=2)
            draw.text(item["box"][:2], f"{item['class_id']} {item['confidence']:.2f}", fill="red")
        image.save(args.annotated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
