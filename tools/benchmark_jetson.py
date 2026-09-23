#!/usr/bin/env python3
"""Benchmark one YOLO ONNX model in one process on the target device."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from yolo_inference import Detector, load_image, make_session


def percentile(sorted_values: list[float], value: float) -> float:
    index = (len(sorted_values) - 1) * value / 100
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def device_metadata() -> dict[str, str]:
    details = {"platform": platform.platform()}
    release = Path("/etc/nv_tegra_release")
    if release.is_file():
        details["jetson_release"] = release.read_text(encoding="utf-8").strip()
    for key, command in {
        "power_mode": ["nvpmodel", "-q"],
        "clock_settings": ["jetson_clocks", "--show"],
    }.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
            if result.returncode == 0:
                details[key] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--format", choices=("v5", "v8"), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "tensorrt"), default="cpu")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.size <= 0:
        parser.error("runs and size must be positive; warmup must be nonnegative")

    image = load_image(args.image)
    session = make_session(args.model, args.backend)
    detector = Detector(session, args.format, args.size)
    for _ in range(args.warmup):
        detector.detect(image)

    total_ms: list[float] = []
    model_ms: list[float] = []
    detection_count = 0
    for _ in range(args.runs):
        start = time.perf_counter_ns()
        detections, inference_ms = detector.detect(image)
        total_ms.append((time.perf_counter_ns() - start) / 1_000_000)
        model_ms.append(inference_ms)
        detection_count = len(detections)
    total_ms.sort()
    model_ms.sort()
    result = {
        "model": str(args.model.resolve()),
        "image": str(args.image.resolve()),
        "format": args.format,
        "backend_requested": args.backend,
        "providers_active": session.get_providers(),
        "device": device_metadata(),
        "warmup": args.warmup,
        "runs": args.runs,
        "detections": detection_count,
        "end_to_end_ms": {
            "mean": statistics.mean(total_ms),
            "median": statistics.median(total_ms),
            "p95": percentile(total_ms, 95),
            "min": total_ms[0],
            "max": total_ms[-1],
        },
        "inference_ms": {
            "mean": statistics.mean(model_ms),
            "median": statistics.median(model_ms),
            "p95": percentile(model_ms, 95),
        },
        "end_to_end_fps": 1000 / statistics.mean(total_ms),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
