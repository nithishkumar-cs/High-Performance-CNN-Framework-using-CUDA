#!/usr/bin/env python3
"""Evaluate a trained VOC detector on the official held-out VOC 2007 test split."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/voc2007/yolo/data.yaml"))
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        parser.error("weights and data.yaml must both exist")
    if not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch is required for this evaluation script")

    run_dir = args.weights.resolve().parent.parent
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        project=str(run_dir),
        name="test",
        exist_ok=False,
        plots=False,
    )
    results_csv = run_dir / "results.csv"
    epochs_completed = max(0, len(results_csv.read_text(encoding="utf-8").splitlines()) - 1)
    summary_path = run_dir / "summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary = {
        **previous,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "best_weights": str(args.weights.resolve()),
        "dataset": str(args.data.resolve()),
        "epochs_completed": epochs_completed,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "test_map50_95": float(metrics.box.map),
        "test_map50": float(metrics.box.map50),
        "test_map75": float(metrics.box.map75),
        "test_precision": float(metrics.box.mp),
        "test_recall": float(metrics.box.mr),
        "per_class_map50_95": {
            str(model.names[index]): float(value)
            for index, value in enumerate(metrics.box.maps)
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
