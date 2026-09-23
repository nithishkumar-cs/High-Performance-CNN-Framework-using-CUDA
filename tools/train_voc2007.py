#!/usr/bin/env python3
"""Fine-tune a compact YOLO detector on VOC 2007 and evaluate on its test set."""

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
    parser.add_argument("--data", type=Path, default=Path("data/voc2007/yolo/data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("runs/voc2007"))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset configuration is missing: {args.data}")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or args.workers < 0:
        parser.error("epochs, imgsz and batch must be positive; workers must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch is required for this training script")

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        project=str(args.project.resolve()),
        name="yolov8n",
        exist_ok=False,
        seed=args.seed,
        deterministic=True,
        patience=10,
        plots=False,
    )
    run_dir = Path(model.trainer.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not produce {best_weights}")

    best = YOLO(str(best_weights))
    test_metrics = best.val(
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
    summary = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "model_initial": args.model,
        "best_weights": str(best_weights.resolve()),
        "dataset": str(args.data.resolve()),
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "test_map50_95": float(test_metrics.box.map),
        "test_map50": float(test_metrics.box.map50),
        "test_map75": float(test_metrics.box.map75),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
