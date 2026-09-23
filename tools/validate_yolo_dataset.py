#!/usr/bin/env python3
"""Validate an Ultralytics/Darknet-style YOLO detection dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_split(root: Path, split: str, class_count: int) -> tuple[int, int]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.is_dir():
        print(f"warning: missing {image_dir}")
        return 0, 0

    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    errors = 0
    labels = 0
    unlabeled = 0
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            # YOLO permits images with no objects and no label file.
            unlabeled += 1
            continue
        for line_number, line in enumerate(label.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                print(f"error: {label}:{line_number}: expected 5 fields")
                errors += 1
                continue
            try:
                class_id = int(fields[0])
                coords = [float(value) for value in fields[1:]]
            except ValueError:
                print(f"error: {label}:{line_number}: non-numeric label")
                errors += 1
                continue
            if not 0 <= class_id < class_count:
                print(f"error: {label}:{line_number}: class id out of range")
                errors += 1
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coords):
                print(f"error: {label}:{line_number}: coordinates must be normalized")
                errors += 1
            if coords[2] <= 0 or coords[3] <= 0:
                print(f"error: {label}:{line_number}: width and height must be positive")
                errors += 1
            labels += 1
    print(f"{split}: images={len(images)} labels={labels} unlabeled={unlabeled} errors={errors}")
    return labels, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--classes", type=Path, required=True)
    args = parser.parse_args()

    class_names = [line.strip() for line in args.classes.read_text().splitlines() if line.strip()]
    if not class_names:
        raise SystemExit("classes file is empty")
    total_errors = 0
    for split in ("train", "val", "test"):
        _, errors = check_split(args.dataset, split, len(class_names))
        total_errors += errors
    print(f"classes={len(class_names)} total_errors={total_errors}")
    return int(total_errors != 0)


if __name__ == "__main__":
    raise SystemExit(main())
