"""Prepare an external Darknet YOLOv3-tiny VOC training run.

The input configuration is supplied by the caller from pjreddie/darknet. This
script does not bundle Darknet code or configuration. Generated cfg files are
adaptations of https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg;
see that repository's LICENSE and LICENSE.mit before redistributing them.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


VOC_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm"}


def images_and_labels(images_dir: Path, labels_dir: Path) -> list[Path]:
    """Validate that every image has a well-formed matching YOLO label file and return the sorted image paths."""
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"missing image/label directory: {images_dir}, {labels_dir}")
    images = sorted(p.resolve() for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"no images in {images_dir}")
    for image in images:
        label = labels_dir / (image.stem + ".txt")
        if not label.is_file():
            raise ValueError(f"missing label for {image}: {label}")
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label}:{line_number}: expected 5 YOLO fields")
            class_id = int(fields[0])
            box = [float(value) for value in fields[1:]]
            if not 0 <= class_id < len(VOC_NAMES):
                raise ValueError(f"{label}:{line_number}: class outside VOC range")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
                raise ValueError(f"{label}:{line_number}: nonfinite/out-of-range box")
            if box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"{label}:{line_number}: empty box")
    return images


def adapt_cfg(source: str, max_batches: int) -> str:
    """Rewrite the upstream Darknet yolov3-tiny.cfg to 20 VOC classes/75 filters and this run's training schedule; raises ValueError if the upstream layout doesn't match expectations."""
    lines = source.splitlines()
    section = ""
    yolo_count = 0
    heads = 0
    output = [
        "# Generated adaptation of pjreddie/darknet cfg/yolov3-tiny.cfg.",
        "# Source: https://github.com/pjreddie/darknet",
        "# See upstream LICENSE and LICENSE.mit for notices.",
    ]
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            if section == "yolo":
                yolo_count += 1
            output.append(line)
            continue
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(line)
            continue
        key, value = (part.strip() for part in stripped.split("=", 1))
        replacement = None
        if section == "net":
            replacement = {
                "batch": "32", "subdivisions": "16", "learning_rate": "0.001",
                "burn_in": "1000", "max_batches": str(max_batches),
                "steps": "8000,9000", "scales": ".1,.1",
            }.get(key)
        elif section == "yolo":
            if key == "classes":
                if value != "80":
                    raise ValueError("expected original 80-class YOLO head")
                replacement = "20"
                heads += 1
            elif key == "random":
                replacement = "0"  # fixed 416 input fits Jetson memory
        elif section == "convolutional" and key == "filters" and value == "255":
            replacement = "75"  # 3 anchors * (20 classes + 5 box/objectness values)
        output.append(f"{key}={replacement}" if replacement is not None else line)
    result = "\n".join(output) + "\n"
    if yolo_count != 2 or heads != 2 or result.count("filters=75") != 2:
        raise ValueError("expected exactly two YOLOv3-tiny VOC heads")
    if "batch=32" not in result or "subdivisions=16" not in result:
        raise ValueError("upstream cfg layout changed; training batch was not set")
    return result


def write_new(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def main() -> None:
    """CLI entry point: validate the VOC dataset layout and write an adapted Darknet VOC training cfg."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-cfg", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train = images_and_labels(args.train_images, args.train_labels)
    val = images_and_labels(args.val_images, args.val_labels)
    if set(train) & set(val):
        raise ValueError("training and validation image paths overlap")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "backup").mkdir()
    source = args.upstream_cfg.read_text(encoding="utf-8")
    write_new(args.output_dir / "yolov3-tiny-voc.cfg", adapt_cfg(source, 10000))
    write_new(args.output_dir / "yolov3-tiny-voc-preflight.cfg", adapt_cfg(source, 4))
    write_new(args.output_dir / "train.txt", "\n".join(map(str, train)) + "\n")
    write_new(args.output_dir / "val.txt", "\n".join(map(str, val)) + "\n")
    write_new(args.output_dir / "voc.names", "\n".join(VOC_NAMES) + "\n")
    write_new(args.output_dir / "voc.data", "\n".join((
        "classes=20", f"train={args.output_dir.resolve() / 'train.txt'}",
        f"valid={args.output_dir.resolve() / 'val.txt'}",
        f"names={args.output_dir.resolve() / 'voc.names'}",
        f"backup={args.output_dir.resolve() / 'backup'}", "",
    )))
    print(f"Prepared YOLOv3-tiny VOC: train={len(train)} val={len(val)} "
          f"batch=32 subdivisions=16 max_batches=10000 output={args.output_dir}")


if __name__ == "__main__":
    main()
