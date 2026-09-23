#!/usr/bin/env python3
"""Convert the official VOC 2007 XML annotations to a YOLO detection dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


def parse_annotation(path: Path) -> tuple[list[str], int]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size in {path}")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {path}")

    labels: list[str] = []
    ignored = 0
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_TO_ID:
            raise ValueError(f"Unknown VOC class {name!r} in {path}")
        if obj.findtext("difficult", "0") == "1":
            ignored += 1
            continue
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bounding box in {path}")
        # VOC uses one-based inclusive image coordinates.
        x1 = max(0.0, min(float(box.findtext("xmin", "0")) - 1.0, width))
        y1 = max(0.0, min(float(box.findtext("ymin", "0")) - 1.0, height))
        x2 = max(0.0, min(float(box.findtext("xmax", "0")), width))
        y2 = max(0.0, min(float(box.findtext("ymax", "0")), height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box in {path}")
        cx = (x1 + x2) / (2 * width)
        cy = (y1 + y2) / (2 * height)
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        labels.append(f"{CLASS_TO_ID[name]} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    return labels, ignored


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image IDs in {path}")
    return ids


def prepare(source: Path, output: Path, seed: int, val_fraction: float) -> dict:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    trainval_ids = read_ids(source / "ImageSets" / "Main" / "trainval.txt")
    test_ids = read_ids(source / "ImageSets" / "Main" / "test.txt")
    if set(trainval_ids) & set(test_ids):
        raise ValueError("VOC trainval and test sets overlap")
    random.Random(seed).shuffle(trainval_ids)
    val_count = round(len(trainval_ids) * val_fraction)
    splits = {
        "train": trainval_ids[val_count:],
        "val": trainval_ids[:val_count],
        "test": test_ids,
    }

    counts = Counter()
    ignored = 0
    for split, image_ids in splits.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            source_image = source / "JPEGImages" / f"{image_id}.jpg"
            target_image = image_dir / source_image.name
            if not target_image.exists():
                try:
                    os.link(source_image, target_image)
                except OSError:
                    shutil.copy2(source_image, target_image)
            labels, skipped = parse_annotation(source / "Annotations" / f"{image_id}.xml")
            (label_dir / f"{image_id}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
            )
            ignored += skipped
            counts[split] += len(labels)

    dataset_yaml = (
        f"path: '{output.resolve().as_posix()}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: {json.dumps(list(CLASSES))}\n"
    )
    (output / "data.yaml").write_text(dataset_yaml, encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    summary = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "seed": seed,
        "val_fraction": val_fraction,
        "images": {split: len(ids) for split, ids in splits.items()},
        "objects": dict(counts),
        "difficult_objects_excluded": ignored,
    }
    (output / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/voc2007/VOCdevkit/VOC2007"))
    parser.add_argument("--output", type=Path, default=Path("data/voc2007/yolo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.seed, args.val_fraction), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
