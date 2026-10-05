"""Convert official VOC2007 XML to the YOLO labels used by this project.

This is the project's own converter. It excludes VOC 'difficult' objects and
records that choice in preparation.json; the native evaluator does not have
VOC ignore-region behavior. Use an empty output directory to avoid clobbering
an existing dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from prepare_darknet_voc import VOC_NAMES


CLASS_IDS = {name: index for index, name in enumerate(VOC_NAMES)}


def read_ids(path: Path) -> list[str]:
    """Read a VOC ImageSet file into a list of non-empty, unique image IDs; raises ValueError otherwise."""
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
           if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"empty or duplicate image IDs in {path}")
    return ids


def split_ids(source: Path, seed: int = 42, val_fraction: float = 0.10) -> dict[str, list[str]]:
    """Deterministically split VOC trainval into train/val (seeded shuffle) and keep the official test set separate."""
    if not 0 < val_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    trainval = read_ids(source / "ImageSets" / "Main" / "trainval.txt")
    test = read_ids(source / "ImageSets" / "Main" / "test.txt")
    if set(trainval) & set(test):
        raise ValueError("VOC trainval and test IDs overlap")
    random.Random(seed).shuffle(trainval)
    val_count = round(len(trainval) * val_fraction)
    if not 0 < val_count < len(trainval):
        raise ValueError("validation split would be empty or consume training set")
    return {"train": trainval[val_count:], "val": trainval[:val_count], "test": test}


def parse_annotation(path: Path) -> tuple[list[str], int]:
    """Parse one VOC XML annotation into YOLO-format label lines, excluding 'difficult' objects; returns (labels, excluded_count)."""
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing image size in {path}")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in {path}")
    labels: list[str] = []
    ignored = 0
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_IDS:
            raise ValueError(f"unknown VOC class {name!r} in {path}")
        if obj.findtext("difficult", "0") == "1":
            ignored += 1
            continue
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"missing bounding box in {path}")
        # VOC coordinates are one-based and inclusive.
        x1 = max(0.0, min(float(box.findtext("xmin", "0")) - 1.0, width))
        y1 = max(0.0, min(float(box.findtext("ymin", "0")) - 1.0, height))
        x2 = max(0.0, min(float(box.findtext("xmax", "0")), width))
        y2 = max(0.0, min(float(box.findtext("ymax", "0")), height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid bounding box in {path}")
        cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        labels.append(f"{CLASS_IDS[name]} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    return labels, ignored


def prepare(source: Path, output: Path, seed: int = 42,
            val_fraction: float = 0.10) -> dict:
    """Convert official VOC2007 into the project's YOLO dataset layout; raises ValueError if `output` is non-empty."""
    splits = split_ids(source, seed, val_fraction)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    ignored = 0
    for split, ids in splits.items():
        images = output / "images" / split
        labels = output / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for image_id in ids:
            source_image = source / "JPEGImages" / f"{image_id}.jpg"
            source_xml = source / "Annotations" / f"{image_id}.xml"
            lines, skipped = parse_annotation(source_xml)
            target_image = images / source_image.name
            try:
                os.link(source_image, target_image)
            except OSError:
                shutil.copy2(source_image, target_image)
            (labels / f"{image_id}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts[split] += len(lines)
            ignored += skipped
    (output / "classes.txt").write_text("\n".join(VOC_NAMES) + "\n", encoding="utf-8")
    summary = {
        "source": str(source.resolve()), "output": str(output.resolve()),
        "seed": seed, "val_fraction": val_fraction,
        "images": {split: len(ids) for split, ids in splits.items()},
        "objects": dict(counts), "difficult_objects_excluded": ignored,
    }
    (output / "preparation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def verify_existing(source: Path, output: Path, seed: int = 42,
                    val_fraction: float = 0.10) -> dict:
    """Check every converted image/label against the source XML without writes."""
    splits = split_ids(source, seed, val_fraction)
    counts: Counter[str] = Counter()
    ignored = 0
    for split, ids in splits.items():
        images = output / "images" / split
        labels = output / "labels" / split
        expected = set(ids)
        actual_images = {path.stem for path in images.glob("*.jpg")}
        actual_labels = {path.stem for path in labels.glob("*.txt")}
        if actual_images != expected or actual_labels != expected:
            raise ValueError(f"image/label IDs disagree in {split}")
        for image_id in ids:
            lines, skipped = parse_annotation(source / "Annotations" / f"{image_id}.xml")
            expected_text = "\n".join(lines) + ("\n" if lines else "")
            label = labels / f"{image_id}.txt"
            if label.read_text(encoding="utf-8") != expected_text:
                raise ValueError(f"converted label disagrees with source XML: {label}")
            counts[split] += len(lines)
            ignored += skipped
    summary = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    if (summary["images"] != {split: len(ids) for split, ids in splits.items()} or
            summary["objects"] != dict(counts) or
            summary["difficult_objects_excluded"] != ignored or
            summary["seed"] != seed or summary["val_fraction"] != val_fraction):
        raise ValueError("preparation.json disagrees with source XML or split")
    return summary

def main() -> None:
    """CLI entry point: convert or verify a VOC2007 -> YOLO dataset conversion and print the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="official VOCdevkit/VOC2007 directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="new/empty YOLO dataset directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--verify-existing", action="store_true",
                        help="read-only comparison of an existing conversion with all source XML")
    args = parser.parse_args()
    action = verify_existing if args.verify_existing else prepare
    print(json.dumps(action(args.source, args.output, args.seed, args.val_fraction), indent=2))


if __name__ == "__main__":
    main()

