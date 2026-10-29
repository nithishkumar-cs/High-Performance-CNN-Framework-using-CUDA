"""Build semantic and per-instance TSV manifests from a VOC2007 segmentation split.

The resulting manifests feed pack_segmentation.py. VOC's 255 semantic value is
void; pass --ignore-value 255 when packing semantic records. Object-mask value
255 is also void and is excluded when enumerating instance IDs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def relative(path: Path, directory: Path) -> str:
    """Use a manifest-relative path so the VOC tree can move with its manifests."""
    return os.path.relpath(path, directory)


def prepare(root: Path, split: str, directory: Path) -> tuple[int, int]:
    """Write aligned semantic pairs and one box/ID row for each visible object region."""
    ids_path = root / "ImageSets" / "Segmentation" / f"{split}.txt"
    image_ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if not image_ids:
        raise ValueError(f"no image IDs in {ids_path}")
    semantic_path = directory / f"{split}-semantic.tsv"
    instance_path = directory / f"{split}-instance.tsv"
    if semantic_path.exists() or instance_path.exists():
        raise ValueError("refusing to overwrite existing VOC manifests")
    semantic_rows = []
    instance_rows = []
    for image_id in image_ids:
        image = root / "JPEGImages" / f"{image_id}.jpg"
        class_mask = root / "SegmentationClass" / f"{image_id}.png"
        object_mask = root / "SegmentationObject" / f"{image_id}.png"
        if not image.is_file() or not class_mask.is_file() or not object_mask.is_file():
            raise ValueError(f"missing VOC image or segmentation mask for {image_id}")
        semantic_rows.append(f"{relative(image, directory)}\t{relative(class_mask, directory)}\n")
        ids = np.asarray(Image.open(object_mask))
        for instance_id in np.unique(ids):
            instance_id = int(instance_id)
            if instance_id in (0, 255):
                continue
            ys, xs = np.where(ids == instance_id)
            if len(xs) == 0:
                continue
            bounds = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            instance_rows.append(
                f"{relative(image, directory)}\t{relative(object_mask, directory)}\t"
                f"{bounds[0]}\t{bounds[1]}\t{bounds[2]}\t{bounds[3]}\t{instance_id}\n"
            )
    directory.mkdir(parents=True, exist_ok=True)
    semantic_path.write_text("".join(semantic_rows), encoding="utf-8")
    instance_path.write_text("".join(instance_rows), encoding="utf-8")
    return len(semantic_rows), len(instance_rows)


def main() -> None:
    """Parse a VOC root/split and report manifest record counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("voc_root", type=Path, help="directory containing JPEGImages and SegmentationObject")
    parser.add_argument("split", choices=("train", "val", "trainval"))
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    semantic, instances = prepare(args.voc_root, args.split, args.output_dir)
    print(f"semantic_images={semantic} instance_regions={instances} output={args.output_dir}")


if __name__ == "__main__":
    main()
