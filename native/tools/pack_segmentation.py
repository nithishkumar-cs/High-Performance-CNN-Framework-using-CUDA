"""Pack paired image/mask manifests into this project's float32 vector record format.

Requires Pillow and NumPy. Paths in the tab-separated manifest are relative to that
manifest. Semantic rows: image<TAB>class-index-mask. Instance rows:
image<TAB>instance-ID-mask<TAB>x0<TAB>y0<TAB>x1<TAB>y1<TAB>instance_id. Instance boxes
are integer pixel bounds [x0,y0,x1,y1); each row is one object, even when
multiple objects share an image. Masks must be single-channel index images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def manifest_rows(path: Path, fields: int):
    """Yield validated tab-separated rows with image paths resolved against the manifest."""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        cells = line.split("\t")
        if len(cells) != fields:
            raise ValueError(f"{path}:{number}: expected {fields} tab-separated fields")
        yield number, cells


def read_pair(manifest: Path, image_name: str, mask_name: str):
    """Load one RGB image and its same-size single-channel integer mask."""
    image = Image.open(manifest.parent / image_name).convert("RGB")
    mask = Image.open(manifest.parent / mask_name)
    if mask.mode not in {"L", "P", "I", "I;16", "1"}:
        raise ValueError(f"mask must hold one integer class/instance value per pixel: {mask_name}")
    if image.size != mask.size:
        raise ValueError(f"image and mask dimensions differ: {image_name}, {mask_name}")
    return image, mask


def semantic_record(image: Image.Image, mask: Image.Image, width: int, height: int,
                    classes: int, ignore_value: int | None) -> np.ndarray:
    """Resize an aligned class map; encode optional void pixels as target -1."""
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    mask = mask.resize((width, height), Image.Resampling.NEAREST)
    target = np.asarray(mask, dtype=np.int32)
    if ignore_value is not None:
        target = np.where(target == ignore_value, -1, target)
    if np.any(target < -1) or np.any(target >= classes):
        raise ValueError("semantic mask has a class ID outside [0, classes)")
    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).ravel() / 255.0
    return np.concatenate((pixels, target.ravel().astype(np.float32))).astype("<f4")


def instance_record(image: Image.Image, mask: Image.Image, bounds: tuple[int, ...],
                    instance_id: int, width: int, height: int) -> np.ndarray:
    """Crop one object's ROI and letterbox RGB and its selected instance ID mask."""
    x0, y0, x1, y1 = bounds
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError("instance box must be a nonempty in-image [x0,y0,x1,y1) rectangle")
    if instance_id < 1 or instance_id > 255:
        raise ValueError("instance_id must be in [1,255]")
    roi = image.crop(bounds)
    binary = mask.crop(bounds)
    scale = min(width / roi.width, height / roi.height)
    resized_width = min(width, max(1, int(np.floor(roi.width * scale + 0.5))))
    resized_height = min(height, max(1, int(np.floor(roi.height * scale + 0.5))))
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    roi = roi.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    binary = binary.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    pixels = np.full((3, height, width), 0.5, dtype=np.float32)
    pixels[:, top:top + resized_height, left:left + resized_width] = (
        np.asarray(roi, dtype=np.float32).transpose(2, 0, 1) / 255.0
    )
    targets = np.zeros((height, width), dtype=np.float32)
    targets[top:top + resized_height, left:left + resized_width] = (
        np.asarray(binary) == instance_id
    ).astype(np.float32)
    return np.concatenate((pixels.ravel(), targets.ravel())).astype("<f4")


def main() -> None:
    """Write one fixed-size vecbin record per manifest row without buffering the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("semantic", "instance"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--classes", type=int, help="semantic class count, including background")
    parser.add_argument("--ignore-value", type=int,
                        help="semantic input mask value to encode as target -1 (VOC uses 255)")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    if args.mode == "semantic" and (args.classes is None or args.classes < 2):
        parser.error("semantic mode requires --classes >= 2")
    count = 0
    try:
        with args.output.open("xb") as output:
            for number, cells in manifest_rows(args.manifest, 2 if args.mode == "semantic" else 7):
                image, mask = read_pair(args.manifest, cells[0], cells[1])
                try:
                    if args.mode == "semantic":
                        record = semantic_record(image, mask, args.width, args.height,
                                                 args.classes, args.ignore_value)
                    else:
                        bounds = tuple(int(value) for value in cells[2:6])
                        record = instance_record(image, mask, bounds, int(cells[6]),
                                                 args.width, args.height)
                except ValueError as error:
                    raise ValueError(f"{args.manifest}:{number}: {error}") from error
                output.write(record.tobytes())
                count += 1
    except FileExistsError:
        parser.error("refusing to overwrite existing output")
    except Exception:
        args.output.unlink(missing_ok=True)
        raise
    if not count:
        args.output.unlink()
        parser.error("manifest contains no records")
    print(f"records={count} output={args.output}")


if __name__ == "__main__":
    main()
