"""Convert a directory of JPEG/PNG images to binary PPM (P6).

The native image reader only supports PPM unless the build was configured with
-DDETECTOR_OPENCV=ON (see docs/native-detector.md). This build does not have OpenCV
wired up, so cuda_detector_train/cuda_detector_eval need a PPM copy of any JPEG dataset
(e.g. the VOC2007 YOLO conversion's images/{train,val,test}); this is that one-time copy
step, not part of the tracked dataset pipeline (output stays under runs/, which is
git-ignored).
"""
import argparse
from pathlib import Path

from PIL import Image


def convert(source: Path, output: Path) -> int:
    """Convert every .jpg/.jpeg/.png in `source` to .ppm in `output` (created if needed).

    Returns the number of images converted. Raises FileNotFoundError if `source` does
    not exist.
    """
    if not source.is_dir():
        raise FileNotFoundError(f"not a directory: {source}")
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        Image.open(path).convert("RGB").save(output / (path.stem + ".ppm"))
        count += 1
    return count


def main() -> None:
    """CLI entry point: convert `--source` images to PPM under `--output` and print the count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="directory of JPEG/PNG images")
    parser.add_argument("--output", type=Path, required=True, help="directory to write .ppm copies into")
    args = parser.parse_args()
    print(f"converted {convert(args.source, args.output)} images from {args.source} to {args.output}")


if __name__ == "__main__":
    main()
