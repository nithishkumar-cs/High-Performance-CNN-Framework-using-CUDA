"""Exercise the detector CLI's crop/hue/saturation/exposure/flip/multiscale augmentation pipeline
end to end: writes tiny synthetic images+labels, trains a few steps with every jitter flag (and
Darknet-style multiscale resize) enabled, and checks training completes with a finite loss and a
saved checkpoint (a regression test for the crop/HSV augmentation added alongside ignore_thresh,
and for the later --multiscale-min/--multiscale-max/--multiscale-interval resize-jitter support -
it does not assert accuracy, only that the augmented/resizing data pipeline runs end to end and
keeps producing valid, finite training signal)."""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile


def invoke(*arguments: object) -> str:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run(
        [str(argument) for argument in arguments],
        capture_output=True,
        text=True,
        check=True,
        creationflags=flags,
    )
    return result.stdout


def write_ppm(path: pathlib.Path, width: int, height: int) -> None:
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    # A simple horizontal gradient so hue/saturation jitter has non-trivial color to perturb.
    pixels = bytes(((x * 255) // max(1, width - 1), 128, 255 - (x * 255) // max(1, width - 1))[c]
                    for y in range(height) for x in range(width) for c in range(3))
    path.write_bytes(header + pixels)


def main() -> None:
    executable = pathlib.Path(sys.argv[1])
    config = pathlib.Path(sys.argv[2])
    with tempfile.TemporaryDirectory(prefix="cuda-framework-aug-") as temporary:
        directory = pathlib.Path(temporary)
        images_dir, labels_dir = directory / "images", directory / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        for name in ("a", "b"):
            write_ppm(images_dir / f"{name}.ppm", 20, 20)
            (labels_dir / f"{name}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        weights = directory / "trained.weights"
        output = invoke(
            executable, config, images_dir, labels_dir, weights,
            "--epochs", 2, "--lr", 0.001, "--batch", 1,
            "--flip-probability", 0.5, "--exposure-jitter", 0.3,
            "--saturation-jitter", 0.3, "--hue-jitter", 0.1, "--crop-jitter", 0.2,
            # mini.cfg's one maxpool (stride 2) and one upsample (stride 2) cancel at any even
            # size, so resizing within [32,64] stays structurally valid; interval=1 forces a
            # resize after every single-image batch so both sizes are actually exercised.
            "--multiscale-min", 32, "--multiscale-max", 64, "--multiscale-interval", 1,
        )
        if "epoch=2" not in output or not weights.is_file():
            raise AssertionError(f"augmented training did not complete: {output}")
        for line in output.splitlines():
            if line.startswith("epoch=") and "mean_loss=nan" in line:
                raise AssertionError(f"augmented training produced a non-finite loss: {line}")
        print(output.strip())


if __name__ == "__main__":
    main()
