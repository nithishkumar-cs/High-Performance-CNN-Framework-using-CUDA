"""Check detector GPU selection and a real one-device CUDA training CLI run."""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile


def main() -> None:
    """Train a tiny batch on GPU 0 and reject a duplicate device list before training."""
    executable = pathlib.Path(sys.argv[1])
    config = pathlib.Path(sys.argv[2])
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with tempfile.TemporaryDirectory(prefix="cuda-framework-gpus-") as temporary:
        root = pathlib.Path(temporary)
        images, labels = root / "images", root / "labels"
        images.mkdir()
        labels.mkdir()
        for name in ("a", "b"):
            (images / f"{name}.ppm").write_bytes(b"P6\n8 8\n255\n" + bytes([127, 64, 192]) * 64)
            (labels / f"{name}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        output = root / "trained.weights"
        base = [str(executable), str(config), str(images), str(labels)]
        trained = subprocess.run(
            base + [str(output), "--backend", "cuda", "--gpus", "0", "--batch", "2",
                    "--epochs", "1", "--lr", "0.001"],
            capture_output=True, text=True, creationflags=flags,
        )
        if trained.returncode or not output.is_file() or "epoch=1" not in trained.stdout:
            raise AssertionError(f"single-GPU CLI training failed: {trained.stdout}\n{trained.stderr}")
        rejected = subprocess.run(
            base + [str(root / "invalid.weights"), "--backend", "cuda", "--gpus", "0,0",
                    "--batch", "2"],
            capture_output=True, text=True, creationflags=flags,
        )
        if rejected.returncode == 0 or "distinct CUDA device indices" not in rejected.stderr:
            raise AssertionError(f"duplicate GPUs were accepted: {rejected.stdout}\n{rejected.stderr}")
    print("detector CUDA GPU selection passed")


if __name__ == "__main__":
    main()
