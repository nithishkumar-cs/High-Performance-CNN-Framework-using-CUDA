"""Exercise IDX classifier training, checkpoint loading, and evaluation via the CLI."""
from __future__ import annotations

import pathlib
import struct
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


def main() -> None:
    executable = pathlib.Path(sys.argv[1])
    config = pathlib.Path(sys.argv[2])
    cuda = len(sys.argv) > 3 and sys.argv[3] == "--cuda"
    with tempfile.TemporaryDirectory(prefix="cuda-framework-cli-") as temporary:
        directory = pathlib.Path(temporary)
        images = directory / "images.idx3-ubyte"
        labels = directory / "labels.idx1-ubyte"
        weights = directory / "trained.weights"
        count = 40
        images.write_bytes(
            struct.pack(">IIII", 2051, count, 4, 4)
            + b"".join(bytes([255 if i % 2 else 0] * 16) for i in range(count))
        )
        labels.write_bytes(
            struct.pack(">II", 2049, count) + bytes(i % 2 for i in range(count))
        )
        train = invoke(
            executable, "train", config, images, labels, weights,
            "--backend", "cuda" if cuda else "cpu", "--epochs", 8, "--lr", 0.05,
        )
        if "epoch=8" not in train or not weights.is_file():
            raise AssertionError("classifier CLI training did not save a checkpoint")
        evaluate = invoke(
            executable, "eval", config, images, labels, weights,
            "--backend", "cuda" if cuda else "cpu",
        )
        if "samples=40 correct=40" not in evaluate:
            raise AssertionError(f"classifier CLI did not learn or reload: {evaluate}")
        print(evaluate.strip())


if __name__ == "__main__":
    main()

