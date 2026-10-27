"""Exercise the byte-level char-sequence CLI (rnn_text): train on a tiny repeating corpus until
next-byte prediction is perfect, then confirm generate() reproduces the learned pattern."""
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


def main() -> None:
    executable = pathlib.Path(sys.argv[1])
    config = pathlib.Path(sys.argv[2])
    with tempfile.TemporaryDirectory(prefix="cuda-framework-rnn-text-") as temporary:
        directory = pathlib.Path(temporary)
        corpus = directory / "corpus.txt"
        weights = directory / "trained.weights"
        corpus.write_text("ab" * 40, encoding="utf-8")
        train = invoke(
            executable, "train", config, corpus, weights,
            "--epochs", 60, "--time-steps", 16, "--lr", 0.1, "--seed", 7,
        )
        if "epoch=60" not in train or "next_byte_accuracy=1.000000" not in train.splitlines()[-1]:
            raise AssertionError(f"rnn_text training did not converge: {train.splitlines()[-1]}")
        if not weights.is_file():
            raise AssertionError("rnn_text training did not save a checkpoint")
        generated = invoke(
            executable, "generate", config, weights,
            "--seed-text", "a", "--length", "30", "--temperature", "0.3", "--seed", "3",
        ).strip()
        if generated != "a" + "ba" * 15:
            raise AssertionError(f"rnn_text generate did not reproduce the learned pattern: {generated!r}")
        print(f"train_ok generated={generated!r}")


if __name__ == "__main__":
    main()
