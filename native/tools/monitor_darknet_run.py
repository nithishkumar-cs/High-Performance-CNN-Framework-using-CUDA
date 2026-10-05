"""Snapshot and evaluate exact Darknet training milestones without stopping training.

This is for an external pjreddie/darknet GPU trainer and this project's native
evaluator. It does not implement training itself. Artifacts stay outside Git.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path


def images_seen(path: Path) -> int | None:
    """Read the Darknet weights header's images-seen counter; returns None if the file doesn't exist yet or is too short."""
    if not path.is_file() or path.stat().st_size < 20:
        return None
    with path.open("rb") as handle:
        major, minor, revision, seen = struct.unpack("<iiiQ", handle.read(20))
    if (major, minor, revision) != (0, 2, 0):
        raise ValueError(f"unexpected Darknet weight header in {path}")
    return seen


def wait_for_weights(path: Path, seen: int, expected_bytes: int, deadline: float,
                     poll_seconds: float, train_pid: int) -> None:
    """Poll until the checkpoint at `path` settles at exactly `seen` images and `expected_bytes`.

    Raises RuntimeError if a later checkpoint is seen first or the trainer process exits early,
    and TimeoutError if `deadline` (monotonic time) passes first.
    """
    while time.monotonic() < deadline:
        if path.is_file():
            current = images_seen(path)
            if current is not None and current > seen:
                raise RuntimeError(f"missed exact checkpoint: wanted {seen}, found {current}")
            if current == seen and path.stat().st_size == expected_bytes:
                time.sleep(2)
                if images_seen(path) == seen and path.stat().st_size == expected_bytes:
                    return
        if train_pid:
            try:
                os.kill(train_pid, 0)
            except ProcessLookupError as error:
                raise RuntimeError("trainer exited before expected checkpoint") from error
        time.sleep(poll_seconds)
    raise TimeoutError(f"checkpoint did not reach {seen} images: {path}")


def snapshot_exact(source: Path, destination: Path, seen: int, expected_bytes: int) -> None:
    """Atomically copy `source` to `destination` only if it is unchanged and matches `seen`/`expected_bytes` throughout the copy."""
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    before = source.stat()
    shutil.copyfile(source, partial)
    after = source.stat()
    if (before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size or
            partial.stat().st_size != expected_bytes or images_seen(partial) != seen):
        partial.unlink()
        raise RuntimeError(f"checkpoint changed while copying: {source}")
    os.replace(partial, destination)


def evaluate(evaluator: Path, cfg: Path, weights: Path, images: Path, labels: Path,
             output: Path) -> str:
    """Run the native evaluator CLI on a checkpoint and return its single mAP50_present_classes= line; raises on missing/duplicate result."""
    with output.open("x", encoding="utf-8") as handle:
        subprocess.run([str(evaluator), str(cfg), str(weights), str(images), str(labels),
                        "--backend", "cuda", "--confidence", "0.01"],
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    results = [line for line in output.read_text(encoding="utf-8").splitlines()
               if line.startswith("mAP50_present_classes=")]
    if len(results) != 1:
        raise RuntimeError(f"missing or duplicate mAP50 result in {output}")
    return results[0]


def main() -> None:
    """CLI entry point: watch a running external Darknet training, snapshot exact-batch checkpoints, and evaluate each."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("backup", "final", "size-reference", "cfg", "images", "labels",
                 "evaluator", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--milestones", type=int, nargs="+", default=[5000, 8000, 9000])
    parser.add_argument("--final-step", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--timeout-hours", type=float, default=6)
    args = parser.parse_args()
    if (args.final_step <= 0 or args.batch_size <= 0 or args.poll_seconds <= 0 or
            args.timeout_hours <= 0 or any(step <= 0 or step >= args.final_step
                                       for step in args.milestones) or
            sorted(set(args.milestones)) != args.milestones):
        parser.error("invalid milestone schedule")
    expected_bytes = args.size_reference.stat().st_size
    if expected_bytes <= 20 or not args.output_dir.is_dir():
        parser.error("missing size reference or output directory")
    deadline = time.monotonic() + args.timeout_hours * 3600
    for step in args.milestones:
        wanted = step * args.batch_size
        wait_for_weights(args.backup, wanted, expected_bytes, deadline,
                         args.poll_seconds, args.train_pid)
        snapshot = args.output_dir / f"milestone-{step}.weights"
        snapshot_exact(args.backup, snapshot, wanted, expected_bytes)
        metric = evaluate(args.evaluator, args.cfg, snapshot, args.images, args.labels,
                          args.output_dir / f"val-{step}.log")
        print(f"batch={step} {metric} checkpoint={snapshot}", flush=True)
    wanted = args.final_step * args.batch_size
    wait_for_weights(args.final, wanted, expected_bytes, deadline,
                     args.poll_seconds, args.train_pid)
    metric = evaluate(args.evaluator, args.cfg, args.final, args.images, args.labels,
                      args.output_dir / "val-final.log")
    print(f"batch={args.final_step} {metric} checkpoint={args.final}", flush=True)


if __name__ == "__main__":
    main()
