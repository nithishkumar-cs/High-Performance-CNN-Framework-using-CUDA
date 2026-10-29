"""Exercise semantic and per-instance mask training/evaluation with real CLI checkpoints."""

import array
import pathlib
import subprocess
import sys
import tempfile


def run(*args: object) -> str:
    """Run a CLI and expose its output when a segmentation check fails."""
    result = subprocess.run([str(arg) for arg in args], capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(f"command failed: {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def write_records(path: pathlib.Path, records: list[list[float]]) -> None:
    """Write little-endian float32 input/target records for the two tiny cfgs."""
    with path.open("wb") as file:
        for record in records:
            values = array.array("f", record)
            if sys.byteorder != "little":
                values.byteswap()
            values.tofile(file)


def main() -> None:
    """Train both heads and require their held-out synthetic metrics to converge."""
    segmenter, instance_segmenter, semantic_cfg, instance_cfg = sys.argv[1:5]
    backend = ["--backend", "cuda"] if "--cuda" in sys.argv[5:] else []
    segmenter = str(pathlib.Path(segmenter).resolve())
    instance_segmenter = str(pathlib.Path(instance_segmenter).resolve())
    semantic_cfg = str(pathlib.Path(semantic_cfg).resolve())
    instance_cfg = str(pathlib.Path(instance_cfg).resolve())
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        semantic_data = root / "semantic.vecbin"
        semantic_input = [1.0, 0.0, 1.0, 0.0]
        semantic_target = [1.0, 0.0, 1.0, 0.0]
        write_records(semantic_data, [semantic_input + semantic_target] * 8)
        semantic_weights = root / "semantic.weights"
        run(segmenter, "train", semantic_cfg, semantic_weights, semantic_data,
            "--epochs", 30, "--lr", 0.2, *backend)
        semantic_eval = run(segmenter, "eval", semantic_cfg, semantic_weights, semantic_data)
        if "pixel_accuracy=1.000000" not in semantic_eval or "mean_iou=1.000000" not in semantic_eval:
            raise AssertionError(semantic_eval)
        ignored_data = root / "semantic-void.vecbin"
        write_records(ignored_data, [semantic_input + [1.0, -1.0, 1.0, 0.0]])
        ignored_eval = run(segmenter, "eval", semantic_cfg, semantic_weights, ignored_data)
        if "pixel_accuracy=1.000000" not in ignored_eval or "mean_iou=1.000000" not in ignored_eval:
            raise AssertionError(ignored_eval)

        instance_data = root / "instance.vecbin"
        red = [0.0, 0.0, 1.0, 1.0] * 4
        input_chw = red + [0.0] * 32
        write_records(instance_data, [input_chw + red] * 8)
        instance_weights = root / "instance.weights"
        run(instance_segmenter, "train", instance_cfg, instance_weights, instance_data,
            "--epochs", 40, "--lr", 0.2, *backend)
        instance_eval = run(instance_segmenter, "eval", instance_cfg, instance_weights, instance_data)
        if "mean_mask_iou=1.000000" not in instance_eval:
            raise AssertionError(instance_eval)


if __name__ == "__main__":
    main()
