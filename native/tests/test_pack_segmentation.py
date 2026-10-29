"""Check semantic/instance vecbin packing and cleanup on invalid masks."""

import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


def invoke(script: str, *args: object, success: bool = True) -> None:
    """Run the packer and assert whether the input manifest should be accepted."""
    result = subprocess.run([sys.executable, script, *(str(arg) for arg in args)],
                            capture_output=True, text=True)
    if (result.returncode == 0) != success:
        raise AssertionError(f"unexpected packer result: {result.stdout}\n{result.stderr}")


def main() -> None:
    """Check class-ID layout, ROI letterboxing, and rejection of out-of-range labels."""
    script = sys.argv[1]
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        rgb = np.zeros((2, 4, 3), dtype=np.uint8)
        rgb[:, 2:, 0] = 255
        mask = np.zeros((2, 4), dtype=np.uint8)
        mask[:, 2:] = 1
        Image.fromarray(rgb).save(root / "image.png")
        Image.fromarray(mask).save(root / "mask.png")
        semantic_manifest = root / "semantic.tsv"
        semantic_manifest.write_text("image.png\tmask.png\n", encoding="utf-8")
        semantic_data = root / "semantic.vecbin"
        invoke(script, "semantic", semantic_manifest, semantic_data,
               "--width", 4, "--height", 2, "--classes", 2)
        semantic = np.fromfile(semantic_data, dtype="<f4")
        if semantic.size != 32 or not np.array_equal(semantic[-8:], mask.ravel()):
            raise AssertionError("semantic class IDs were not packed in HxW order")
        instance_manifest = root / "instance.tsv"
        instance_manifest.write_text("image.png\tmask.png\t0\t0\t4\t2\t1\n", encoding="utf-8")
        instance_data = root / "instance.vecbin"
        invoke(script, "instance", instance_manifest, instance_data,
               "--width", 4, "--height", 4)
        instance = np.fromfile(instance_data, dtype="<f4")
        expected = np.zeros((4, 4), dtype=np.float32)
        expected[1:3, 2:] = 1.0
        if instance.size != 64 or not np.array_equal(instance[-16:].reshape(4, 4), expected):
            raise AssertionError("instance ROI mask was not letterboxed with its image")
        bad = mask.copy()
        bad[0, 0] = 2
        Image.fromarray(bad).save(root / "bad.png")
        semantic_manifest.write_text("image.png\tbad.png\n", encoding="utf-8")
        bad_output = root / "bad.vecbin"
        invoke(script, "semantic", semantic_manifest, bad_output,
               "--width", 4, "--height", 2, "--classes", 2, success=False)
        if bad_output.exists():
            raise AssertionError("invalid pack left a partial dataset")
        void = mask.copy()
        void[0, 0] = 255
        Image.fromarray(void).save(root / "void.png")
        semantic_manifest.write_text("image.png\tvoid.png\n", encoding="utf-8")
        void_output = root / "void.vecbin"
        invoke(script, "semantic", semantic_manifest, void_output,
               "--width", 4, "--height", 2, "--classes", 2, "--ignore-value", 255)
        if np.fromfile(void_output, dtype="<f4")[-8] != -1:
            raise AssertionError("VOC void label was not encoded as ignored target")


if __name__ == "__main__":
    main()
