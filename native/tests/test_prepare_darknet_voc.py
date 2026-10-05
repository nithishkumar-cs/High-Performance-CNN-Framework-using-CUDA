"""Standard-library tests for the external Darknet training adapter."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_darknet_voc import adapt_cfg, images_and_labels  # noqa: E402


SYNTHETIC_CFG = """[net]
batch=1
subdivisions=1
max_batches=500200
[convolutional]
filters=255
[yolo]
classes=80
random=1
[convolutional]
filters=255
[yolo]
classes=80
random=1
"""


class PrepareDarknetVocTest(unittest.TestCase):
    def test_two_heads_and_training_schedule(self):
        result = adapt_cfg(SYNTHETIC_CFG, 10000)
        self.assertEqual(result.count("filters=75"), 2)
        self.assertEqual(result.count("classes=20"), 2)
        self.assertEqual(result.count("random=0"), 2)
        self.assertIn("batch=32", result)
        self.assertIn("subdivisions=16", result)
        self.assertIn("max_batches=10000", result)

    def test_refuses_unexpected_head(self):
        with self.assertRaises(ValueError):
            adapt_cfg(SYNTHETIC_CFG.replace("classes=80", "classes=19", 1), 10000)

    def test_validates_label_class_and_box(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            (images / "example.jpg").touch()
            label = labels / "example.txt"
            label.write_text("14 0.5 0.5 0.2 0.3\n", encoding="utf-8")
            self.assertEqual(len(images_and_labels(images, labels)), 1)
            label.write_text("20 0.5 0.5 0.2 0.3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                images_and_labels(images, labels)


if __name__ == "__main__":
    unittest.main()
