"""ONNX export smoke test for the supported Darknet layer types."""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from export_darknet_onnx import export  # noqa: E402


CFG = """[net]
channels=3
height=4
width=4
[convolutional]
batch_normalize=1
filters=6
size=1
stride=1
pad=1
activation=leaky
[maxpool]
size=2
stride=2
[convolutional]
filters=6
size=1
stride=1
pad=1
activation=linear
[yolo]
classes=1
mask=0
anchors=1,1
[upsample]
stride=2
[route]
layers=-1,0
[convolutional]
filters=6
size=1
stride=1
pad=1
activation=linear
[yolo]
classes=1
mask=0
anchors=1,1
"""


class ExportDarknetOnnxTest(unittest.TestCase):
    def test_all_supported_layers_and_weight_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, weights, model = (root / name for name in ("tiny.cfg", "tiny.weights", "tiny.onnx"))
            cfg.write_text(CFG, encoding="utf-8")
            payload = struct.pack("<iiiQ", 0, 2, 0, 64) + np.full(162, 0.1, dtype="<f4").tobytes()
            weights.write_bytes(payload)
            result = export(cfg, weights, model)
            self.assertEqual(result["heads"], [[1, 6, 2, 2], [1, 6, 4, 4]])
            self.assertEqual(result["images_seen"], 64)
            session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            heads = session.run(None, {"images": np.ones((1, 3, 4, 4), dtype=np.float32)})
            self.assertEqual([list(head.shape) for head in heads], result["heads"])
            self.assertTrue(all(np.isfinite(head).all() for head in heads))
            with self.assertRaises(FileExistsError):
                export(cfg, weights, model)
            weights.write_bytes(payload[:-4])
            with self.assertRaises(ValueError):
                export(cfg, weights, root / "short.onnx")
            weights.write_bytes(payload + struct.pack("<f", 1.0))
            with self.assertRaises(ValueError):
                export(cfg, weights, root / "long.onnx")


if __name__ == "__main__":
    unittest.main()
