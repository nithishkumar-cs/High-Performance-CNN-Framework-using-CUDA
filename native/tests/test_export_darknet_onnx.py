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

    def test_shortcut_layer(self):
        """A residual (shortcut) block, as used by the full YOLOv3/Darknet-53 graph, exports as Add+activation."""
        cfg_text = """[net]
channels=3
height=4
width=4
[convolutional]
filters=4
size=1
stride=1
pad=1
activation=leaky
[convolutional]
filters=4
size=1
stride=1
pad=1
activation=leaky
[shortcut]
from=-2
activation=linear
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, weights, model = (root / name for name in ("shortcut.cfg", "shortcut.weights", "shortcut.onnx"))
            cfg.write_text(cfg_text, encoding="utf-8")
            float_count = (4 + 4 * 3) + (4 + 4 * 4) + (6 + 6 * 4)
            payload = struct.pack("<iiiQ", 0, 2, 0, 1) + np.full(float_count, 0.05, dtype="<f4").tobytes()
            weights.write_bytes(payload)
            result = export(cfg, weights, model)
            self.assertEqual(result["heads"], [[1, 6, 4, 4]])
            session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            heads = session.run(None, {"images": np.ones((1, 3, 4, 4), dtype=np.float32)})
            self.assertEqual([list(head.shape) for head in heads], result["heads"])
            self.assertTrue(np.isfinite(heads[0]).all())

    def test_route_passthrough(self):
        """A standalone passthrough route (jumping to a layer with a DIFFERENT spatial shape than
        the immediately-preceding layer, Darknet's standard YOLOv2 `[route] layers=-9` pattern)
        must not be rejected by comparing against the wrong reference shape (the bug independently
        found and fixed in network.cpp's layer_output_shape, req-31/req-37, and now also fixed
        here)."""
        cfg_text = """[net]
channels=1
height=4
width=4
[convolutional]
filters=2
size=1
stride=1
pad=1
activation=linear
[maxpool]
size=2
stride=2
[route]
layers=-2
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, weights, model = (root / name for name in ("route.cfg", "route.weights", "route.onnx"))
            cfg.write_text(cfg_text, encoding="utf-8")
            float_count = (2 + 2 * 1) + (6 + 6 * 2)
            payload = struct.pack("<iiiQ", 0, 2, 0, 1) + np.full(float_count, 0.05, dtype="<f4").tobytes()
            weights.write_bytes(payload)
            result = export(cfg, weights, model)
            self.assertEqual(result["heads"], [[1, 6, 4, 4]])
            session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            heads = session.run(None, {"images": np.ones((1, 1, 4, 4), dtype=np.float32)})
            self.assertEqual([list(head.shape) for head in heads], result["heads"])
            self.assertTrue(np.isfinite(heads[0]).all())

    def test_reorg_layer(self):
        """A [reorg] (YOLOv2 passthrough space-to-depth) layer exports and produces a finite,
        correctly-shaped output; network.cpp's smoke.cpp already covers this project's own exact
        channel/spatial mapping with a hand-derived check, so this only exercises the ONNX path."""
        cfg_text = """[net]
channels=1
height=4
width=4
[reorg]
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
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, weights, model = (root / name for name in ("reorg.cfg", "reorg.weights", "reorg.onnx"))
            cfg.write_text(cfg_text, encoding="utf-8")
            float_count = 6 + 6 * 4
            payload = struct.pack("<iiiQ", 0, 2, 0, 1) + np.full(float_count, 0.05, dtype="<f4").tobytes()
            weights.write_bytes(payload)
            result = export(cfg, weights, model)
            self.assertEqual(result["heads"], [[1, 6, 2, 2]])
            session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            heads = session.run(None, {"images": np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)})
            self.assertEqual([list(head.shape) for head in heads], result["heads"])
            self.assertTrue(np.isfinite(heads[0]).all())

    def test_region_layer(self):
        """A YOLOv2-style [region] head exports as a raw NCHW output, same as [yolo]."""
        cfg_text = """[net]
channels=3
height=4
width=4
[convolutional]
filters=6
size=1
stride=1
pad=1
activation=linear
[region]
classes=1
num=1
anchors=1,1
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, weights, model = (root / name for name in ("region.cfg", "region.weights", "region.onnx"))
            cfg.write_text(cfg_text, encoding="utf-8")
            float_count = 6 + 6 * 3
            payload = struct.pack("<iiiQ", 0, 2, 0, 1) + np.full(float_count, 0.05, dtype="<f4").tobytes()
            weights.write_bytes(payload)
            result = export(cfg, weights, model)
            self.assertEqual(result["heads"], [[1, 6, 4, 4]])
            session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            heads = session.run(None, {"images": np.ones((1, 3, 4, 4), dtype=np.float32)})
            self.assertEqual([list(head.shape) for head in heads], result["heads"])
            self.assertTrue(np.isfinite(heads[0]).all())


if __name__ == "__main__":
    unittest.main()
