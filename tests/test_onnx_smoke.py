import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.yolo_inference import Detector, make_session

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:
    onnx = None


@unittest.skipUnless(onnx is not None, "onnx package is needed to build the fixture")
class OnnxSmokeTest(unittest.TestCase):
    def test_cpu_session_runs_decoded_v8_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "constant_detector.onnx"
            values = helper.make_tensor(
                "detections", TensorProto.FLOAT, [1, 5, 1],
                [50, 50, 40, 20, 0.9],
            )
            node = helper.make_node("Constant", [], ["output"], value=values)
            graph = helper.make_graph(
                [node], "smoke",
                [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 100, 100])],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 1])],
            )
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 13)]
            )
            model.ir_version = 8
            onnx.save(model, model_path)

            session = make_session(model_path, "cpu")
            detections, elapsed_ms = Detector(session, "v8", size=100).detect(
                Image.new("RGB", (100, 50), "white")
            )
            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0]["box"], [30, 15, 70, 35])
            self.assertGreaterEqual(elapsed_ms, 0)

            image_path = Path(temp_dir) / "sample.png"
            report_path = Path(temp_dir) / "benchmark.json"
            Image.new("RGB", (100, 50), "white").save(image_path)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "tools" / "benchmark_jetson.py"),
                    "--model", str(model_path),
                    "--image", str(image_path),
                    "--format", "v8",
                    "--size", "100",
                    "--runs", "2",
                    "--warmup", "1",
                    "--output", str(report_path),
                ],
                check=True, capture_output=True, text=True,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["runs"], 2)
            self.assertEqual(report["detections"], 1)
            self.assertGreater(report["end_to_end_ms"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
