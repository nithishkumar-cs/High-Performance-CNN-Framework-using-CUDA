import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

from tools.validate_yolo_dataset import check_split
from tools.yolo_inference import decode, letterbox


class DetectorTests(unittest.TestCase):
    def test_native_tensorrt_backend_dispatch(self):
        from tools.yolo_inference import make_session

        marker = object()
        with patch('tools.tensorrt_inference.TensorRTSession', return_value=marker) as session:
            result = make_session(Path('model.engine'), 'tensorrt-native')
        self.assertIs(result, marker)
        session.assert_called_once_with(Path('model.engine'))

    def test_v5_suppresses_same_class_only(self):
        output = np.array([[
            [50, 50, 40, 40, .9, .9, .1],
            [51, 51, 40, 40, .8, .8, .1],
            [50, 50, 40, 40, .9, .1, .9],
        ]], dtype=np.float32)
        detections = decode(output, "v5", (100, 100), 1, 0, 0)
        self.assertEqual(len(detections), 2)
        self.assertEqual({item["class_id"] for item in detections}, {0, 1})

    def test_v8_undoes_letterbox(self):
        image = Image.new("RGB", (100, 50), "white")
        tensor, scale, left, top = letterbox(image, 100)
        self.assertEqual(tensor.shape, (1, 3, 100, 100))
        self.assertEqual((scale, left, top), (1, 0, 25))
        output = np.array([[[50], [50], [40], [20], [.9]]], dtype=np.float32)
        detections = decode(output, "v8", image.size, scale, left, top)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["box"], [30, 15, 70, 35])

    def test_unlabeled_image_and_invalid_box(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train" / "negative.jpg").touch()
            (root / "images" / "train" / "positive.jpg").touch()
            (root / "labels" / "train" / "positive.txt").write_text(
                "0 0.5 0.5 nan 0\n", encoding="utf-8"
            )
            labels, errors = check_split(root, "train", 1)
            self.assertEqual(labels, 1)
            self.assertEqual(errors, 2)


if __name__ == "__main__":
    unittest.main()
