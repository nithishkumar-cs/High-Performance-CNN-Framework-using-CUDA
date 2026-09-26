import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from scratch_yolo.data import VocYoloDataset, collate_detection
from scratch_yolo.decode import decode_heads
from scratch_yolo.metrics import ap50
from scratch_yolo.numpy_decode import decode_numpy_heads
from scratch_yolo.model import TinyDetector
from scratch_yolo.objective import DetectionLoss


class ScratchYoloTests(unittest.TestCase):
    def test_two_head_forward_and_backward(self):
        model = TinyDetector(classes=2)
        heads = model(torch.rand(1, 3, 64, 64))
        self.assertEqual([tuple(head.shape) for head in heads],
                         [(1, 3, 7, 2, 2), (1, 3, 7, 4, 4)])
        objective = DetectionLoss(classes=2)
        targets = torch.tensor([[0, 1, .5, .5, .2, .3]])
        loss = objective(heads, targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.coarse_head.weight.grad)
        self.assertIsNotNone(model.fine_head.weight.grad)

    def test_decode_raw_head(self):
        coarse = torch.full((1, 3, 7, 2, 2), -20.0)
        fine = torch.full((1, 3, 7, 4, 4), -20.0)
        fine[0, 0, :4, 1, 1] = 0
        fine[0, 0, 4, 1, 1] = 10
        fine[0, 0, 6, 1, 1] = 10
        boxes = decode_heads((coarse, fine), threshold=.25)
        self.assertEqual(boxes[0].shape, (1, 6))
        self.assertEqual(int(boxes[0][0, 5]), 1)
        self.assertAlmostEqual(float(boxes[0][0, 4]), 1, places=3)
        numpy_boxes = decode_numpy_heads([coarse.numpy(), fine.numpy()], threshold=.25)
        np.testing.assert_allclose(numpy_boxes[0], boxes[0].numpy(), atol=1e-6)

    def test_letterbox_preserves_boxes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'images' / 'train').mkdir(parents=True)
            (root / 'labels' / 'train').mkdir(parents=True)
            Image.new('RGB', (100, 50), 'white').save(root / 'images' / 'train' / 'one.jpg')
            (root / 'labels' / 'train' / 'one.txt').write_text(
                '1 0.5 0.5 1 1\n', encoding='utf-8')
            dataset = VocYoloDataset(root, 'train', 64)
            image, labels = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 64, 64))
            self.assertTrue(torch.allclose(labels[0], torch.tensor([1, .5, .5, 1, .5])))
            batch, targets = collate_detection([(image, labels)])
            self.assertEqual(tuple(batch.shape), (1, 3, 64, 64))
            self.assertEqual(tuple(targets.shape), (1, 6))

    def test_fast_nms_matches_reference(self):
        coarse = torch.full((1, 3, 7, 2, 2), -20.0)
        fine = torch.full((1, 3, 7, 4, 4), -20.0)
        for anchor in (0, 1):
            fine[0, anchor, :4, 1, 1] = 0
            fine[0, anchor, 4, 1, 1] = 10 - anchor
            fine[0, anchor, 5, 1, 1] = 10
        fine[0, 1, 2, 1, 1] = float(np.log(10 / 23))
        fine[0, 1, 3, 1, 1] = float(np.log(14 / 27))
        fast = decode_heads((coarse, fine), fast_nms=True)[0]
        reference = decode_heads((coarse, fine), fast_nms=False)[0]
        self.assertEqual(fast.shape, (1, 6))
        torch.testing.assert_close(fast, reference)

    def test_numpy_and_torch_decoders_match_many_candidates(self):
        generator = np.random.default_rng(7)
        coarse = generator.standard_normal((1, 3, 8, 2, 2), dtype=np.float32)
        fine = generator.standard_normal((1, 3, 8, 4, 4), dtype=np.float32)
        expected = decode_heads((torch.from_numpy(coarse), torch.from_numpy(fine)),
                                threshold=.4)[0].numpy()
        actual = decode_numpy_heads([coarse, fine], threshold=.4)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_ap50_counts_false_positive_before_hit(self):
        predictions = [torch.tensor([[0, 0, .2, .2, .99, 0],
                                     [0, 0, 1, 1, .9, 0]], dtype=torch.float32)]
        labels = [torch.tensor([[0, .5, .5, 1, 1]], dtype=torch.float32)]
        result = ap50(predictions, labels, classes=2)
        self.assertAlmostEqual(result['map50'], .5)
        self.assertIsNone(result['per_class_ap50'][1])


if __name__ == '__main__':
    unittest.main()
