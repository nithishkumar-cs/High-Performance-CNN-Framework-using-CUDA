#!/usr/bin/env python3
'''Export the independent detector's raw heads to ONNX for optional deployment.'''

import argparse
from pathlib import Path

import numpy as np
import torch

from scratch_yolo.model import TinyDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    model = TinyDetector(int(checkpoint['classes'])).eval()
    model.load_state_dict(checkpoint['model'])
    side = int(checkpoint['side'])
    sample = torch.rand(1, 3, side, side)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference = [head.numpy() for head in model(sample)]
        torch.onnx.export(model, sample, str(args.output), opset_version=17,
                          input_names=['images'], output_names=['coarse', 'fine'],
                          dynamo=False)
    import onnx
    onnx.checker.check_model(str(args.output))
    try:
        import onnxruntime as ort
    except ImportError:
        print('ONNX exported and checked; install onnxruntime for numeric parity')
    else:
        session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
        actual = session.run(None, {'images': sample.numpy()})
        for expected, got in zip(reference, actual):
            np.testing.assert_allclose(got, expected, rtol=2e-4, atol=2e-4)
        print('ONNX Runtime raw-head numeric parity passed')
    print(f'Exported {args.output} with raw coarse/fine heads')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
