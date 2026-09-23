# Jetson YOLO CUDA

This project grows the original handwritten CUDA MNIST CNN into an object
detection project. The original CNN sources remain at the repository root.
The current detection path runs decoded YOLO ONNX models with ONNX Runtime;
a separate CUDA post-processing prototype lives in src/yolo_cuda.

## Status and scope

- Working: YOLOv5/YOLOv8 ONNX image inference, box decoding, class-aware NMS,
  YOLO label validation, and same-process latency benchmarking.
- Optional: CUDA and TensorRT execution through ONNX Runtime when the installed
  runtime exposes those providers. CPU is available as a reference backend.
- Prototype: CUDA candidate decoding in src/yolo_cuda. It is a standalone
  component, not wired into the ONNX runner and not a complete YOLO network.
- Pending device work: Jetson model validation, accuracy evaluation, power
  sampling, and measured CPU/CUDA/TensorRT comparisons.

No model weights or dataset are bundled. A trained ONNX model and test image
are needed for inference or performance measurements.

## Supported ONNX outputs

The model must produce one decoded, non-NMS output tensor. Coordinates are
center x, center y, width, height in input-image pixels. Scores are
probabilities, not logits.

| --format | Output shape | Fields |
| --- | --- | --- |
| v5 | [1, candidates, 5 + classes] | xywh, objectness, class scores |
| v8 | [1, 4 + classes, candidates] | xywh, class scores |

Raw anchor/grid heads, end-to-end NMS exports, and segmentation models are
not yet supported.

## Set up and run

For a CPU smoke test, install the dependencies in requirements-cpu.txt.
On Jetson, install an ONNX Runtime build compatible with its JetPack/CUDA
stack that exposes CUDAExecutionProvider or TensorrtExecutionProvider.
The runner fails clearly if the requested provider is unavailable.

~~~bash
python -m pip install -r requirements-cpu.txt
python tools/yolo_inference.py --model model.onnx --image sample.jpg --format v8 --backend cpu --annotated result.jpg
python tools/benchmark_jetson.py --model model.onnx --image sample.jpg --format v8 --backend cpu --runs 100 --warmup 10
~~~

Use --backend cuda or --backend tensorrt on a device with those providers.
The benchmark loads the model and image once, warms up the session, then
measures inference and preprocessing-plus-inference-plus-postprocessing
latency. It reports the providers activated by ONNX Runtime. Provider
activation alone does not prove every graph node ran on that provider.
Keep model, input size, image, power mode, and clock settings fixed when
comparing backends. JSON output includes the available Jetson release,
power-mode, and clock-setting metadata; it does not yet sample watts.

## Dataset validation

Use images/train and labels/train, with optional val and test splits. A label
line is: class_id x_center y_center width height, normalized to [0, 1].
The classes file contains one name per line.

~~~bash
python tools/validate_yolo_dataset.py path/to/dataset --classes path/to/classes.txt
~~~

## CUDA prototype

The CUDA component decodes independent candidate logits and runs CPU NMS.
It is useful for testing kernel behavior, but its input contract is not the
same as the supported ONNX outputs above.

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j
./build/yolo_postprocess_demo
~~~

Set CMAKE_CUDA_ARCHITECTURES to the target GPU architecture. The original
Makefile builds the MNIST CNN separately.

## Background

The existing CNN implementation is the author's prior project. Darknet and
YOLO informed the direction of this independent detector work; no Darknet
source code is included.
