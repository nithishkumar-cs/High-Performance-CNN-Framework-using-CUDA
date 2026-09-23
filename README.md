# Jetson YOLO CUDA

This project evolved from a handwritten CUDA MNIST CNN into a 20-class
Pascal VOC object detector. The earlier CNN sources remain recoverable in
Git history. Detection training uses Ultralytics YOLOv8n; deployment runs
decoded YOLO ONNX models with ONNX Runtime. A separate CUDA
post-processing prototype lives in src/yolo_cuda.

## Status and scope

- Working: YOLOv5/YOLOv8 ONNX image inference, box decoding, class-aware NMS,
  YOLO label validation, and same-process latency benchmarking.
- Optional: CUDA and TensorRT execution through ONNX Runtime when the installed
  runtime exposes those providers. CPU is available as a reference backend.
- Prototype: CUDA candidate decoding in src/yolo_cuda. It is a standalone
  component, not wired into the ONNX runner and not a complete YOLO network.
- Training: VOC 2007 conversion, fine-tuning, and held-out test evaluation
  through tools/prepare_voc2007.py and tools/train_voc2007.py.
- Pending device work: Jetson model validation, power sampling, and measured
  CPU/CUDA/TensorRT comparisons.

No model weights or dataset are bundled. A trained ONNX model and test image
are needed for inference or performance measurements.

## Train on Pascal VOC 2007

Darknet documents Pascal VOC 2007 and 2012 as YOLO training data. This
repository uses the smaller VOC 2007 subset for a laptop-scale experiment.
Download the official train/validation and annotated test archives from
https://www.pjreddie.com/projects/pascal-voc-dataset-mirror/ and extract
both into data/voc2007. The result should include
data/voc2007/VOCdevkit/VOC2007.

~~~bash
python tools/prepare_voc2007.py
python tools/validate_yolo_dataset.py data/voc2007/yolo --classes data/voc2007/yolo/classes.txt
python tools/train_voc2007.py --epochs 30 --imgsz 416 --batch 16
python tools/evaluate_voc2007.py --weights runs/voc2007/yolov8n/weights/best.pt
python tools/export_voc2007.py --weights runs/voc2007/yolov8n/weights/best.pt --imgsz 416
~~~

Use a CUDA-enabled PyTorch environment containing Ultralytics for training.
The tested environment uses Python 3.10, PyTorch 2.8.0+cu126, and
Ultralytics 8.3.175 on an RTX 3070 Ti Laptop GPU. Preparation shuffles the
5,011 VOC train/validation IDs with seed 42 into 4,510 training and 501
validation images. All 4,952 official test IDs stay held out. VOC difficult
objects are excluded from training labels. The script saves best.pt and
summary.json under runs/voc2007. If training was interrupted after writing
the checkpoint, the standalone evaluation command can evaluate best.pt
without repeating training. The ONNX export can be run in a separate
CPU Python environment that has onnx, onnxruntime, and Ultralytics.
Scores reported by Ultralytics use its current mAP implementation and
should not be compared directly with historical Darknet VOC paper numbers.

The YOLOv8n starting weights and training package come from Ultralytics.
Ultralytics states that its software and trained models are AGPL-3.0 by
default; review https://www.ultralytics.com/license before redistributing
the trained checkpoint or deploying it in a private product. Dataset
images and model weights are kept out of Git.

The completed 30-epoch VOC 2007 run scored 0.771 mAP@0.5 and 0.557
mAP@0.5:0.95 on the held-out 2007 test set. See
docs/voc2007-training.md for the exact split, settings, per-class scores,
artifact checksums, and local CPU baseline.

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

Set CMAKE_CUDA_ARCHITECTURES to the target GPU architecture.

## Background

The original CNN implementation is the author's prior project and remains
in this repository's Git history. Darknet and YOLO informed the direction
of the detector; no Darknet source code is included.
