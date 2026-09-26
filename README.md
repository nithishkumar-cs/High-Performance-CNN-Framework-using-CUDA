# Scratch YOLO detector for CUDA and Jetson

The primary project is an independently implemented, YOLOv3-tiny-inspired
20-class detector in `scratch_yolo/`. It has its own two-scale network,
anchor assignment, training objective, VOC loader, raw-head decoder, and
inference command. It starts from random weights; it does not import
Darknet or Ultralytics code or pretrained weights. Architecture and anchor
ideas follow the [original YOLOv3 paper](https://pjreddie.com/static/papers/YOLOv3.pdf)
and [Darknet tiny configuration](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg).

This is a working *training prototype*, not a finished custom-CUDA inference
engine. The network currently runs through PyTorch CPU/CUDA operations. The
loss is deliberately simplified and is not claimed to reproduce Darknet
training exactly. A 416-pixel GPU smoke run, raw-head ONNX export with
numerical parity, and Jetson CPU ONNX/native TensorRT smoke tests passed.
The two TensorRT raw heads matched ONNX CPU within 0.00009 maximum absolute
error on one smoke image. The *trained* model has since been measured on the
Jetson: 70.75 ms ONNX CPU and 14.99 ms native TensorRT FP16 mean end-to-end
on one image. Native custom-CUDA network kernels remain unimplemented.
The completed 30-epoch run selected epoch 30 by validation AP and measured
0.0433 AP@0.5 on the held-out VOC 2007 test set; the pretrained YOLOv8n
comparison measured 0.7569 with the same evaluator. This scratch recipe is
not yet accurate enough for practical detection. See
`docs/scratch-yolo-training.md` for exact settings and limitations.
See `docs/scratch-yolo-jetson.md` for the trained-model timing and parity record.

## Train the primary detector

Prepare the official VOC 2007 trainval/test archives using the baseline-neutral
`tools/prepare_voc2007.py` script below. Then use a CUDA-enabled PyTorch
environment with NumPy and Pillow:

~~~bash
python -m tools.train_scratch_yolo --epochs 30 --side 416 --batch 16
python -m tools.train_scratch_yolo --epochs 30 --side 416 --batch 16 --resume
python -m tools.evaluate_scratch_yolo --weights runs/scratch_yolo/selected.pt
python -m tools.evaluate_ultralytics_common --weights runs/voc2007/yolov8n/weights/best.pt
python -m tools.predict_scratch_yolo --weights runs/scratch_yolo/selected.pt --image data/voc2007/yolo/images/test/000001.jpg
python -m tools.export_scratch_yolo --weights runs/scratch_yolo/selected.pt --output runs/scratch_yolo/selected.onnx
python -m tools.benchmark_scratch_jetson --model runs/scratch_yolo/selected.onnx --image data/voc2007/yolo/images/test/000001.jpg --backend cpu
python -m unittest discover -s tests -p test_scratch_yolo.py
~~~

Training saves `best.pt` by validation loss, `last.pt`, periodic epoch
checkpoints for validation-AP selection, and `summary.json`. Select
`selected.pt` using the full-validation steps in `docs/scratch-yolo.md`
before running the held-out test command above.
These artifacts are ignored by Git. `--limit-train` and `--limit-val` are
smoke-test options; checkpoints made with either flag are not trained models.
TensorRT is **optional**, not required to train or evaluate the detector.
The ONNX export has two *raw* heads, so it is not compatible with the older
YOLOv8 ONNX runner. Use `benchmark_scratch_jetson.py` for it. A Jetson-built
FP16 TensorRT engine can be passed with `--backend tensorrt-native`; this
deployment path still needs trained-model validation. See
`docs/scratch-yolo.md` for architecture, evaluation protocol, and limits.

## Ultralytics YOLOv8n comparison baseline

The earlier, published work fine-tuned pretrained Ultralytics YOLOv8n on the
same fixed VOC split. Its training, ONNX runner, and Jetson TensorRT results
remain here for comparison only. They are **not** results of `scratch_yolo/`.
YOLOv8n and YOLOv3-tiny are different architectures; a fair performance
comparison needs the same test split, input size, evaluation implementation,
Jetson power mode, and full-pipeline timing. The older MNIST CNN and unrelated
post-processing prototype are recoverable in Git history, not this working tree.

## Status and scope

- Working: YOLOv5/YOLOv8 ONNX image inference, box decoding, class-aware NMS,
  YOLO label validation, and same-process latency benchmarking.
- Optional: CUDA and TensorRT execution through ONNX Runtime when the installed
  runtime exposes those providers. CPU is available as a reference backend.
- Training: VOC 2007 conversion, fine-tuning, and held-out test evaluation
  through tools/prepare_voc2007.py and tools/train_voc2007.py.
- Measured on Jetson Orin Nano Super: ONNX Runtime CPU and native TensorRT FP16
  full-detector timing, plus model-only TensorRT timing and board-input power.
  ONNX Runtime CUDA/TensorRT provider tests are still pending.

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

## Jetson measurements

On a Jetson Orin Nano Super in 25W mode, the warmed CPU ONNX detector measured
62.27 ms mean end-to-end (16.06 FPS) on one held-out image. The native TensorRT
FP16 detector measured 16.17 ms end-to-end (61.83 FPS) on that image. TensorRT's
separate model-only `trtexec` test measured 2.86 ms host latency and 375
inferences/s; that is not full-detector FPS. See docs/jetson-benchmark.md for
methodology, commands, power samples, and limitations. TensorRT is optional.

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
On Jetson, the CPU path needs ONNX Runtime. The CUDA and TensorRT backends in
this runner require a compatible ONNX Runtime GPU build exposing those
providers; the runner fails clearly if one is unavailable. Native TensorRT
uses `--backend tensorrt-native` with a device-built `.engine` file and the
installed TensorRT Python bindings. See docs/jetson-benchmark.md for setup.

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

## Background

The original CNN implementation is the author's prior project and remains
in this repository's Git history. Darknet and YOLO informed the direction
of the detector; no Darknet source code is included.
