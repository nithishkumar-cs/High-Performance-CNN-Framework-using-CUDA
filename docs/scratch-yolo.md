# Independent tiny YOLO detector

This is the primary implementation, separate from the pretrained Ultralytics
YOLOv8n comparison. It is based on the ideas and published architecture in
the [YOLOv3 paper](https://pjreddie.com/static/papers/YOLOv3.pdf) and the
[Darknet YOLOv3-tiny configuration](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg).
No Darknet source file, configuration file, or pretrained weight file is
included. The historical MNIST CUDA CNN remains available in Git history.

## Implemented

- An independently defined CNN with five convolution/pool stages, a deep
  branch, and two anchor heads at 13x13 and 26x26 for 416-pixel input.
- Six fixed anchor priors, with three per head. Heads emit raw logits in
  `[batch, 3, 5 + classes, grid, grid]` layout.
- VOC 2007 loader with one shared RGB letterbox transform for training and
  inference. The already-prepared split is 4,510 train, 501 validation,
  and 4,952 held-out test images.
- Random-initialized SGD training, checkpoint selection by validation loss,
  periodic epoch archives for later AP selection, atomic weight writes,
  and `--resume` with run-setting validation.
- Raw-head decoding and class-aware NMS in PyTorch and in torch-free NumPy.
  Unit tests compare the two decoders on a synthetic head.
- An AP@0.5 evaluation command with 101-point interpolated precision/recall.
  It does not report mAP@0.5:0.95 and should not be treated as numerically
  identical to Ultralytics' evaluator. A separate adapter evaluates the
  YOLOv8n comparison through this same AP implementation.
- Static-shape ONNX export of both raw heads, ONNX checker validation, and
  ONNX Runtime numeric comparison with PyTorch when available.
- A separate full-detector benchmark for ONNX Runtime CPU/CUDA and native
  TensorRT, including letterbox, execution, raw-head decoding/NMS, and box
  remapping. Image-file loading and model/session construction are excluded.
- Jetson Orin Nano Super smoke deployment: a random/smoke checkpoint exported
  to ONNX, built as a two-output FP16 TensorRT engine, and ran through the
  full-detector adapter. Both raw heads matched ONNX CPU within 0.00009
  maximum absolute error on one VOC image. This validates compatibility,
  **not** trained-model accuracy or a publishable latency comparison.

The loss is a compact teaching/prototype objective: each object is assigned
to its best width/height anchor; sigmoid x/y, log-width/height, objectness,
and class terms are optimized. It is **not** the exact Darknet loss. It does
not yet implement ignore regions for near-miss anchors, collision resolution,
scale-aware weighting, augmentation, or a mature learning-rate schedule.
These omissions may materially limit accuracy. PyTorch supplies the CNN
operators, so this is not yet a native custom-CUDA network engine.

## Reproduce

Use a CUDA-enabled PyTorch environment for training; NumPy and Pillow are
also required. The prepared VOC split lives under `data/voc2007/yolo` and is
ignored by Git. From the repository root:

```bash
python -m tools.train_scratch_yolo --epochs 30 --side 416 --batch 16 --workers 4
python -m tools.train_scratch_yolo --epochs 30 --side 416 --batch 16 --workers 4 --resume
python -m tools.evaluate_scratch_yolo --weights runs/scratch_yolo/epochs/epoch_025.pt --split val --output runs/scratch_yolo/val_epoch25.json
python -m tools.evaluate_scratch_yolo --weights runs/scratch_yolo/epochs/epoch_030.pt --split val --output runs/scratch_yolo/val_epoch30.json
python -m tools.select_scratch_checkpoint runs/scratch_yolo/val_epoch25.json runs/scratch_yolo/val_epoch30.json
python -m tools.evaluate_scratch_yolo --weights runs/scratch_yolo/selected.pt --split test
python -m tools.evaluate_ultralytics_common --weights runs/voc2007/yolov8n/weights/best.pt --split test
python -m tools.export_scratch_yolo --weights runs/scratch_yolo/selected.pt --output runs/scratch_yolo/selected.onnx
python -m tools.benchmark_scratch_jetson --model runs/scratch_yolo/selected.onnx --image data/voc2007/yolo/images/test/000001.jpg --backend cpu
```

The second command is for interruption recovery, not something to run after
training already reached the requested epoch. `best.pt` is selected by
validation *loss*, which may not track AP; score periodic checkpoints on
the validation split and select by validation AP before running the held-out
test evaluation. The selector refuses partial-validation or smoke-test
records. The default archive interval is five epochs, configurable with
`--save-every`. To smoke-test quickly, add
`--limit-train 16 --limit-val 16 --output runs/scratch_yolo_smoke`; do not
publish its checkpoint or metric as trained-model results. A training
summary records real run timestamps and epoch losses; no dates are inferred.

For optional native TensorRT on the Jetson, copy the exported ONNX model,
one test image, `scratch_yolo/{__init__,config,numpy_decode,preprocess}.py`,
`tools/{benchmark_scratch_jetson,tensorrt_inference}.py` into a project
folder on the device. With TensorRT installed there:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best_fp16.engine --fp16 --skipInference
PYTHONPATH=/usr/lib/python3.10/dist-packages python -m tools.check_scratch_parity --onnx best.onnx --engine best_fp16.engine --image 000001.jpg
python -m tools.benchmark_scratch_jetson --model best_fp16.engine --image 000001.jpg --backend tensorrt-native
```

The engine is specific to the Jetson/software stack. The existing YOLOv8n
ONNX runner cannot decode these raw heads; use the scratch benchmark command.
Keep the same image, input size, confidence/NMS settings, warmups, power mode,
and clock policy when comparing CPU and TensorRT. A TensorRT `trtexec`
model-only rate is not full-detector throughput.

## What remains

The first full run and same-evaluator held-out comparison are in
[scratch-yolo-training.md](scratch-yolo-training.md); scratch accuracy is
currently poor. Trained-model Jetson timing and three-image FP16 output
parity are in [scratch-yolo-jetson.md](scratch-yolo-jetson.md). Full-dataset
TensorRT AP and power tracing remain open. Add native CUDA convolution,
pooling, and route kernels or clearly retain the PyTorch/TensorRT deployment
approach. Improve the training objective and data augmentation. The optional
TensorRT path is an optimization, not a prerequisite for the model.
