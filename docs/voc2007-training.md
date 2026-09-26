# VOC 2007 detector training record

This record describes the local run completed on 2026-09-23 UTC. It uses
Pascal VOC 2007, a detection dataset documented in
[Darknet's YOLO training instructions](https://www.pjreddie.com/darknet/yolo/).
The experiment uses only VOC 2007, not the VOC 2007+2012 combination from
the full Darknet training recipe.

## Dataset

- Source: [VOC 2007 train/validation and annotated test archives](https://www.pjreddie.com/projects/pascal-voc-dataset-mirror/).
- 20 VOC object classes.
- Train: 4,510 images, 11,303 labeled objects.
- Validation: 501 images, 1,305 labeled objects.
- Official 2007 test IDs: 4,952 images, 12,032 labeled objects.
- Split seed: 42; 10% of trainval IDs assigned to validation.
- 5,998 difficult objects excluded across all splits.
- All YOLO labels passed tools/validate_yolo_dataset.py with zero errors.
- The test IDs were not used for model selection or training.

Archive SHA-256:

| Archive | SHA-256 |
| --- | --- |
| VOCtrainval_06-Nov-2007.tar | 7d8cd951101b0957ddfd7a530bdc8a94f06121cfc1e511bb5937e973020c7508 |
| VOCtest_06-Nov-2007.tar | 6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892 |

## Training setup

- Starting model: Ultralytics pretrained YOLOv8n checkpoint.
- Training library: Ultralytics 8.3.175.
- Python 3.10.18; PyTorch 2.8.0+cu126.
- NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8 GiB.
- 30 epochs, 416-pixel square input, batch size 16, seed 42.
- Best checkpoint selected on the 501-image validation split.
- Run command: python tools/train_voc2007.py --epochs 30 --imgsz 416 --batch 16 --workers 4

## Held-out test results

The test run used the best checkpoint on all 4,952 test images at 416 pixels.
Scores use Ultralytics' mAP calculation with difficult objects excluded.
They are not directly comparable to historical Darknet VOC paper scores.

| Metric | Value |
| --- | ---: |
| Precision | 0.799 |
| Recall | 0.690 |
| mAP@0.5 | 0.771 |
| mAP@0.75 | 0.605 |
| mAP@0.5:0.95 | 0.557 |

Per-class mAP@0.5:0.95:

| Class | AP | Class | AP |
| --- | ---: | --- | ---: |
| aeroplane | 0.582 | bicycle | 0.633 |
| bird | 0.484 | boat | 0.423 |
| bottle | 0.364 | bus | 0.717 |
| car | 0.704 | cat | 0.635 |
| chair | 0.398 | cow | 0.555 |
| diningtable | 0.572 | dog | 0.599 |
| horse | 0.675 | motorbike | 0.598 |
| person | 0.596 | pottedplant | 0.288 |
| sheep | 0.542 | sofa | 0.576 |
| train | 0.639 | tvmonitor | 0.566 |

## Export and local inference

The best checkpoint was exported to ONNX with input [1, 3, 416, 416] and
decoded output [1, 24, 3549]. The repository's ONNX runner detected a
person and a dog in held-out image 000001.jpg.

On the Windows laptop, a 100-run CPU ONNX Runtime 1.28.0 baseline on that
one image measured 15.89 ms mean end-to-end, 18.04 ms p95 end-to-end,
and 10.32 ms mean model inference. The implied sequential throughput is
62.93 FPS. This includes preprocessing and NMS but excludes model loading
and image decoding. It is a single-image latency benchmark, not dataset
throughput and not a Jetson result.

Local artifacts, excluded from Git:

| Artifact | Relative path | SHA-256 |
| --- | --- | --- |
| Best PyTorch checkpoint | runs/voc2007/yolov8n/weights/best.pt | b29d6c6df198d14d4883e1c5c96f5a240ec47f033701d95ec8482667d1ae3b17 |
| Exported ONNX model | runs/voc2007/yolov8n/weights/best.onnx | 2a873c7beba722d45a7b197657aee32547d9b2d88255f5e5098c8884612db7b1 |
| Test metrics | runs/voc2007/yolov8n/summary.json | — |
| CPU benchmark | runs/voc2007/yolov8n/cpu_benchmark.json | — |

Jetson CPU and native TensorRT measurements were subsequently completed for
this **Ultralytics YOLOv8n baseline**; see [jetson-benchmark.md](jetson-benchmark.md).
They are not measurements of the independent scratch detector. TensorRT is
optional for deployment.

Ultralytics supplies the training package and pretrained weights. Its
[published license page](https://www.ultralytics.com/license) states that
its trained models are AGPL-3.0 by default.
