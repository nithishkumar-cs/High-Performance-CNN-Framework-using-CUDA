# Scratch tiny-detector training record

The independent, YOLOv3-tiny-inspired detector was trained from **random
weights** on the fixed VOC 2007 split on 2026-09-26 UTC. No Darknet or
Ultralytics source code or weights were used for this model. The architecture
was informed by the [YOLOv3 paper](https://pjreddie.com/static/papers/YOLOv3.pdf)
and [published tiny configuration](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg).

## Setup and selection

- 4,510 train, 501 validation, and 4,952 held-out test images; 20 classes.
  The same prepared VOC split is used by the YOLOv8n comparison. Difficult
  objects were excluded in the existing conversion.
- 416x416 RGB letterbox input; 8,713,766 trainable parameters; two raw
  anchor heads. Fixed published six-anchor prior, not fitted to VOC.
- Python 3.10.18, PyTorch 2.8.0+cu126, RTX 3070 Ti Laptop GPU.
- 30 epochs, batch 16, seed 42; SGD learning rate 0.001, momentum 0.9,
  weight decay 0.0005. No pretrained weights, augmentation, learning-rate
  schedule, or ignore-region target logic.
- Recorded run window: 2026-09-26 17:21:34 to 18:00:08 UTC. The last
  checkpoint reached all 30 epochs. Times come from run output, not inferred
  from Git history.
- The loss-selected `best.pt` was epoch 5. It was **not** the AP-best model.
  The saved epochs below were scored on all 501 validation images with
  confidence 0.01, class-aware NMS IoU 0.6, at most 300 detections/image,
  and 101-point interpolated AP@IoU 0.5. Epoch 30 was selected before the
  test split was evaluated.

| Epoch | Validation AP@0.5 |
| ---: | ---: |
| 5 | 0.00127 |
| 17 | 0.02382 |
| 21 | 0.03729 |
| 25 | 0.04332 |
| 30, selected | 0.04694 |

## Held-out test comparison

Both rows below use `scratch_yolo.metrics.ap50` on all 4,952 official VOC
2007 test images, 416-pixel input, confidence 0.01, NMS IoU 0.6, and a
maximum of 300 detections per image. This is a **same-evaluator** comparison,
not a controlled training-regime comparison: YOLOv8n began with pretrained
weights and has a different architecture and parameter count.

| Model | Parameters | Starting weights | Test AP@0.5 |
| --- | ---: | --- | ---: |
| Independent tiny detector, epoch 30 | 8,713,766 | Random | 0.04333 |
| Ultralytics YOLOv8n baseline | 3,014,748 | Pretrained | 0.75691 |

The older baseline report gives 0.771 mAP@0.5 through Ultralytics' own
evaluator; it is not the number used in the table. Scratch test AP is low,
not a competitive detector. On held-out image `000001.jpg`, the selected
model returned several person boxes and missed the dog at confidence 0.25;
that example is qualitative, not a separate metric. No test metric was used
to select a checkpoint.

The likely contributors to the gap are the small VOC 2007-only training set,
random initialization, simplified objective/anchor assignment, lack of
augmentation and learning-rate schedule, and limited 30-epoch run. These
are hypotheses, not isolated causal measurements. A useful next experiment
is to improve one factor at a time and rerun the same full validation/test
protocol. Native custom-CUDA network kernels remain unimplemented; training
used PyTorch CUDA operations.

## Local artifacts

Weights and data remain outside Git under ignored `runs/` and `data/`.

| Artifact | SHA-256 |
| --- | --- |
| `runs/scratch_yolo/selected.pt` | `5dde402e98ee63e04fa321055a7898c21ca1fff0d643ff3bd251600335899a7e` |
| `runs/scratch_yolo/selected.onnx` | `739523e232acc43cce5852be49ebb3e35b97fcb6f377964c8e0e84806c7f88b2` |

The ONNX export has static `[1,3,416,416]` input and raw outputs
`[1,3,25,13,13]` and `[1,3,25,26,26]`. ONNX checker and ONNX Runtime
numeric comparison against PyTorch passed. See [scratch-yolo.md](scratch-yolo.md)
for commands and limitations. Jetson deployment results are recorded
separately so model-only, full-detector, and baseline timings are not mixed.
