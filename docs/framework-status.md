# Framework status

This project is an independent C++/CUDA neural-network framework inspired by [pjreddie/darknet](https://github.com/pjreddie/darknet). It keeps the original CUDA CNN in Git history and implements configurable training and inference in `native/`. Supported Darknet cfg layers and weight files are compatible where documented; unsupported layers fail explicitly.

## Working paths

- Config-driven CPU/CUDA graph inference and training for convolution, max-pooling, connected, route, shortcut, upsample, reorg, dropout, YOLO and region heads, and terminal softmax. Batch normalization is trainable with true minibatch statistics in the detector trainers. Classifier examples cover linear, MLP and CNN models.
- RNN, LSTM and GRU cfg layers have persistent sequence state, CPU/CUDA inference, and truncated backpropagation through time in the CPU/CUDA classifier trainers. The detector trainers reject recurrent layers with a clear error.
- Native CUDA detector training includes anchor assignment and YOLO/region loss, crop/flip/HSV augmentation, multiscale resizing, burn-in/step-decay scheduling, SGD momentum and weight decay. The default large-convolution path uses cuBLAS; cuDNN convolution and spatial batch normalization are available with `-DDETECTOR_CUDNN=ON`.
- CUDA detector training can use multiple selected devices via `--gpus 0,1`: one replica per device, balanced shards of the total `--batch`, concurrent local steps, and sample-weighted state synchronization after each global step. The one-GPU path and weighted synchronization math are tested. This laptop has only one GPU, so an actual two-GPU run and its throughput have not been measured.
- Darknet-format weight load/save, partial backbone initialization, checkpoint step/epoch resume, IDX/MNIST classifier CLI, native detection/evaluation, and Jetson inference. ONNX/TensorRT is optional and currently specific to YOLOv3-tiny's two-head graph.

## Measured model results

All detector scores below use the same native CUDA evaluator with confidence 0.01, class-aware NMS IoU 0.45 and all-point AP@0.5. VOC2007 labels were converted without difficult objects; these are not official VOC leaderboard scores.

| Checkpoint origin and architecture | Validation: 501 images, 1,305 objects | Test: 4,952 images, 12,032 objects |
| --- | ---: | ---: |
| This framework, YOLOv3-tiny with COCO backbone initialization | 0.59801 | 0.58182 |
| This framework, YOLOv2 / Darknet19 with pretrained backbone | 0.56054 | 0.57215 |
| This framework, full YOLOv3 / Darknet-53 with pretrained backbone | 0.62098 | 0.61593 |
| Historical original-Darknet YOLOv3-tiny weights, evaluated here | 0.57612 | 0.55534 |

The corrected native YOLOv3-tiny checkpoint was also evaluated on Jetson with native CUDA and TensorRT FP32/FP16. Historical full-dataset MNIST Jetson runs of the project's classifier trainer scored 97.24% test accuracy for an MLP and 97.27% for a small CNN.

## Current limits

- This is the supported subset of Darknet's cfg vocabulary, not an arbitrary upstream cfg interpreter. A same-evaluator external-Darknet YOLOv2 baseline has not been run.
- Actual multi-GPU execution needs a host with at least two CUDA devices. Each device uses local batch-normalization statistics during its forward/backward pass; running statistics are averaged between devices. The current cross-device state transfer uses host memory, and its throughput is unmeasured.
- Full YOLOv3 and YOLOv2 have native CPU/CUDA inference and training, but the optional ONNX/TensorRT export/inference path still targets YOLOv3-tiny only.
- Native training from scratch has not produced a useful VOC detector; the measured native checkpoints above use pretrained backbones.

See [engineering milestones](engineering-log.md) and [the training design](../DESIGN.yaml) for the implementation and validation history.
