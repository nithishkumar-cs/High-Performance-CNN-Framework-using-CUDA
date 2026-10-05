# Laptop training, Jetson inference

Future training runs use the laptop's RTX 3070 Ti (CUDA architecture 86). The Jetson runs inference and evaluation only. The reported 10,000-batch VOC checkpoint was trained on Jetson before this location decision; its published measurements remain historical results, not evidence of laptop training.

The project's native detector trainer does not yet implement faithful YOLOv3-tiny training. For a useful VOC model, this workflow builds the pinned original Darknet GPU trainer in a laptop Docker container. No upstream source is committed here. The container uses CUDA 12.4 without cuDNN and targets the laptop GPU. The [VOC training guide](yolov3-tiny-training.md) describes the model, dataset split, checkpoint selection, and evaluation limits.

From PowerShell in the repository root, build the training image and verify GPU access:

```powershell
docker build -f native/docker/Dockerfile.darknet-train -t darknet-voc-laptop .
$repo = (Get-Location).Path
docker run --rm --gpus all --mount "type=bind,source=$repo,target=/work" darknet-voc-laptop nvidia-smi
```

The official VOC2007 data must already be converted at `data/voc2007/yolo/` using `native/tools/convert_voc2007.py`. Place the official COCO `yolov3-tiny.weights` in ignored `data/` and check its SHA-256 against the value in the [training guide](yolov3-tiny-training.md). Prepare a new, empty run directory from inside the container so the generated image lists and `.data` file have Linux paths:

```powershell
docker run --rm --gpus all --mount "type=bind,source=$repo,target=/work" darknet-voc-laptop python3 /work/native/tools/prepare_darknet_voc.py --upstream-cfg /opt/darknet/cfg/yolov3-tiny.cfg --train-images /work/data/voc2007/yolo/images/train --train-labels /work/data/voc2007/yolo/labels/train --val-images /work/data/voc2007/yolo/images/val --val-labels /work/data/voc2007/yolo/labels/val --output-dir /work/runs/yolov3-tiny-voc/laptop
```

Initialize the compatible first 15 layers, run the four-batch preflight, then train the full 10,000-batch configuration. The output stays in the ignored `runs/` directory on the laptop. The final command is a long-running GPU job; keep its terminal active and preserve its log.

```powershell
docker run --rm --gpus all --mount "type=bind,source=$repo,target=/work" darknet-voc-laptop /opt/darknet/darknet partial /opt/darknet/cfg/yolov3-tiny.cfg /work/data/yolov3-tiny.weights /work/runs/yolov3-tiny-voc/laptop/yolov3-tiny.conv.15 15
docker run --rm --gpus all --mount "type=bind,source=$repo,target=/work" darknet-voc-laptop /opt/darknet/darknet detector train /work/runs/yolov3-tiny-voc/laptop/voc.data /work/runs/yolov3-tiny-voc/laptop/yolov3-tiny-voc-preflight.cfg /work/runs/yolov3-tiny-voc/laptop/yolov3-tiny.conv.15
docker run --rm --gpus all --mount "type=bind,source=$repo,target=/work" darknet-voc-laptop /opt/darknet/darknet detector train /work/runs/yolov3-tiny-voc/laptop/voc.data /work/runs/yolov3-tiny-voc/laptop/yolov3-tiny-voc.cfg /work/runs/yolov3-tiny-voc/laptop/backup/yolov3-tiny-voc-preflight_final.weights
```

Evaluate and select a laptop checkpoint on the validation split before touching the held-out test split. Transfer the selected `.weights`, matching `.cfg`, and `voc.names` to Jetson for native CUDA inference. For TensorRT, export ONNX from those same weights on the laptop and build the engine on Jetson as described in the [TensorRT guide](onnx-tensorrt.md). Engine construction is deployment preparation; all model training remains on the laptop. Do not transfer the VOC dataset or training process to Jetson.
