# Trained scratch detector on Jetson

Measured on 2026-09-26 on the author's Jetson Orin Nano Super, Jetson Linux
R36.5.0, TensorRT 10.3.0, in 25W power mode. Clocks were not locked. The
validation-selected, random-initialized epoch-30 model from
[scratch-yolo-training.md](scratch-yolo-training.md) was exported to ONNX
and copied into an isolated `scratch-trained` subfolder on the Jetson. The
copied ONNX SHA-256 matched the local export:
`739523e232acc43cce5852be49ebb3e35b97fcb6f377964c8e0e84806c7f88b2`.
The device-built FP16 engine SHA-256 was
`7a981f43627ce656fe2fafb4b1fccf9f3c21c8e91defb562a503c975aa338667`.

## Full-detector latency

One held-out VOC image, `000001.jpg`, was loaded before timing. Each path
used 416x416 RGB, batch 1, confidence 0.25, class-aware NMS IoU 0.45,
10 warmups and 100 timed calls. Timed work includes letterbox, model
execution with synchronous transfers, raw-head decode/NMS, and box remap;
it excludes file decoding and session/engine construction. This is
single-image sequential latency, not camera or dataset throughput.

| Backend | Mean end-to-end | p95 end-to-end | Mean model call | Sequential FPS | Detections |
| --- | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime CPU | 70.75 ms | 73.55 ms | 61.72 ms | 14.14 | 8 |
| Native TensorRT FP16 | 14.99 ms | 15.42 ms | 5.88 ms | 66.73 | 8 |

The FP16 full-detector path was 4.72x faster than ONNX CPU for this image.
The earlier [YOLOv8n Jetson baseline](jetson-benchmark.md) measured 16.17 ms
full-detector FP16 on the same image and nominal 25W mode, but used a
different architecture and decoder and returned two detections. The scratch
model's held-out AP@0.5 was only 0.0433 versus the pretrained baseline's
0.7569 under a shared evaluator. A small single-image latency edge does
**not** make the scratch model the better detector. Neither run used fixed
clocks, and run-to-run/device variation is not characterized.

## Numerical parity

The device-built FP16 engine exposes the expected two raw heads, `[1,3,25,13,13]`
and `[1,3,25,26,26]`. Against ONNX CPU on three held-out images, using the
same 0.25 confidence threshold:

| Image | CPU/TRT detections matched | Worst raw-head absolute error | Minimum matched box IoU | Largest matched score error |
| --- | ---: | ---: | ---: | ---: |
| `000001.jpg` | 8/8 | 0.09084 | 0.99496 | 0.00538 |
| `000002.jpg` | 29/29 | 0.07583 | 0.99574 | 0.00386 |
| `000003.jpg` | 13/13 | 0.07456 | 0.99646 | 0.00141 |

All matched classes agreed. FP16 differences are expected. This is a
three-image correctness check, not a full-dataset TensorRT AP evaluation.

## Reproduce and limitations

The Jetson used the existing project virtual environment for NumPy, Pillow,
and CPU ONNX Runtime; PyTorch was not installed on the device. The native
TensorRT adapter used the device's Python TensorRT bindings and CUDA runtime.
From the isolated folder after copying the model, image, and torch-free
runtime modules:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=selected.onnx --saveEngine=selected_fp16.engine --fp16 --skipInference
../.venv/bin/python -m tools.benchmark_scratch_jetson --model selected.onnx --image 000001.jpg --backend cpu --side 416 --confidence .25 --nms-iou .45 --runs 100 --warmup 10 --output cpu_benchmark.json
PYTHONPATH=/usr/lib/python3.10/dist-packages ../.venv/bin/python -m tools.check_scratch_parity --onnx selected.onnx --engine selected_fp16.engine --image 000001.jpg --confidence .25
PYTHONPATH=/usr/lib/python3.10/dist-packages ../.venv/bin/python -m tools.benchmark_scratch_jetson --model selected_fp16.engine --image 000001.jpg --backend tensorrt-native --side 416 --confidence .25 --nms-iou .45 --runs 100 --warmup 10 --output trt_benchmark.json
```

The raw benchmark JSON is retained locally under ignored
`runs/scratch_yolo/jetson/` and on the Jetson in the isolated folder. No
board-input power trace was taken for this scratch run. Native custom-CUDA
CNN kernels remain unimplemented; TensorRT supplies the accelerated model
execution. TensorRT is optional for training and evaluation.
