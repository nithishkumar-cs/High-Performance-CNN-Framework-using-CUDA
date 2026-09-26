# Jetson Orin Nano Super benchmark

Measured on 2026-09-23 on the author's Jetson Orin Nano Super Engineering
Reference Developer Kit, running Jetson Linux R36.5.0, Ubuntu 22.04,
TensorRT 10.3.0, and 25W power mode (`nvpmodel -q` returned mode 1).
Clocks were not locked. The same SHA-256 ONNX export documented in
[voc2007-training.md](voc2007-training.md) was copied to the Jetson and checked
after transfer. Input size was 1x3x416x416, batch 1.

## Results

| Path | Workload | Mean latency | p95 latency | Reported rate |
| --- | --- | ---: | ---: | ---: |
| ONNX Runtime 1.23.2 CPU, first run | Full detector, one VOC test image | 68.18 ms end-to-end | 76.78 ms | 14.67 FPS, sequential |
| ONNX Runtime 1.23.2 CPU, second warmed run | Full detector, same image | 62.27 ms end-to-end | 69.24 ms | 16.06 FPS, sequential |
| Native TensorRT 10.3 FP16, 100 runs | Full detector, same image | 16.17 ms end-to-end | 16.56 ms | 61.83 FPS, sequential |
| Native TensorRT 10.3 FP16, 1,000 runs | Full detector, same image | 16.21 ms end-to-end | 16.48 ms | 61.70 FPS, sequential |
| TensorRT 10.3 FP16, cached engine | Model-only, generated input of same shape | 2.856 ms host latency | 2.872 ms | 375.1 inferences/s, `trtexec` throughput |

CPU results use 10 warmups and 100 timed iterations in
`tools/benchmark_jetson.py`. They include image preprocessing, ONNX model
execution, decoding, and class-aware NMS, but exclude image-file decoding and
model loading. The second CPU run's model execution alone averaged 54.98 ms.
The first run's model execution averaged 60.92 ms. The selected ONNX Runtime
provider was `CPUExecutionProvider` only.

The native TensorRT rows use the same preprocessing and NMS code as the CPU
rows through `tools/yolo_inference.py`, but load a device-built FP16 engine.
They also exclude image-file decoding and model loading. The 100-run native
test used 10 warmups and averaged 8.36 ms for inference plus synchronous
host/device transfers; the 1,000-run check used 20 warmups and averaged
8.36 ms for the same segment. The 100-run end-to-end comparison against the
warmed CPU run is 3.85x faster on this one-image workload. It is not a
dataset-throughput or multi-stream camera measurement.

The TensorRT engine was built on this Jetson from the FP32 ONNX model with
`--fp16`. Its cached-engine run used a 2-second warmup and at least 10 seconds
of timing. `trtexec` generated input data for the timed run. Host latency
includes host-to-device and device-to-host transfers (mean 0.155 and 0.039 ms);
GPU compute averaged 2.662 ms. This excludes preprocessing, NMS, image-file
decoding, engine loading, and application overhead. Its throughput is pipelined
and should not be treated as end-to-end detection FPS or directly compared
with the CPU detector's latency. See [NVIDIA's benchmarking guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)
for `trtexec` metric definitions and its default generated-input behavior.

## Correctness check

The repository's `letterbox` function produced a float32 tensor from VOC
test image `000001.jpg`. `trtexec --loadInputs=images:input_000001.bin` ran the
cached engine on that tensor and exported its `[1,24,3549]` output. The same
decoder produced two detections from both ONNX CPU and TensorRT FP16:
class 14 (person) and class 11 (dog). ONNX confidences were 0.9137 and
0.7822; TensorRT confidences were 0.9149 and 0.7773. FP16 numeric differences
are expected. This is a one-image parity check, not a full-dataset TensorRT
accuracy evaluation.
The native Python TensorRT backend also returned the same two classes on
that image using the repository's preprocessing and NMS path.

## Board-input power samples

`tegrastats --interval 500` reported the instantaneous `VDD_IN` rail during
each run. These are internal board estimates, not wall-plug measurements.

| Window | Samples | Mean `VDD_IN` | Range |
| --- | ---: | ---: | ---: |
| Native TensorRT full detector, 13:48:55-13:49:10 | 32 | 5.99 W | 5.94-6.02 W |
| Short idle sample | 11 | 4.60 W | 4.54–4.78 W |
| CPU detector, second run, 13:32:53–58 | 12 | 7.89 W | 7.35–7.99 W |
| TensorRT cached-engine run, 13:31:26–35 | 20 | 11.12 W | 10.81–11.17 W |

The windows were selected from timestamped telemetry during active inference.
No attempt was made to isolate other processes or control ambient temperature.
The model-only `trtexec` power sample used generated inputs, not representative
camera frames. Do not infer energy per *completed detection* from its
model-only power sample.
The native TensorRT detector did use the held-out image and showed about 47%
`GR3D_FREQ` utilization in its active window. Its lower power than the
model-only `trtexec` run reflects a different workload and scheduling pattern.

## Reproduce on the device

The model `best.onnx`, VOC test image `000001.jpg`, and scripts
`tools/benchmark_jetson.py`, `tools/yolo_inference.py`, and
`tools/tensorrt_inference.py` were placed in
`/home/nithish/projects/jetson-yolo-cuda`. The Jetson lacked `python3-venv`,
so an isolated `virtualenv` was bootstrapped inside that project folder:

```bash
cd /home/nithish/projects/jetson-yolo-cuda
python3 -m pip install --target .bootstrap virtualenv
PYTHONPATH=$PWD/.bootstrap python3 -m virtualenv .venv
.venv/bin/python -m pip install numpy==1.26.4 Pillow==10.4.0 onnxruntime==1.23.2
.venv/bin/python benchmark_jetson.py --model best.onnx --image 000001.jpg --format v8 --backend cpu --size 416 --runs 100 --warmup 10 --output cpu_benchmark.json
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best_fp16.engine --fp16 --warmUp=2000 --duration=10 --iterations=100 --exportTimes=trt_fp16_times.json
/usr/src/tensorrt/bin/trtexec --loadEngine=best_fp16.engine --warmUp=2000 --duration=10 --iterations=100 --exportTimes=trt_fp16_cached_times.json
PYTHONPATH=/usr/lib/python3.10/dist-packages .venv/bin/python benchmark_jetson.py --model best_fp16.engine --image 000001.jpg --format v8 --backend tensorrt-native --size 416 --runs 100 --warmup 10 --output trt_native_stream_benchmark.json
```

The serialized TensorRT engine is device/software-stack specific and is not
committed. Raw benchmark JSON, TensorRT timing traces, and `tegrastats` logs
are stored locally under ignored `runs/jetson/`; the Jetson copies remain in
the device project folder. To sample power, run
`tegrastats --interval 500 --logfile trt_power.log` concurrently with the
cached-engine command and stop only the logger you started.

## Remaining work

The installed PyPI ONNX Runtime wheel exposed CPU execution only. A compatible
Jetson GPU-enabled ONNX Runtime build has not been installed or tested, so the
runner's CUDA and TensorRT execution-provider options have no device results.
The native TensorRT engine is integrated with the Python detector and has
end-to-end device timings. Multi-image accuracy validation, repeated runs at
fixed clocks, camera/video pipeline timing, and external power measurement
remain open. The old standalone CUDA post-processing prototype was removed
from the current working tree during the scratch-detector rebuild; it remains
recoverable in Git history.
