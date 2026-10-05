# CUDA Neural Network Framework (in progress)

An independently written C++/CUDA neural-network framework project inspired by [pjreddie/darknet](https://github.com/pjreddie/darknet). It is not a fork and does not contain Darknet source. The goal is reusable training and inference across multiple models, with Jetson inference as a deployment target. **This goal is not complete yet.** The original fixed-shape MNIST CUDA CNN remains in this repository; `native/` is the newer configurable graph implementation. See the [scope and remaining work](docs/framework-status.md).

The graph currently supports convolution, max-pooling, dense (`[connected]`), route, upsample, YOLO heads, and terminal softmax. Configurable classifier examples include [linear](native/examples/mnist-linear.cfg), [two-layer dense](native/examples/mnist-mlp.cfg), and [convolutional](native/examples/mnist-conv.cfg) models. The IDX/MNIST CLI supports CPU or native CUDA SGD and inference for all three (batch size one, with no trainable batch normalization). The separate experimental YOLO CPU trainer has a simplified loss and is **not** equivalent to Darknet training.

The YOLOv3-tiny application loads compatible Darknet `.weights` on CPU or CUDA and provides letterboxing, two-head decoding, class-aware NMS, benchmarking, and AP@0.5 evaluation. It remains a model validation case, not the whole framework.

For an actual 20-class YOLOv3-tiny model, the [VOC2007 training workflow](docs/yolov3-tiny-training.md) uses the unmodified original Darknet GPU trainer **externally** and evaluates its weights in this native engine. The completed 10,000-batch Jetson run reached **0.57379 validation mAP@0.5** (501 images); that validation-selected final checkpoint reached **0.55534 test mAP@0.5** (4,952 images). This is this project's all-point AP calculation on converted VOC labels, not an official VOC leaderboard score. The final local checkpoint and logs are in ignored `runs/yolov3-tiny-voc/`. No upstream source, original config, pretrained weights, dataset, or trained checkpoint is committed. The [optional ONNX/TensorRT path](docs/onnx-tensorrt.md) exports **these same Darknet weights** and runs FP32/FP16 engines through the same C++ YOLO decoder on Jetson. FP16 reached **0.55595 test mAP@0.5** and 11.970 ms mean in-memory pipeline time; ONNX/TensorRT are not required for native Darknet-format inference.

Build on Jetson Orin with CUDA and OpenCV development libraries:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDETECTOR_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DDETECTOR_OPENCV=ON -DBUILD_TESTING=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

For CPU-only builds, use `-DDETECTOR_CUDA=OFF` and omit `CMAKE_CUDA_ARCHITECTURES`. Without OpenCV, input images must be PPM. Select the CUDA architecture for your GPU.

Train and evaluate a classifier from standard MNIST IDX files on Jetson. The downloaded files are in ignored `data/mnist/`; the [measured full-dataset runs](docs/mnist-jetson.md) reached **97.24% test accuracy** for a two-layer dense model and **97.27%** for a small CNN. Use a fresh output checkpoint name because the CLI refuses to overwrite existing weights:

```sh
./build/cuda_classifier train native/examples/mnist-mlp.cfg data/mnist/train-images-idx3-ubyte data/mnist/train-labels-idx1-ubyte runs/mnist/retrain-mlp.weights --backend cuda --epochs 3 --lr 0.01
./build/cuda_classifier eval native/examples/mnist-mlp.cfg data/mnist/t10k-images-idx3-ubyte data/mnist/t10k-labels-idx1-ubyte runs/mnist/mnist-mlp-full.weights --backend cuda
```

```sh
./build/cuda_detector inspect runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg
./build/cuda_detector detect runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg runs/yolov3-tiny-voc/final.weights image.jpg --backend cuda --names runs/yolov3-tiny-voc/voc.names
./build/cuda_detector benchmark runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg runs/yolov3-tiny-voc/final.weights image.jpg --backend cuda --scope pipeline --warmup 5 --iterations 30
./build/cuda_detector_eval runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg runs/yolov3-tiny-voc/final.weights images/test labels/test --backend cuda --confidence 0.01
```

For TensorRT, export with `native/tools/export_darknet_onnx.py`, build the engine on the Jetson, and configure with `-DDETECTOR_TENSORRT=ON`; see the [tested commands and results](docs/onnx-tensorrt.md). Full YOLOv3 is not required for this path.

Do not pair 80-class COCO weights with the 20-class VOC config. The included `native/tests/voc-smoke.cfg` is **not** YOLOv3-tiny. See [scope and Jetson measurements](docs/native-detector.md) for the implementation's limits.
