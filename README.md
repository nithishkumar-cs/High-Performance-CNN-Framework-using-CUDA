# CUDA Neural Network Framework

An independently written C++/CUDA neural-network framework inspired by [pjreddie/darknet](https://github.com/pjreddie/darknet). It is not a fork and does not contain Darknet source. `native/` contains the configurable framework; the original fixed-shape MNIST CUDA CNN remains in the repository. Training runs on the laptop GPU and Jetson is an inference target. See [current support and limits](docs/framework-status.md).

The graph supports convolution, max-pooling, dense (`[connected]`), route, shortcut, upsample, reorg, dropout, YOLO and region detector heads, softmax, and RNN/LSTM/GRU layers. Classifiers and detectors have CPU and CUDA training paths; recurrent classifiers support sequence inference and truncated backpropagation through time on both. Detector training uses true minibatches, trainable batch normalization, augmentation, a learning-rate schedule, and SGD momentum. Optional cuDNN kernels can replace the default cuBLAS convolution path. Unsupported cfg layers fail explicitly.

Three detector architectures have been trained end to end by this framework on VOC2007 with pretrained backbones. Using the same native CUDA evaluator at confidence 0.01 and class-aware NMS IoU 0.45:

| Native-trained architecture | Validation mAP@0.5, 501 images | Test mAP@0.5, 4,952 images |
| --- | ---: | ---: |
| YOLOv3-tiny | 0.59801 | 0.58182 |
| YOLOv2 / Darknet19 | 0.56054 | 0.57215 |
| Full YOLOv3 / Darknet-53 | 0.62098 | 0.61593 |

These are this project's all-point AP calculations on converted VOC labels excluding difficult objects, not official VOC leaderboard scores. The earlier **external-Darknet** YOLOv3-tiny checkpoint remains a historical comparison (0.57612 validation / 0.55534 test with the comparable native evaluator); it is not evidence of native training. Checkpoints, datasets and run logs are local ignored artifacts. The corrected native YOLOv3-tiny checkpoint was evaluated on Jetson with native CUDA and optional TensorRT FP32/FP16; see [Jetson results](docs/onnx-tensorrt.md).

Build on Jetson Orin with CUDA and OpenCV development libraries:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDETECTOR_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DDETECTOR_OPENCV=ON -DBUILD_TESTING=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

For CPU-only builds, use `-DDETECTOR_CUDA=OFF` and omit `CMAKE_CUDA_ARCHITECTURES`. Add `-DDETECTOR_CUDNN=ON` to an existing CUDA build to use the optional cuDNN kernels after staging the SDK with `native/tools/setup_cudnn.ps1`. Without OpenCV, input images must be PPM. Select the CUDA architecture for your GPU.

Train future models on the laptop GPU. The [deeper CNN example](native/examples/mnist-deep-conv.cfg), trained here for three epochs on all 60,000 MNIST images, reached **98.49% test accuracy** with matching CPU/CUDA evaluation. Historical [full-dataset MNIST Jetson runs](docs/mnist-jetson.md) reached **97.24%** for a two-layer dense model and **97.27%** for a small CNN. Use a fresh output checkpoint name because the CLI refuses to overwrite existing weights:

```sh
./build/cuda_classifier train native/examples/mnist-deep-conv.cfg data/mnist/train-images-idx3-ubyte data/mnist/train-labels-idx1-ubyte runs/mnist/new-deep-conv.weights --backend cuda --epochs 3 --lr 0.01
./build/cuda_classifier eval native/examples/mnist-deep-conv.cfg data/mnist/t10k-images-idx3-ubyte data/mnist/t10k-labels-idx1-ubyte runs/mnist/new-deep-conv.weights --backend cuda
```

```sh
./build/cuda_detector inspect runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg
./build/cuda_detector detect runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg runs/yolov3-tiny-voc/final.weights image.jpg --backend cuda --names runs/yolov3-tiny-voc/voc.names
./build/cuda_detector benchmark runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg runs/yolov3-tiny-voc/final.weights image.jpg --backend cuda --scope pipeline --warmup 5 --iterations 30
./build/cuda_detector_eval runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg runs/yolov3-tiny-voc/final.weights images/test labels/test --backend cuda --confidence 0.01
```

For TensorRT, export with `native/tools/export_darknet_onnx.py`, build the engine on the Jetson, and configure with `-DDETECTOR_TENSORRT=ON`; see the [tested commands and results](docs/onnx-tensorrt.md). Full YOLOv3 is not required for this path.

Detector training accepts `--backend cuda --gpus 0,1 --batch N` on a host with two GPUs; `N` is the total minibatch size. The default is GPU 0. The code synchronizes learned weights, momentum and batch-normalization running state after each global step. This laptop has one RTX 3070 Ti, so two-device execution and throughput remain unmeasured. Its one-GPU path and synchronization math have CUDA regression tests.

Do not pair 80-class COCO weights with the 20-class VOC config. The included `native/tests/voc-smoke.cfg` is **not** YOLOv3-tiny. See [current framework status](docs/framework-status.md) for the implementation's limits.
