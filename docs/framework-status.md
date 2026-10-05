# Framework scope and status

The project goal is an independently implemented C++/CUDA neural-network framework that can configure, train, save, load, and run more than one model, plus measured Jetson inference. It is inspired by [pjreddie/darknet](https://github.com/pjreddie/darknet), not a fork or a claim of Darknet source compatibility beyond the supported layer/weight subset. YOLOv3-tiny is one validation workload; the original Darknet trainer was used externally for the reported VOC2007 weights.

## Implemented now

- Config-driven inference graph with convolution, max-pooling, dense (`[connected]`), route, upsample, YOLO heads, and terminal softmax. Unsupported layers fail explicitly. Classifiers can use different layer topologies and class counts; YOLOv3-tiny inference is retained.
- CPU reference SGD training for simple convolutional/dense/softmax classifiers with cross-entropy and batch size one, with fixed batch-normalization statistics/scales. CUDA SGD training supports sequential convolution, max-pooling, and dense layers followed by softmax, also at batch size one; batch normalization is not supported in its training path. A separate CPU YOLO trainer uses a simplified objective; it has not trained a useful YOLOv3-tiny model.
- Darknet-style weight load/save for the supported layers, an IDX/MNIST dataset adapter, and a classifier CLI for CPU training and CPU/CUDA inference. Config examples are in `native/examples/`.
- Jetson native CUDA inference for classifiers and YOLOv3-tiny; the classifier tests compare CPU and CUDA probabilities and one-step parameter updates. Linear, MLP, and convolutional classifiers were trained end-to-end with this project's CUDA trainer on full MNIST and evaluated on the held-out test set; see the [MNIST Jetson report](mnist-jetson.md). The YOLOv3-tiny path also has optional ONNX export and TensorRT inference, with [Jetson measurements](onnx-tensorrt.md).

## Not yet implemented or proven

- CUDA backward/update for route, upsample, YOLO heads, and trainable batch normalization. The current CUDA trainer supports sequential classifiers, not general branched graphs or detector training.
- General minibatch training, trainable batch-normalization statistics/scales, additional losses/optimizers, data augmentation, and the layer coverage needed for full YOLOv3/Darknet-53 or arbitrary upstream configs.
- Faithful Darknet YOLO training and a model trained end-to-end by this framework with competitive VOC accuracy. The existing VOC2007 YOLOv3-tiny checkpoint was trained by **original Darknet**, not by this framework.
- A broad or tuned classifier benchmark. The measured MNIST results are for fixed-epoch linear, MLP, and small CNN models; per-sample timing includes transfer overhead and has no warmup. The synthetic 4×4 tests cover convergence and CPU/CUDA parity across configs.

The next substantial milestone is reusable CUDA training for branched graphs, trainable batch normalization, and a faithful YOLO detection loss/target pipeline. Keep the full-dataset MNIST and YOLOv3-tiny evaluations as regression benchmarks.
