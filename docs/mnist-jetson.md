# Configurable classifier training and Jetson inference

This is a second model family in the project's cfg-driven C++/CUDA graph, alongside the YOLOv3-tiny detection workload. The examples are [linear](../native/examples/mnist-linear.cfg), [two-layer dense](../native/examples/mnist-mlp.cfg), and [convolutional](../native/examples/mnist-conv.cfg) classifiers. All three train with batch-one SGD and run inference on CPU or native CUDA. This is a limited framework milestone, not full Darknet training support.

## Data and reproducibility

The four standard MNIST IDX files were downloaded from the [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist/blob/master/README.md), which links to the original [MNIST database](https://yann.lecun.org/exdb/mnist/index.html), and unpacked without changes. They are preserved locally under ignored `data/mnist/`. MNIST contains 60,000 training and 10,000 test images. The test set was not used to select training duration; each model used a fixed number of epochs.

The Jetson runs used the existing Jetson Orin 25W mode and the project's native CUDA convolution/pooling/dense forward, backward, and SGD code, with no PyTorch or original Darknet trainer. All three runs used seed 2026, learning rate 0.01, weight decay 0, batch size one, 28x28 grayscale inputs normalized to [0,1], and cross-entropy. The linear model trained for five epochs; the 128-hidden-unit leaky-activation MLP and small four-filter CNN trained for three. These are baseline configurations, not tuned leaderboard submissions.

| Model | Epochs | Final training loss | Final training accuracy | Held-out test accuracy | Jetson CUDA mean/sample | Jetson CPU mean/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear 784-to-10 | 5 | 0.280913 | 0.921150 | 0.922900 (9,229/10,000) | 0.147369 ms | 0.012946 ms |
| Dense 784-to-128-to-10 | 3 | 0.076026 | 0.976583 | 0.972400 (9,724/10,000) | 0.274039 ms | 0.151330 ms |
| CNN: 4 3x3 filters, 2x2 pool, dense 10 | 3 | 0.089904 | 0.972267 | 0.972700 (9,727/10,000) | 0.182525 ms | 0.132971 ms |

CPU and CUDA inference produced the same correct-count for each checkpoint on all 10,000 test images. The reported per-sample mean is measured by `cuda_classifier eval` over all samples; it includes byte-to-float conversion and forward execution, but excludes IDX file reading, network construction, and CUDA session setup. There is no separate warmup phase. **CUDA was not faster than CPU** for these tiny models in this measurement; launch/transfer overhead is a plausible explanation, but was not profiled separately. Do not interpret the numbers as camera frame rate or general GPU/CPU performance.

The final checkpoints and plain-text logs are in ignored `runs/mnist/`. SHA-256: linear `e921125132952a9971431cf4baa4ddb89166c1937b02edfa202d4e6a71e1dbf2`; MLP `e1d6afb8bca0ab461b460b03a274b225ec1ef9e378d0b309d714ef3a93f632b8`; CNN `776bd8bac452c7534b879a4ee0edaa15c8358c357feba88883c65e0d465e0faf`. The copied MLP and CNN checkpoints and IDX files reproduced 9,724/10,000 and 9,727/10,000 correct, respectively, with the local Windows CPU build. The Jetson validation copy under `/tmp/darknet-validation.koQJO48n/` is temporary; the local ignored files are the preserved artifacts.

## Commands

Build with `-DDETECTOR_CUDA=ON` on Jetson, then use a **new** output filename to rerun training (the CLI refuses to overwrite a checkpoint that already exists):

```sh
./build/cuda_classifier train native/examples/mnist-linear.cfg \
  data/mnist/train-images-idx3-ubyte data/mnist/train-labels-idx1-ubyte \
  runs/mnist/retrain-linear.weights --backend cuda --epochs 5 --lr 0.01

./build/cuda_classifier train native/examples/mnist-mlp.cfg \
  data/mnist/train-images-idx3-ubyte data/mnist/train-labels-idx1-ubyte \
  runs/mnist/retrain-mlp.weights --backend cuda --epochs 3 --lr 0.01

./build/cuda_classifier train native/examples/mnist-conv.cfg \
  data/mnist/train-images-idx3-ubyte data/mnist/train-labels-idx1-ubyte \
  runs/mnist/retrain-conv.weights --backend cuda --epochs 3 --lr 0.01

./build/cuda_classifier eval native/examples/mnist-conv.cfg \
  data/mnist/t10k-images-idx3-ubyte data/mnist/t10k-labels-idx1-ubyte \
  runs/mnist/mnist-conv-full.weights --backend cuda
```

For CPU-only builds, use `--backend cpu`. The classifier CLI currently reads grayscale IDX datasets; it is not a general image-dataset API. CUDA training supports sequential convolution, max-pooling, and dense layers without batch normalization; branched graphs and YOLO detection training remain unsupported. YOLOv3-tiny's reported VOC weights were trained by original Darknet externally and remain a separate detector benchmark.
