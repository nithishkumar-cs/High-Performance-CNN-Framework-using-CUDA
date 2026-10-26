#pragma once

#include <cstddef>

// cuDNN-accelerated kernel surface, built only when DETECTOR_CUDNN=ON. Ports the
// inference-path convolution (native/src/cuda_backend.cu's single-sample,
// NCHW, no-bias `convolution` kernel - BN scale/offset are applied separately
// by the caller) to cudnnConvolutionForward. The training path's existing
// cuBLAS im2col/col2im convolution (cuda_training.cu) and every batch-norm
// kernel are NOT yet ported - that is a separate, larger follow-on item (see
// DESIGN.yaml todo) since it touches the parity-tested CudaTrainer hot path.

namespace detector {

// Returns cuDNN's runtime version (CUDNN_VERSION-style integer, e.g. 90400) by
// creating and immediately destroying a handle, or throws std::runtime_error if
// cuDNN failed to initialize (e.g. its DLLs are not on PATH at run time).
int cudnn_runtime_version();

// Single-sample (N=1), NCHW, no-bias, square-kernel 2D convolution forward via
// cudnnConvolutionForward - the same math as cuda_backend.cu's `convolution`
// kernel before its multiplier/offset/leaky-activation epilogue (the caller
// applies those separately, matching that kernel's split of concerns). Creates
// and destroys its own handle/descriptors/workspace per call (correctness-first;
// not yet optimized for repeated use in a hot loop - see the header comment above).
// `weights` layout is [out_channels][in_channels][kernel][kernel], matching this
// project's Darknet-format weight layout. Throws std::runtime_error on any cuDNN
// or CUDA failure.
void cudnn_convolution_forward(const float* input, const float* weights, float* output,
                                int in_channels, int in_h, int in_w, int out_channels,
                                int kernel, int stride, int pad);

}  // namespace detector
