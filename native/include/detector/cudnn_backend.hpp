#pragma once

#include <cstddef>

#include <memory>

// Optional cuDNN convolution and batch-normalization primitives. Tensor storage is NCHW.

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

/** Caches cuDNN descriptors, handle and workspace for one convolution shape. */
class CudnnConvolution {
public:
    /** Prepares descriptors for one filter and spatial shape. */
    CudnnConvolution(int in_channels, int in_h, int in_w, int out_channels,
                     int kernel, int stride, int pad);
    /** Releases the cuDNN descriptors and workspace. */
    ~CudnnConvolution();
    CudnnConvolution(const CudnnConvolution&) = delete;
    CudnnConvolution& operator=(const CudnnConvolution&) = delete;
    /** Writes n contiguous NCHW samples' raw, bias-free convolution to output. */
    void forward(const float* input, const float* weights, float* output, int n = 1);
    /** Adds one minibatch's filter gradient to the existing accumulator. */
    void backward_filter(const float* input, const float* output_gradient, float* weight_gradient, int n = 1);
    /** Adds one minibatch's input gradient to the existing destination. */
    void backward_data(const float* weights, const float* output_gradient, float* input_gradient, int n = 1);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/** Caches cuDNN tensor descriptors for minibatch spatial batch normalization. */
class CudnnBatchNorm {
public:
    /** Prepares a spatial batch-normalization descriptor for one feature-map shape. */
    CudnnBatchNorm(int channels, int height, int width);
    /** Releases the cuDNN descriptors. */
    ~CudnnBatchNorm();
    CudnnBatchNorm(const CudnnBatchNorm&) = delete;
    CudnnBatchNorm& operator=(const CudnnBatchNorm&) = delete;
    /** Normalizes and affinely transforms n samples, saving mean and inverse variance for backward. */
    void forward(const float* raw, const float* scale, const float* bias, float* output,
                 float* running_mean, float* running_variance, float* saved_mean,
                 float* saved_inv_variance, int n) const;
    /** Writes raw-input gradient and adds scale/bias gradients for n samples. */
    void backward(const float* raw, const float* output_gradient, const float* scale,
                  const float* saved_mean, const float* saved_inv_variance,
                  float* raw_gradient, float* scale_gradient, float* bias_gradient, int n) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace detector
