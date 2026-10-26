#include "detector/cudnn_backend.hpp"

#include <cudnn.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace detector {
namespace {

void check(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(status));
    }
}

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

// Owns one cuDNN resource and frees it (if non-null) on scope exit - avoids leaking
// descriptors/memory on an exception thrown mid-setup by any `check()` above it.
template <typename Handle, cudnnStatus_t (*Destroy)(Handle)>
struct Scoped {
    Handle handle = nullptr;
    ~Scoped() { if (handle) Destroy(handle); }
};

}  // namespace

int cudnn_runtime_version() {
    cudnnHandle_t handle = nullptr;
    check(cudnnCreate(&handle), "cudnnCreate");
    int version = static_cast<int>(cudnnGetVersion());
    check(cudnnDestroy(handle), "cudnnDestroy");
    return version;
}

void cudnn_convolution_forward(const float* input, const float* weights, float* output,
                                int in_channels, int in_h, int in_w, int out_channels,
                                int kernel, int stride, int pad) {
    const int out_h = (in_h + 2 * pad - kernel) / stride + 1;
    const int out_w = (in_w + 2 * pad - kernel) / stride + 1;

    Scoped<cudnnHandle_t, cudnnDestroy> handle;
    check(cudnnCreate(&handle.handle), "cudnnCreate");

    Scoped<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor> input_desc;
    check(cudnnCreateTensorDescriptor(&input_desc.handle), "cudnnCreateTensorDescriptor(input)");
    check(cudnnSetTensor4dDescriptor(input_desc.handle, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                      1, in_channels, in_h, in_w), "cudnnSetTensor4dDescriptor(input)");

    Scoped<cudnnFilterDescriptor_t, cudnnDestroyFilterDescriptor> filter_desc;
    check(cudnnCreateFilterDescriptor(&filter_desc.handle), "cudnnCreateFilterDescriptor");
    check(cudnnSetFilter4dDescriptor(filter_desc.handle, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                       out_channels, in_channels, kernel, kernel), "cudnnSetFilter4dDescriptor");

    Scoped<cudnnConvolutionDescriptor_t, cudnnDestroyConvolutionDescriptor> conv_desc;
    check(cudnnCreateConvolutionDescriptor(&conv_desc.handle), "cudnnCreateConvolutionDescriptor");
    check(cudnnSetConvolution2dDescriptor(conv_desc.handle, pad, pad, stride, stride, 1, 1,
                                            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "cudnnSetConvolution2dDescriptor");

    Scoped<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor> output_desc;
    check(cudnnCreateTensorDescriptor(&output_desc.handle), "cudnnCreateTensorDescriptor(output)");
    check(cudnnSetTensor4dDescriptor(output_desc.handle, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       1, out_channels, out_h, out_w), "cudnnSetTensor4dDescriptor(output)");

    const cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    std::size_t workspace_bytes = 0;
    check(cudnnGetConvolutionForwardWorkspaceSize(handle.handle, input_desc.handle, filter_desc.handle,
                                                    conv_desc.handle, output_desc.handle, algo, &workspace_bytes),
          "cudnnGetConvolutionForwardWorkspaceSize");
    void* workspace = nullptr;
    if (workspace_bytes > 0) check(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc(workspace)");

    const float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t status = cudnnConvolutionForward(handle.handle, &alpha, input_desc.handle, input,
                                                     filter_desc.handle, weights, conv_desc.handle, algo,
                                                     workspace, workspace_bytes, &beta, output_desc.handle, output);
    if (workspace) cudaFree(workspace);
    check(status, "cudnnConvolutionForward");
}

}  // namespace detector
