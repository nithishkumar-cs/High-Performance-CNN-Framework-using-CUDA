#pragma once

#include <cuda_runtime.h>

namespace detector {

/** Normalizes one flattened sample per block, reducing squared magnitude across its threads. */
static __global__ void l2norm_forward_kernel(const float* input, float* output, int elements_per_sample) {
    const int sample = blockIdx.x;
    const int offset = sample * elements_per_sample;
    __shared__ double partial[256];
    double squared = 0.0;
    for (int i = threadIdx.x; i < elements_per_sample; i += blockDim.x)
        squared += static_cast<double>(input[offset + i]) * input[offset + i];
    partial[threadIdx.x] = squared;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(static_cast<float>(partial[0] + 1e-12));
    for (int i = threadIdx.x; i < elements_per_sample; i += blockDim.x)
        output[offset + i] = input[offset + i] * inverse;
}

/** Adds the L2-normalization Jacobian-vector product for one sample per block. */
static __global__ void l2norm_backward_kernel(const float* input, const float* output,
                                                const float* gradient, float* input_gradient,
                                                int elements_per_sample) {
    const int sample = blockIdx.x;
    const int offset = sample * elements_per_sample;
    __shared__ double partial_projection[256];
    __shared__ double partial_squared[256];
    double projection = 0.0;
    double squared = 0.0;
    for (int i = threadIdx.x; i < elements_per_sample; i += blockDim.x) {
        projection += static_cast<double>(output[offset + i]) * gradient[offset + i];
        squared += static_cast<double>(input[offset + i]) * input[offset + i];
    }
    partial_projection[threadIdx.x] = projection;
    partial_squared[threadIdx.x] = squared;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_projection[threadIdx.x] += partial_projection[threadIdx.x + stride];
            partial_squared[threadIdx.x] += partial_squared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float projected = static_cast<float>(partial_projection[0]);
    const float inverse = rsqrtf(static_cast<float>(partial_squared[0] + 1e-12));
    for (int i = threadIdx.x; i < elements_per_sample; i += blockDim.x)
        input_gradient[offset + i] += inverse * (gradient[offset + i] - output[offset + i] * projected);
}

}  // namespace detector
