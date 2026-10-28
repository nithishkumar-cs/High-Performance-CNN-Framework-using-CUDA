#pragma once

#include <cuda_runtime.h>

namespace detector {

/** Reduces one sample/channel plane per block into its global average. */
static __global__ void avgpool_forward_kernel(const float* input, float* output, int channels, int spatial) {
    const int channel = blockIdx.x % channels;
    const int sample = blockIdx.x / channels;
    const int offset = (sample * channels + channel) * spatial;
    __shared__ double partial[256];
    double total = 0.0;
    for (int p = threadIdx.x; p < spatial; p += blockDim.x) total += input[offset + p];
    partial[threadIdx.x] = total;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[sample * channels + channel] = static_cast<float>(partial[0] / spatial);
}

/** Adds each channel's mean gradient to every position in its input plane. */
static __global__ void avgpool_backward_kernel(const float* gradient, float* input_gradient,
                                                int spatial, int count) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= count) return;
    input_gradient[position] += gradient[position / spatial] / spatial;
}

}  // namespace detector
