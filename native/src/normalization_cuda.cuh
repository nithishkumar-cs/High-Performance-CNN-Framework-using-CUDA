#pragma once

#include <cstddef>

/** Applies channel-local response normalization to a sample-major NCHW batch. */
static __global__ void normalization_forward_kernel(const float* input, float* output,
                                                     float* norms, int elements,
                                                     int channels, int spatial, int window,
                                                     float alpha, float beta, float kappa) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= elements) return;
    const int c = pos / spatial % channels;
    const int sample_base = pos / (channels * spatial) * channels * spatial;
    const int p = pos % spatial;
    const int begin = max(0, c - window / 2);
    const int end = min(channels, c + (window - 1) / 2 + 1);
    float squared = 0.0f;
    for (int neighbor = begin; neighbor < end; ++neighbor) {
        const float value = input[sample_base + neighbor * spatial + p];
        squared += value * value;
    }
    const float norm = kappa + alpha * squared;
    if (norms) norms[pos] = norm;
    output[pos] = input[pos] * powf(norm, -beta);
}

/** Adds the channel-local normalization Jacobian-vector product for a batch. */
static __global__ void normalization_backward_kernel(const float* input, const float* output,
                                                      const float* norms, const float* gradient,
                                                      float* input_gradient, int elements,
                                                      int channels, int spatial, int window,
                                                      float alpha, float beta) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= elements) return;
    const int c = pos / spatial % channels;
    const int sample_base = pos / (channels * spatial) * channels * spatial;
    const int p = pos % spatial;
    const int begin = max(0, c - (window - 1) / 2);
    const int end = min(channels, c + window / 2 + 1);
    float cross = 0.0f;
    for (int neighbor = begin; neighbor < end; ++neighbor) {
        const int other = sample_base + neighbor * spatial + p;
        cross += gradient[other] * output[other] / norms[other];
    }
    input_gradient[pos] += gradient[pos] * powf(norms[pos], -beta) -
        2.0f * alpha * beta * input[pos] * cross;
}
