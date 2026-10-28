#pragma once

#include <cstddef>

/** Evaluates spatially unshared filters for a sample-major NCHW batch. */
static __global__ void local_forward_kernel(const float* input, const float* weights,
                                            const float* biases, float* output, int total,
                                            int channels, int ih, int iw, int filters,
                                            int oh, int ow, int kernel, int stride, int pad,
                                            bool leaky, bool logistic) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total) return;
    const int locations = oh * ow;
    const int sample = pos / (filters * locations);
    const int oc = pos / locations % filters;
    const int location = pos % locations;
    const int oy = location / ow, ox = location % ow;
    const int fan_in = channels * kernel * kernel;
    const std::size_t base = (static_cast<std::size_t>(location) * filters + oc) * fan_in;
    float value = biases[oc * locations + location];
    for (int ic = 0; ic < channels; ++ic)
        for (int ky = 0; ky < kernel; ++ky)
            for (int kx = 0; kx < kernel; ++kx) {
                const int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
                value += input[(static_cast<std::size_t>(sample) * channels + ic) * ih * iw + iy * iw + ix] *
                         weights[base + (ic * kernel + ky) * kernel + kx];
            }
    output[pos] = logistic ? 1.0f / (1.0f + expf(-value)) :
        leaky && value < 0.0f ? value * 0.1f : value;
}

/** Multiplies an output gradient by the local layer's activation derivative. */
static __global__ void local_activation_backward_kernel(const float* output, float* gradient,
                                                         int total, bool leaky, bool logistic) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total) return;
    const float value = output[pos];
    gradient[pos] *= logistic ? value * (1.0f - value) : leaky && value < 0.0f ? 0.1f : 1.0f;
}

/** Adds local-layer input gradients without inter-thread atomics. */
static __global__ void local_backward_input_kernel(const float* gradient, const float* weights,
                                                   float* input_gradient, int total,
                                                   int channels, int ih, int iw, int filters,
                                                   int oh, int ow, int kernel, int stride, int pad) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total) return;
    const int sample = pos / (channels * ih * iw);
    const int ic = pos / (ih * iw) % channels;
    const int iy = pos / iw % ih, ix = pos % iw;
    const int locations = oh * ow;
    const int fan_in = channels * kernel * kernel;
    float value = 0.0f;
    for (int oc = 0; oc < filters; ++oc)
        for (int ky = 0; ky < kernel; ++ky)
            for (int kx = 0; kx < kernel; ++kx) {
                const int row = iy + pad - ky, col = ix + pad - kx;
                if (row < 0 || col < 0 || row % stride || col % stride) continue;
                const int oy = row / stride, ox = col / stride;
                if (oy >= oh || ox >= ow) continue;
                const int location = oy * ow + ox;
                const std::size_t wi = (static_cast<std::size_t>(location) * filters + oc) * fan_in +
                    (ic * kernel + ky) * kernel + kx;
                value += gradient[(static_cast<std::size_t>(sample) * filters + oc) * locations + location] *
                         weights[wi];
            }
    input_gradient[pos] += value;
}

/** Computes one spatially unshared weight's gradient over all samples. */
static __device__ float local_weight_delta(const float* input, const float* gradient,
                                           int pos, int samples, int channels, int ih, int iw,
                                           int filters, int oh, int ow, int kernel,
                                           int stride, int pad) {
    const int fan_in = channels * kernel * kernel;
    const int locations = oh * ow;
    const int kx = pos % kernel, ky = pos / kernel % kernel;
    const int ic = pos / (kernel * kernel) % channels;
    const int oc = pos / fan_in % filters;
    const int location = pos / (fan_in * filters);
    const int iy = location / ow * stride + ky - pad;
    const int ix = location % ow * stride + kx - pad;
    if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) return 0.0f;
    float sum = 0.0f;
    for (int sample = 0; sample < samples; ++sample)
        sum += gradient[(static_cast<std::size_t>(sample) * filters + oc) * locations + location] *
               input[(static_cast<std::size_t>(sample) * channels + ic) * ih * iw + iy * iw + ix];
    return sum;
}

/** Accumulates local weight gradients across a minibatch. */
static __global__ void local_weight_gradient_kernel(const float* input, const float* gradient,
                                                    float* weight_gradient, int count, int samples,
                                                    int channels, int ih, int iw, int filters,
                                                    int oh, int ow, int kernel, int stride, int pad) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    weight_gradient[pos] += local_weight_delta(input, gradient, pos, samples, channels, ih, iw,
        filters, oh, ow, kernel, stride, pad);
}

/** Accumulates location-specific bias gradients across a minibatch. */
static __global__ void local_bias_gradient_kernel(const float* gradient, float* bias_gradient,
                                                  int outputs, int samples) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= outputs) return;
    float sum = 0.0f;
    for (int sample = 0; sample < samples; ++sample)
        sum += gradient[static_cast<std::size_t>(sample) * outputs + pos];
    bias_gradient[pos] += sum;
}

/** Applies one averaged classifier SGD update to a local weight. */
static __global__ void local_sgd_weights_kernel(float* weights, const float* input,
                                                const float* gradient, int count, int samples,
                                                int channels, int ih, int iw, int filters,
                                                int oh, int ow, int kernel, int stride, int pad,
                                                float learning_rate, float decay) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    const float old = weights[pos];
    const float delta = local_weight_delta(input, gradient, pos, samples, channels, ih, iw,
        filters, oh, ow, kernel, stride, pad);
    weights[pos] = old - learning_rate * (delta / samples + decay * old);
}

/** Applies one averaged classifier SGD update to a location-specific bias. */
static __global__ void local_sgd_biases_kernel(float* biases, const float* gradient,
                                               int outputs, int samples, float learning_rate) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= outputs) return;
    float delta = 0.0f;
    for (int sample = 0; sample < samples; ++sample)
        delta += gradient[static_cast<std::size_t>(sample) * outputs + pos];
    biases[pos] -= learning_rate * delta / samples;
}
