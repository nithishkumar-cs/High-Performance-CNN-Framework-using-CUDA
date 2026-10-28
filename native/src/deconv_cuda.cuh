#pragma once

#include <cstddef>

/** Evaluates a transposed convolution for a sample-major NCHW batch by gathering contributions. */
static __global__ void deconv_forward_kernel(const float* input, const float* weights,
                                             const float* biases, float* output, int total,
                                             int channels, int ih, int iw, int filters,
                                             int oh, int ow, int kernel, int stride, int pad,
                                             bool leaky, bool logistic) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total) return;
    const int sample = pos / (filters * oh * ow);
    const int oc = pos / (oh * ow) % filters;
    const int oy = pos / ow % oh, ox = pos % ow;
    float value = biases[oc];
    for (int ky = 0; ky < kernel; ++ky) {
        const int row = oy + pad - ky;
        if (row < 0 || row % stride) continue;
        const int iy = row / stride;
        if (iy >= ih) continue;
        for (int kx = 0; kx < kernel; ++kx) {
            const int col = ox + pad - kx;
            if (col < 0 || col % stride) continue;
            const int ix = col / stride;
            if (ix >= iw) continue;
            for (int ic = 0; ic < channels; ++ic)
                value += input[(static_cast<std::size_t>(sample) * channels + ic) * ih * iw + iy * iw + ix] *
                         weights[((static_cast<std::size_t>(ic) * filters + oc) * kernel + ky) * kernel + kx];
        }
    }
    output[pos] = logistic ? 1.0f / (1.0f + expf(-value)) :
        leaky && value < 0.0f ? value * 0.1f : value;
}

/** Multiplies an output gradient by the deconvolution activation derivative. */
static __global__ void deconv_activation_backward_kernel(const float* output, float* gradient,
                                                         int total, bool leaky, bool logistic) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total) return;
    const float value = output[pos];
    gradient[pos] *= logistic ? value * (1.0f - value) : leaky && value < 0.0f ? 0.1f : 1.0f;
}

/** Adds the deconvolution input gradient; each thread owns one input element. */
static __global__ void deconv_backward_input_kernel(const float* gradient, const float* weights,
                                                    float* input_gradient, int total,
                                                    int channels, int ih, int iw, int filters,
                                                    int oh, int ow, int kernel, int stride, int pad) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total) return;
    const int sample = pos / (channels * ih * iw);
    const int ic = pos / (ih * iw) % channels;
    const int iy = pos / iw % ih, ix = pos % iw;
    float value = 0.0f;
    for (int oc = 0; oc < filters; ++oc)
        for (int ky = 0; ky < kernel; ++ky) {
            const int oy = iy * stride + ky - pad;
            if (oy < 0 || oy >= oh) continue;
            for (int kx = 0; kx < kernel; ++kx) {
                const int ox = ix * stride + kx - pad;
                if (ox < 0 || ox >= ow) continue;
                value += gradient[(static_cast<std::size_t>(sample) * filters + oc) * oh * ow + oy * ow + ox] *
                         weights[((static_cast<std::size_t>(ic) * filters + oc) * kernel + ky) * kernel + kx];
            }
        }
    input_gradient[pos] += value;
}

/** Computes one deconvolution weight's gradient over all samples and input positions. */
static __device__ float deconv_weight_delta(const float* input, const float* gradient, int pos,
                                            int samples, int channels, int ih, int iw, int filters,
                                            int oh, int ow, int kernel, int stride, int pad) {
    const int kx = pos % kernel, ky = pos / kernel % kernel;
    const int oc = pos / (kernel * kernel) % filters;
    const int ic = pos / (kernel * kernel * filters);
    float sum = 0.0f;
    for (int iy = 0; iy < ih; ++iy) {
        const int oy = iy * stride + ky - pad;
        if (oy < 0 || oy >= oh) continue;
        for (int ix = 0; ix < iw; ++ix) {
            const int ox = ix * stride + kx - pad;
            if (ox < 0 || ox >= ow) continue;
            for (int sample = 0; sample < samples; ++sample)
                sum += gradient[(static_cast<std::size_t>(sample) * filters + oc) * oh * ow + oy * ow + ox] *
                       input[(static_cast<std::size_t>(sample) * channels + ic) * ih * iw + iy * iw + ix];
        }
    }
    return sum;
}

/** Accumulates deconvolution weight gradients across a minibatch. */
static __global__ void deconv_weight_gradient_kernel(const float* input, const float* gradient,
                                                     float* weight_gradient, int count, int samples,
                                                     int channels, int ih, int iw, int filters,
                                                     int oh, int ow, int kernel, int stride, int pad) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    weight_gradient[pos] += deconv_weight_delta(input, gradient, pos, samples, channels, ih, iw,
        filters, oh, ow, kernel, stride, pad);
}

/** Accumulates per-filter bias gradients across a minibatch. */
static __global__ void deconv_bias_gradient_kernel(const float* gradient, float* bias_gradient,
                                                   int filters, int plane, int samples) {
    const int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= filters) return;
    float sum = 0.0f;
    for (int sample = 0; sample < samples; ++sample)
        for (int p = 0; p < plane; ++p)
            sum += gradient[(static_cast<std::size_t>(sample) * filters + oc) * plane + p];
    bias_gradient[oc] += sum;
}

/** Applies one averaged classifier SGD update to a deconvolution weight. */
static __global__ void deconv_sgd_weights_kernel(float* weights, const float* input,
                                                 const float* gradient, int count, int samples,
                                                 int channels, int ih, int iw, int filters,
                                                 int oh, int ow, int kernel, int stride, int pad,
                                                 float learning_rate, float decay) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    const float old = weights[pos];
    const float delta = deconv_weight_delta(input, gradient, pos, samples, channels, ih, iw,
        filters, oh, ow, kernel, stride, pad);
    weights[pos] = old - learning_rate * (delta / samples + decay * old);
}

/** Applies one averaged classifier SGD update to a per-filter bias. */
static __global__ void deconv_sgd_biases_kernel(float* biases, const float* gradient, int filters,
                                                int plane, int samples, float learning_rate) {
    const int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= filters) return;
    float delta = 0.0f;
    for (int sample = 0; sample < samples; ++sample)
        for (int p = 0; p < plane; ++p)
            delta += gradient[(static_cast<std::size_t>(sample) * filters + oc) * plane + p];
    biases[oc] -= learning_rate * delta / samples;
}
