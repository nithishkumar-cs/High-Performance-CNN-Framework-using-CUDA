#pragma once

#include "detector/network.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

namespace detector {

/** Flat index of weight (input channel, filter, ky, kx); transposed-convolution layout. */
inline std::size_t deconv_weight_index(const Layer& layer, int ic, int oc, int ky, int kx) {
    return ((static_cast<std::size_t>(ic) * layer.filters + oc) * layer.size + ky) * layer.size + kx;
}

/** Transposed convolution: each input value scatters a size x size weighted patch into the output. */
inline void deconvolutional_forward(const Layer& layer, const float* input, float* output) {
    const int in_plane = layer.input.h * layer.input.w;
    const int out_plane = layer.output.h * layer.output.w;
    for (int oc = 0; oc < layer.filters; ++oc)
        for (int p = 0; p < out_plane; ++p) output[oc * out_plane + p] = layer.biases[static_cast<std::size_t>(oc)];
    for (int ic = 0; ic < layer.input.c; ++ic)
        for (int iy = 0; iy < layer.input.h; ++iy)
            for (int ix = 0; ix < layer.input.w; ++ix) {
                const float value = input[ic * in_plane + iy * layer.input.w + ix];
                for (int oc = 0; oc < layer.filters; ++oc)
                    for (int ky = 0; ky < layer.size; ++ky) {
                        const int oy = iy * layer.stride + ky - layer.padding;
                        if (oy < 0 || oy >= layer.output.h) continue;
                        for (int kx = 0; kx < layer.size; ++kx) {
                            const int ox = ix * layer.stride + kx - layer.padding;
                            if (ox < 0 || ox >= layer.output.w) continue;
                            output[oc * out_plane + oy * layer.output.w + ox] +=
                                value * layer.weights[deconv_weight_index(layer, ic, oc, ky, kx)];
                        }
                    }
            }
    for (int i = 0; i < layer.filters * out_plane; ++i) {
        const float value = output[i];
        output[i] = layer.logistic ? 1.0f / (1.0f + std::exp(-value)) :
            layer.leaky && value < 0.0f ? value * 0.1f : value;
    }
}

/** Accumulates deconvolution weight and bias gradients and, if requested, the input gradient.
 *  `gradient` is the upstream output gradient and is not modified. */
inline void deconvolutional_backward(const Layer& layer, const float* input, const float* output,
                                     const float* gradient, float* input_gradient,
                                     float* weight_gradient, float* bias_gradient) {
    const int in_plane = layer.input.h * layer.input.w;
    const int out_plane = layer.output.h * layer.output.w;
    std::vector<float> delta(static_cast<std::size_t>(layer.filters) * out_plane);
    for (std::size_t i = 0; i < delta.size(); ++i) {
        const float value = output[i];
        delta[i] = gradient[i] * (layer.logistic ? value * (1.0f - value) :
            layer.leaky && value < 0.0f ? 0.1f : 1.0f);
        bias_gradient[i / out_plane] += delta[i];
    }
    for (int ic = 0; ic < layer.input.c; ++ic)
        for (int iy = 0; iy < layer.input.h; ++iy)
            for (int ix = 0; ix < layer.input.w; ++ix) {
                const std::size_t in_pos = static_cast<std::size_t>(ic) * in_plane + iy * layer.input.w + ix;
                float sum = 0.0f;
                for (int oc = 0; oc < layer.filters; ++oc)
                    for (int ky = 0; ky < layer.size; ++ky) {
                        const int oy = iy * layer.stride + ky - layer.padding;
                        if (oy < 0 || oy >= layer.output.h) continue;
                        for (int kx = 0; kx < layer.size; ++kx) {
                            const int ox = ix * layer.stride + kx - layer.padding;
                            if (ox < 0 || ox >= layer.output.w) continue;
                            const float d = delta[static_cast<std::size_t>(oc) * out_plane + oy * layer.output.w + ox];
                            const std::size_t w = deconv_weight_index(layer, ic, oc, ky, kx);
                            weight_gradient[w] += d * input[in_pos];
                            sum += d * layer.weights[w];
                        }
                    }
                if (input_gradient) input_gradient[in_pos] += sum;
            }
}

}  // namespace detector
