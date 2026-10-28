#pragma once

#include "detector/network.hpp"
#include <cmath>
#include <cstddef>

namespace detector {

/** Applies one spatially unshared filter bank; weights are location, filter, input-kernel ordered. */
inline void local_forward(const Layer& layer, const float* input, float* output) {
    const int in_plane = layer.input.h * layer.input.w;
    const int out_plane = layer.output.h * layer.output.w;
    const int fan_in = layer.input.c * layer.size * layer.size;
    for (int oc = 0; oc < layer.filters; ++oc)
        for (int oy = 0; oy < layer.output.h; ++oy)
            for (int ox = 0; ox < layer.output.w; ++ox) {
                const int location = oy * layer.output.w + ox;
                const int out_pos = oc * out_plane + location;
                const std::size_t base = (static_cast<std::size_t>(location) * layer.filters + oc) * fan_in;
                float value = layer.biases[static_cast<std::size_t>(out_pos)];
                for (int ic = 0; ic < layer.input.c; ++ic)
                    for (int ky = 0; ky < layer.size; ++ky)
                        for (int kx = 0; kx < layer.size; ++kx) {
                            const int iy = oy * layer.stride + ky - layer.padding;
                            const int ix = ox * layer.stride + kx - layer.padding;
                            if (iy < 0 || iy >= layer.input.h || ix < 0 || ix >= layer.input.w) continue;
                            const int kernel = (ic * layer.size + ky) * layer.size + kx;
                            value += input[static_cast<std::size_t>(ic) * in_plane + iy * layer.input.w + ix] *
                                     layer.weights[base + kernel];
                        }
                output[out_pos] = layer.logistic ? 1.0f / (1.0f + std::exp(-value)) :
                    layer.leaky && value < 0.0f ? value * 0.1f : value;
            }
}

/** Accumulates local-layer input, weight, and bias gradients for one sample. */
inline void local_backward(const Layer& layer, const float* input, const float* output,
                           const float* gradient, float* input_gradient,
                           float* weight_gradient, float* bias_gradient) {
    const int in_plane = layer.input.h * layer.input.w;
    const int out_plane = layer.output.h * layer.output.w;
    const int fan_in = layer.input.c * layer.size * layer.size;
    for (int oc = 0; oc < layer.filters; ++oc)
        for (int oy = 0; oy < layer.output.h; ++oy)
            for (int ox = 0; ox < layer.output.w; ++ox) {
                const int location = oy * layer.output.w + ox;
                const int out_pos = oc * out_plane + location;
                const float value = output[out_pos];
                const float delta = gradient[out_pos] * (layer.logistic ? value * (1.0f - value) :
                    layer.leaky && value < 0.0f ? 0.1f : 1.0f);
                bias_gradient[out_pos] += delta;
                const std::size_t base = (static_cast<std::size_t>(location) * layer.filters + oc) * fan_in;
                for (int ic = 0; ic < layer.input.c; ++ic)
                    for (int ky = 0; ky < layer.size; ++ky)
                        for (int kx = 0; kx < layer.size; ++kx) {
                            const int iy = oy * layer.stride + ky - layer.padding;
                            const int ix = ox * layer.stride + kx - layer.padding;
                            if (iy < 0 || iy >= layer.input.h || ix < 0 || ix >= layer.input.w) continue;
                            const int kernel = (ic * layer.size + ky) * layer.size + kx;
                            const std::size_t in_pos = static_cast<std::size_t>(ic) * in_plane + iy * layer.input.w + ix;
                            weight_gradient[base + kernel] += delta * input[in_pos];
                            if (input_gradient) input_gradient[in_pos] += delta * layer.weights[base + kernel];
                        }
            }
}

}  // namespace detector
