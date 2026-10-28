#pragma once

#include <cmath>
#include <cstddef>

namespace detector {

/** Applies channel-local response normalization at every spatial position. */
inline void normalization_forward(const float* input, float* output, float* norms,
                                  int channels, int spatial, int window,
                                  float alpha, float beta, float kappa) {
    for (int c = 0; c < channels; ++c) {
        const int begin = c - window / 2 < 0 ? 0 : c - window / 2;
        const int end = c + (window - 1) / 2 + 1 < channels ?
            c + (window - 1) / 2 + 1 : channels;
        for (int p = 0; p < spatial; ++p) {
            float squared = 0.0f;
            for (int neighbor = begin; neighbor < end; ++neighbor) {
                const float value = input[static_cast<std::size_t>(neighbor) * spatial + p];
                squared += value * value;
            }
            const std::size_t index = static_cast<std::size_t>(c) * spatial + p;
            const float norm = kappa + alpha * squared;
            if (norms) norms[index] = norm;
            output[index] = input[index] * std::pow(norm, -beta);
        }
    }
}

/** Adds the exact Jacobian-vector product of channel-local normalization. */
inline void normalization_backward(const float* input, const float* output,
                                   const float* norms, const float* gradient,
                                   float* input_gradient, int channels, int spatial,
                                   int window, float alpha, float beta) {
    for (int c = 0; c < channels; ++c) {
        const int begin = c - (window - 1) / 2 < 0 ? 0 : c - (window - 1) / 2;
        const int end = c + window / 2 + 1 < channels ? c + window / 2 + 1 : channels;
        for (int p = 0; p < spatial; ++p) {
            const std::size_t index = static_cast<std::size_t>(c) * spatial + p;
            float cross = 0.0f;
            for (int neighbor = begin; neighbor < end; ++neighbor) {
                const std::size_t other = static_cast<std::size_t>(neighbor) * spatial + p;
                cross += gradient[other] * output[other] / norms[other];
            }
            input_gradient[index] += gradient[index] * std::pow(norms[index], -beta) -
                2.0f * alpha * beta * input[index] * cross;
        }
    }
}

}  // namespace detector
