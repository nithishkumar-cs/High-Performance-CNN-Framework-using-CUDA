#pragma once

#include <cstddef>

namespace detector {

/** Computes one global spatial mean per channel for one NCHW sample. */
inline void avgpool_forward(const float* input, float* output, int channels, int spatial) {
    for (int c = 0; c < channels; ++c) {
        double total = 0.0;
        for (int p = 0; p < spatial; ++p)
            total += input[static_cast<std::size_t>(c) * spatial + p];
        output[c] = static_cast<float>(total / spatial);
    }
}

/** Adds a global-average-pool gradient uniformly to every spatial input position. */
inline void avgpool_backward(const float* gradient, float* input_gradient, int channels, int spatial) {
    for (int c = 0; c < channels; ++c) {
        const float share = gradient[c] / spatial;
        for (int p = 0; p < spatial; ++p)
            input_gradient[static_cast<std::size_t>(c) * spatial + p] += share;
    }
}

}  // namespace detector
