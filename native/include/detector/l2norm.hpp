#pragma once

#include <cmath>
#include <cstddef>

namespace detector {

/** Normalizes one sample's complete flattened tensor and returns its inverse norm. */
inline float l2norm_forward(const float* input, float* output, std::size_t count) {
    double squared = 0.0;
    for (std::size_t i = 0; i < count; ++i) squared += static_cast<double>(input[i]) * input[i];
    const float inverse = 1.0f / static_cast<float>(std::sqrt(squared + 1e-12));
    for (std::size_t i = 0; i < count; ++i) output[i] = input[i] * inverse;
    return inverse;
}

/** Adds the Jacobian-vector product for sample-wide L2 normalization to input_gradient. */
inline void l2norm_backward(const float* output, const float* gradient, float* input_gradient,
                            std::size_t count, float inverse) {
    double projection = 0.0;
    for (std::size_t i = 0; i < count; ++i)
        projection += static_cast<double>(output[i]) * gradient[i];
    const float projected = static_cast<float>(projection);
    for (std::size_t i = 0; i < count; ++i)
        input_gradient[i] += inverse * (gradient[i] - output[i] * projected);
}

}  // namespace detector
