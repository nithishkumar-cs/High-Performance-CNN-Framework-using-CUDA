// Link/runtime smoke test plus a correctness parity test for the DETECTOR_CUDNN
// build: proves the pip-installed cuDNN wheel's headers/import-lib/DLLs actually
// link and load (cudnn_runtime_version), and that cudnn_convolution_forward
// matches a hand-written reference convolution (same math as cuda_backend.cu's
// `convolution` kernel before its multiplier/offset/leaky-activation epilogue).
#include "detector/cudnn_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/** Reports CUDA allocation and transfer failures in the cuDNN smoke test. */
void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

// Direct reference convolution (no bias, NCHW, N=1) - independently re-derived here
// rather than reusing cuda_backend.cu's device kernel, so this test does not just
// compare cuDNN against itself.
/** Independently computes a no-bias NCHW convolution for forward parity. */
std::vector<float> reference_convolution(const std::vector<float>& input, const std::vector<float>& weights,
                                          int in_c, int in_h, int in_w, int out_c, int kernel, int stride, int pad) {
    const int out_h = (in_h + 2 * pad - kernel) / stride + 1;
    const int out_w = (in_w + 2 * pad - kernel) / stride + 1;
    std::vector<float> output(static_cast<std::size_t>(out_c) * out_h * out_w, 0.0f);
    for (int oc = 0; oc < out_c; ++oc)
        for (int y = 0; y < out_h; ++y)
            for (int x = 0; x < out_w; ++x) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_c; ++ic)
                    for (int ky = 0; ky < kernel; ++ky)
                        for (int kx = 0; kx < kernel; ++kx) {
                            int iy = y * stride + ky - pad;
                            int ix = x * stride + kx - pad;
                            if (iy < 0 || ix < 0 || iy >= in_h || ix >= in_w) continue;
                            std::size_t wi = ((static_cast<std::size_t>(oc) * in_c + ic) * kernel + ky) * kernel + kx;
                            sum += input[(static_cast<std::size_t>(ic) * in_h + iy) * in_w + ix] * weights[wi];
                        }
                output[(static_cast<std::size_t>(oc) * out_h + y) * out_w + x] = sum;
            }
    return output;
}

}  // namespace

/** Checks cuDNN convolution forward and both backward paths against direct sums. */
int main() {
    const int version = detector::cudnn_runtime_version();
    if (version <= 0) throw std::runtime_error("invalid cuDNN runtime version");
    std::printf("cudnn_smoke: cuDNN runtime version %d\n", version);

    // 2 input channels, 3 output channels, 3x3 kernel, stride 1, pad 1, 5x4 image -
    // asymmetric dimensions and multiple channels to exercise real indexing, not a
    // degenerate 1x1 case.
    const int in_c = 2, in_h = 5, in_w = 4, out_c = 3, kernel = 3, stride = 1, pad = 1;
    std::vector<float> input(static_cast<std::size_t>(in_c) * in_h * in_w);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = std::sin(0.37f * static_cast<float>(i)) * 2.0f;
    std::vector<float> weights(static_cast<std::size_t>(out_c) * in_c * kernel * kernel);
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = std::cos(0.19f * static_cast<float>(i)) * 0.5f;

    const auto expected = reference_convolution(input, weights, in_c, in_h, in_w, out_c, kernel, stride, pad);

    float *device_input = nullptr, *device_weights = nullptr, *device_output = nullptr;
    check(cudaMalloc(&device_input, input.size() * sizeof(float)), "cudaMalloc input");
    check(cudaMalloc(&device_weights, weights.size() * sizeof(float)), "cudaMalloc weights");
    check(cudaMalloc(&device_output, expected.size() * sizeof(float)), "cudaMalloc output");
    check(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D input");
    check(cudaMemcpy(device_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice),
          "H2D weights");

    detector::cudnn_convolution_forward(device_input, device_weights, device_output, in_c, in_h, in_w, out_c,
                                         kernel, stride, pad);

    std::vector<float> actual(expected.size());
    check(cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "D2H output");
    float max_abs_diff = 0.0f;
    for (std::size_t i = 0; i < expected.size(); ++i) max_abs_diff = std::max(max_abs_diff, std::fabs(expected[i] - actual[i]));
    std::printf("cudnn_smoke: conv forward max abs diff vs reference = %g\n", max_abs_diff);
    if (max_abs_diff >= 1e-3f) throw std::runtime_error("cuDNN forward parity failed");

    std::vector<float> dy(expected.size()), dw(weights.size(), 0.125f), dx(input.size(), -0.25f);
    std::vector<float> expected_dw = dw, expected_dx = dx;
    for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = std::sin(0.23f * static_cast<float>(i));
    const int out_h = (in_h + 2 * pad - kernel) / stride + 1;
    const int out_w = (in_w + 2 * pad - kernel) / stride + 1;
    for (int oc = 0; oc < out_c; ++oc)
        for (int y = 0; y < out_h; ++y)
            for (int x = 0; x < out_w; ++x)
                for (int ic = 0; ic < in_c; ++ic)
                    for (int ky = 0; ky < kernel; ++ky)
                        for (int kx = 0; kx < kernel; ++kx) {
                            int iy = y * stride + ky - pad, ix = x * stride + kx - pad;
                            if (iy < 0 || ix < 0 || iy >= in_h || ix >= in_w) continue;
                            std::size_t wi = ((static_cast<std::size_t>(oc) * in_c + ic) * kernel + ky) * kernel + kx;
                            std::size_t ii = (static_cast<std::size_t>(ic) * in_h + iy) * in_w + ix;
                            std::size_t oi = (static_cast<std::size_t>(oc) * out_h + y) * out_w + x;
                            expected_dw[wi] += input[ii] * dy[oi];
                            expected_dx[ii] += weights[wi] * dy[oi];
                        }
    float *device_dy = nullptr, *device_dw = nullptr, *device_dx = nullptr;
    check(cudaMalloc(&device_dy, dy.size() * sizeof(float)), "cudaMalloc dy");
    check(cudaMalloc(&device_dw, dw.size() * sizeof(float)), "cudaMalloc dw");
    check(cudaMalloc(&device_dx, dx.size() * sizeof(float)), "cudaMalloc dx");
    check(cudaMemcpy(device_dy, dy.data(), dy.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D dy");
    check(cudaMemcpy(device_dw, dw.data(), dw.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D dw");
    check(cudaMemcpy(device_dx, dx.data(), dx.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D dx");
    detector::CudnnConvolution convolution(in_c, in_h, in_w, out_c, kernel, stride, pad);
    convolution.backward_filter(device_input, device_dy, device_dw);
    convolution.backward_data(device_weights, device_dy, device_dx);
    check(cudaMemcpy(dw.data(), device_dw, dw.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H dw");
    check(cudaMemcpy(dx.data(), device_dx, dx.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H dx");
    float filter_diff = 0.0f, input_diff = 0.0f;
    for (std::size_t i = 0; i < dw.size(); ++i) filter_diff = std::max(filter_diff, std::fabs(dw[i] - expected_dw[i]));
    for (std::size_t i = 0; i < dx.size(); ++i) input_diff = std::max(input_diff, std::fabs(dx[i] - expected_dx[i]));
    std::printf("cudnn_smoke: backward filter/data max abs diff = %g/%g\n", filter_diff, input_diff);
    if (filter_diff >= 1e-3f || input_diff >= 1e-3f)
        throw std::runtime_error("cuDNN backward parity failed");
    cudaFree(device_dy); cudaFree(device_dw); cudaFree(device_dx);
    cudaFree(device_input); cudaFree(device_weights); cudaFree(device_output);

    return 0;
}
