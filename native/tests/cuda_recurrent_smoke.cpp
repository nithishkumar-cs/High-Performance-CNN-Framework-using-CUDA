#include "detector/cuda_recurrent.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

/** Asserts close CPU and GPU recurrence values with a scale aware tolerance. */
void close(float expected, float actual) {
    float tolerance = 2e-4f * std::max({1.0f, std::fabs(expected), std::fabs(actual)});
    if (std::fabs(expected - actual) > tolerance) {
        std::printf("recurrent parity mismatch: CPU=%g CUDA=%g\n", expected, actual);
        std::abort();
    }
}

/** Compares every timestep output, input gradient, and learnable parameter gradient. */
void check_kind(detector::RecurrentKind kind) {
    detector::RecurrentCell cpu(kind, 3, 4, 41);
    std::vector<std::vector<float>> inputs{{0.3f, -0.8f, 0.2f},
                                           {-0.4f, 0.5f, 0.9f},
                                           {0.7f, 0.1f, -0.6f}};
    std::vector<std::vector<float>> direct{{0.1f, -0.3f, 0.2f, 0.4f},
                                           {-0.2f, 0.5f, -0.1f, 0.3f},
                                           {0.8f, -0.6f, 0.7f, -0.4f}};
    cpu.reset_sequence();
    cpu.zero_gradients();
    std::vector<std::vector<float>> outputs;
    for (const auto& x : inputs) outputs.push_back(cpu.step(x));
    auto input_gradients = cpu.backward_through_time(direct);
    auto gpu = detector::recurrent_forward_backward_cuda(cpu, inputs, direct);
    if (gpu.outputs.size() != inputs.size()) std::abort();
    for (std::size_t t = 0; t < inputs.size(); ++t) {
        for (std::size_t i = 0; i < outputs[t].size(); ++i) close(outputs[t][i], gpu.outputs[t][i]);
        for (std::size_t i = 0; i < inputs[t].size(); ++i) close(input_gradients[t][i], gpu.input_gradients[t][i]);
    }
    auto cpu_gradients = cpu.gradients();
    if (cpu_gradients.size() != gpu.parameter_gradients.size()) std::abort();
    for (std::size_t i = 0; i < cpu_gradients.size(); ++i)
        close(*cpu_gradients[i], gpu.parameter_gradients[i]);
    auto second = detector::recurrent_forward_backward_cuda(cpu, inputs, direct);
    for (std::size_t i = 0; i < second.parameter_gradients.size(); ++i)
        close(gpu.parameter_gradients[i], second.parameter_gradients[i]);
    bool rejected = false;
    try { detector::recurrent_forward_backward_cuda(cpu, inputs, {}); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) std::abort();
}

}  // namespace

/** Runs synthetic forward and BPTT parity for all three recurrent architectures. */
int main() {
    check_kind(detector::RecurrentKind::rnn);
    check_kind(detector::RecurrentKind::lstm);
    check_kind(detector::RecurrentKind::gru);
    std::puts("cuda_recurrent_smoke: RNN/LSTM/GRU forward and BPTT parity passed");
}
