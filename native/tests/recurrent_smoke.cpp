// Gradient-check and loss-decrease regression tests for detector::RecurrentCell (RNN/LSTM/GRU
// with BPTT). No cfg/Network integration or toy end-to-end sequence demo is attempted here -
// deliberately out of scope per the user's own decision; this is a correctness check on the
// cell math and backward pass only, following this project's established verification pattern
// (e.g. training_smoke.cpp's bn_grad_check) for every other from-scratch layer.
#include "detector/recurrent.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using detector::RecurrentCell;
using detector::RecurrentKind;

namespace {

// Loss L = sum over all timesteps and hidden units of h_t[k]^2; dL/dh_t[k] = 2*h_t[k].
float sequence_loss(RecurrentCell& cell, const std::vector<std::vector<float>>& inputs) {
    cell.reset_sequence();
    float loss = 0.0f;
    for (const auto& x : inputs) {
        const auto& h = cell.step(x);
        for (float v : h) loss += v * v;
    }
    return loss;
}

void gradient_check(RecurrentKind kind, const char* name) {
    const int input_size = 3, hidden_size = 2, timesteps = 3;
    RecurrentCell cell(kind, input_size, hidden_size, /*seed=*/42);

    std::vector<std::vector<float>> inputs(timesteps, std::vector<float>(input_size));
    unsigned state = 7;
    for (auto& x : inputs)
        for (float& v : x) {
            state = state * 1103515245u + 12345u;
            v = (static_cast<float>(state >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
        }

    cell.zero_gradients();
    cell.reset_sequence();
    std::vector<std::vector<float>> hs;
    for (const auto& x : inputs) hs.push_back(cell.step(x));

    std::vector<std::vector<float>> dh_targets(timesteps, std::vector<float>(hidden_size));
    for (int t = 0; t < timesteps; ++t)
        for (int k = 0; k < hidden_size; ++k) dh_targets[static_cast<std::size_t>(t)][k] = 2.0f * hs[static_cast<std::size_t>(t)][k];
    cell.backward_through_time(dh_targets);

    auto params = cell.parameters();
    auto grads = cell.gradients();
    assert(params.size() == grads.size());

    const float eps = 1e-3f;
    float max_relative_error = 0.0f;
    // Sampling every few parameters keeps this test fast while still covering every gate/weight
    // matrix (wx/wh/b) for every kind.
    for (std::size_t i = 0; i < params.size(); i += 5) {
        float original = *params[i];
        *params[i] = original + eps;
        float loss_plus = sequence_loss(cell, inputs);
        *params[i] = original - eps;
        float loss_minus = sequence_loss(cell, inputs);
        *params[i] = original;

        float numeric = (loss_plus - loss_minus) / (2.0f * eps);
        float analytic = *grads[i];
        float denominator = std::max(1e-3f, std::fabs(numeric) + std::fabs(analytic));
        float relative_error = std::fabs(numeric - analytic) / denominator;
        max_relative_error = std::max(max_relative_error, relative_error);
    }
    std::printf("recurrent_smoke: %s gradient check max relative error = %g\n", name, max_relative_error);
    assert(max_relative_error < 0.02f);
}

void loss_decreases(RecurrentKind kind, const char* name) {
    const int input_size = 4, hidden_size = 3, timesteps = 4;
    RecurrentCell cell(kind, input_size, hidden_size, /*seed=*/99);
    std::vector<std::vector<float>> inputs(timesteps, std::vector<float>(input_size));
    unsigned state = 123;
    for (auto& x : inputs)
        for (float& v : x) {
            state = state * 1103515245u + 12345u;
            v = (static_cast<float>(state >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
        }

    float first_loss = 0.0f, last_loss = 0.0f;
    for (int step = 0; step < 200; ++step) {
        cell.zero_gradients();
        cell.reset_sequence();
        std::vector<std::vector<float>> hs;
        for (const auto& x : inputs) hs.push_back(cell.step(x));
        float loss = 0.0f;
        std::vector<std::vector<float>> dh_targets(timesteps, std::vector<float>(hidden_size));
        for (int t = 0; t < timesteps; ++t)
            for (int k = 0; k < hidden_size; ++k) {
                float target = (t == timesteps - 1) ? 1.0f : 0.0f;
                float diff = hs[static_cast<std::size_t>(t)][k] - target;
                loss += diff * diff;
                dh_targets[static_cast<std::size_t>(t)][k] = 2.0f * diff;
            }
        cell.backward_through_time(dh_targets);
        cell.sgd_step(0.05f);
        if (step == 0) first_loss = loss;
        last_loss = loss;
    }
    std::printf("recurrent_smoke: %s loss %f -> %f\n", name, first_loss, last_loss);
    assert(last_loss < first_loss);
}

}  // namespace

int main() {
    gradient_check(RecurrentKind::rnn, "rnn");
    gradient_check(RecurrentKind::lstm, "lstm");
    gradient_check(RecurrentKind::gru, "gru");
    loss_decreases(RecurrentKind::rnn, "rnn");
    loss_decreases(RecurrentKind::lstm, "lstm");
    loss_decreases(RecurrentKind::gru, "gru");
    return 0;
}
