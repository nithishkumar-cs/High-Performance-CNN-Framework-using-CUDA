#pragma once

#include <cstddef>
#include <vector>

// Standalone RNN/LSTM/GRU cell with backprop-through-time (BPTT), independently derived from
// the standard recurrence equations (not ported from Darknet's rnn_layer.c/lstm_layer.c/
// gru_layer.c - this project targets CNN detector/classifier graphs; Darknet itself only
// exercises these via its separate rnn.c char-sequence tool, so there is no existing cfg/graph
// concept of a time dimension to integrate into here). Not wired into Network/Trainer's
// single-pass-per-sample cfg graph - that integration (a cfg layer kind, multi-timestep input
// framing) is a separate, larger follow-on item; this is the cell math plus CPU training,
// gradient-checked for correctness, per the user's explicit scope decision to skip an
// end-to-end toy-sequence demo for now.

namespace detector {

enum class RecurrentKind { rnn, lstm, gru };

class RecurrentCell {
public:
    /** Initializes one RNN, LSTM, or GRU cell with positive input and hidden sizes. */
    RecurrentCell(RecurrentKind kind, int input_size, int hidden_size, unsigned seed);

    RecurrentKind kind() const { return kind_; }
    int input_size() const { return input_size_; }
    int hidden_size() const { return hidden_size_; }

    // Clears hidden/cell state and the per-sequence timestep cache - call before feeding a new,
    // independent sequence (state does not carry across sequences, only within one).
    void reset_sequence();

    // Feeds one timestep's input and returns the new hidden state (hidden_size), advancing and
    // caching state for a later backward_through_time() call. `input` must have input_size()
    // elements.
    const std::vector<float>& step(const std::vector<float>& input);

    // Backpropagates one full cached sequence's hidden-state-output gradients through time
    // (reverse order internally). `dh_targets[t]` is dLoss/dh_t injected directly at timestep t
    // (e.g. from a loss computed on that timestep's output); pass an all-zero vector for
    // timesteps with no direct loss contribution - the recurrent dependency still carries
    // gradient backward through earlier timesteps regardless. Must be called with exactly as
    // many entries as step() was called since the last reset_sequence(). Adds to (does not
    // reset) the internal gradient accumulators - call zero_gradients() first for a fresh
    // accumulation. Returns dLoss/dInput for every timestep, same order as the step() calls.
    std::vector<std::vector<float>> backward_through_time(const std::vector<std::vector<float>>& dh_targets);

    void zero_gradients();
    // Plain SGD (no momentum/decay - this class is a narrower-scoped cell than Trainer/
    // CudaTrainer) using the currently accumulated gradients, then zeros them.
    void sgd_step(float lr);

    // Flattened, order-matched access to every learnable parameter and its currently
    // accumulated gradient, for finite-difference gradient-check tests.
    std::vector<float*> parameters();
    std::vector<float*> gradients();
    /** Returns learnable parameters in the same flattened order as parameters(). */
    std::vector<float> parameter_values() const;

private:
    struct Gate {
        std::vector<float> wx;  // hidden_size x input_size
        std::vector<float> wh;  // hidden_size x hidden_size
        std::vector<float> b;   // hidden_size
        std::vector<float> dwx, dwh, db;
    };

    struct Timestep {
        std::vector<float> input;
        std::vector<float> h_prev, c_prev;
        std::vector<float> h, c;
        // Gate pre-activation outputs (post nonlinearity), kind-dependent count/meaning:
        // rnn: {z (tanh output, == h)}; lstm: {i, f, o, g}; gru: {z, r, n}.
        std::vector<std::vector<float>> gates;
    };

    RecurrentKind kind_;
    int input_size_;
    int hidden_size_;
    std::vector<Gate> gates_;  // size depends on kind_: rnn=1, gru=3, lstm=4
    std::vector<float> h_, c_;  // current state, persists across step() calls within a sequence
    std::vector<Timestep> cache_;

    Gate make_gate(unsigned& seed_state);
    std::vector<float> matvec(const std::vector<float>& matrix, int rows, int cols, const std::vector<float>& vector) const;
};

}  // namespace detector
