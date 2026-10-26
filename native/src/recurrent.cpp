#include "detector/recurrent.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace detector {
namespace {

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Deterministic, dependency-free small-weight uniform init (same family as this project's
// other from-scratch layers) - not Darknet's RNG, just enough spread to break symmetry.
float next_random(unsigned& state) {
    state = state * 1664525u + 1013904223u;
    return (static_cast<float>(state >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
}

int gate_count(RecurrentKind kind) {
    switch (kind) {
        case RecurrentKind::rnn: return 1;
        case RecurrentKind::lstm: return 4;
        case RecurrentKind::gru: return 3;
    }
    throw std::runtime_error("unreachable RecurrentKind");
}

// outer_accumulate(dst, delta, rows, vec, cols): dst[r*cols+c] += delta[r] * vec[c].
void outer_accumulate(std::vector<float>& dst, const std::vector<float>& delta, int rows,
                       const std::vector<float>& vec, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) dst[static_cast<std::size_t>(r) * cols + c] += delta[r] * vec[c];
}

// transpose_matvec_accumulate(dst, matrix, rows, cols, delta): dst[c] += sum_r matrix[r*cols+c]*delta[r].
void transpose_matvec_accumulate(std::vector<float>& dst, const std::vector<float>& matrix, int rows, int cols,
                                  const std::vector<float>& delta) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) dst[c] += matrix[static_cast<std::size_t>(r) * cols + c] * delta[r];
}

}  // namespace

RecurrentCell::Gate RecurrentCell::make_gate(unsigned& seed_state) {
    Gate gate;
    gate.wx.resize(static_cast<std::size_t>(hidden_size_) * input_size_);
    gate.wh.resize(static_cast<std::size_t>(hidden_size_) * hidden_size_);
    gate.b.assign(hidden_size_, 0.0f);
    const float scale = 0.1f;
    for (float& w : gate.wx) w = next_random(seed_state) * scale;
    for (float& w : gate.wh) w = next_random(seed_state) * scale;
    gate.dwx.assign(gate.wx.size(), 0.0f);
    gate.dwh.assign(gate.wh.size(), 0.0f);
    gate.db.assign(gate.b.size(), 0.0f);
    return gate;
}

std::vector<float> RecurrentCell::matvec(const std::vector<float>& matrix, int rows, int cols,
                                          const std::vector<float>& vector) const {
    std::vector<float> result(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) sum += matrix[static_cast<std::size_t>(r) * cols + c] * vector[c];
        result[r] = sum;
    }
    return result;
}

RecurrentCell::RecurrentCell(RecurrentKind kind, int input_size, int hidden_size, unsigned seed)
    : kind_(kind), input_size_(input_size), hidden_size_(hidden_size) {
    unsigned state = seed ? seed : 1u;
    gates_.resize(gate_count(kind));
    for (Gate& gate : gates_) gate = make_gate(state);
    reset_sequence();
}

void RecurrentCell::reset_sequence() {
    h_.assign(hidden_size_, 0.0f);
    c_.assign(hidden_size_, 0.0f);
    cache_.clear();
}

const std::vector<float>& RecurrentCell::step(const std::vector<float>& input) {
    if (static_cast<int>(input.size()) != input_size_) throw std::runtime_error("RecurrentCell::step: input size mismatch");
    Timestep ts;
    ts.input = input;
    ts.h_prev = h_;
    ts.c_prev = c_;

    if (kind_ == RecurrentKind::rnn) {
        std::vector<float> z = matvec(gates_[0].wx, hidden_size_, input_size_, input);
        std::vector<float> zh = matvec(gates_[0].wh, hidden_size_, hidden_size_, h_);
        std::vector<float> h(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i) h[i] = std::tanh(z[i] + zh[i] + gates_[0].b[i]);
        ts.gates = {h};
        ts.h = h;
        h_ = h;
    } else if (kind_ == RecurrentKind::lstm) {
        std::vector<std::vector<float>> pre(4);
        for (int g = 0; g < 4; ++g) {
            std::vector<float> zx = matvec(gates_[g].wx, hidden_size_, input_size_, input);
            std::vector<float> zh = matvec(gates_[g].wh, hidden_size_, hidden_size_, h_);
            pre[g].resize(hidden_size_);
            for (int i = 0; i < hidden_size_; ++i) pre[g][i] = zx[i] + zh[i] + gates_[g].b[i];
        }
        std::vector<float> i_gate(hidden_size_), f_gate(hidden_size_), o_gate(hidden_size_), g_gate(hidden_size_);
        for (int k = 0; k < hidden_size_; ++k) {
            i_gate[k] = sigmoid(pre[0][k]);
            f_gate[k] = sigmoid(pre[1][k]);
            o_gate[k] = sigmoid(pre[2][k]);
            g_gate[k] = std::tanh(pre[3][k]);
        }
        std::vector<float> c(hidden_size_), h(hidden_size_);
        for (int k = 0; k < hidden_size_; ++k) {
            c[k] = f_gate[k] * c_[k] + i_gate[k] * g_gate[k];
            h[k] = o_gate[k] * std::tanh(c[k]);
        }
        ts.gates = {i_gate, f_gate, o_gate, g_gate};
        ts.c = c;
        ts.h = h;
        c_ = c;
        h_ = h;
    } else {  // gru
        std::vector<float> zx_z = matvec(gates_[0].wx, hidden_size_, input_size_, input);
        std::vector<float> zh_z = matvec(gates_[0].wh, hidden_size_, hidden_size_, h_);
        std::vector<float> zx_r = matvec(gates_[1].wx, hidden_size_, input_size_, input);
        std::vector<float> zh_r = matvec(gates_[1].wh, hidden_size_, hidden_size_, h_);
        std::vector<float> z_gate(hidden_size_), r_gate(hidden_size_);
        for (int k = 0; k < hidden_size_; ++k) {
            z_gate[k] = sigmoid(zx_z[k] + zh_z[k] + gates_[0].b[k]);
            r_gate[k] = sigmoid(zx_r[k] + zh_r[k] + gates_[1].b[k]);
        }
        std::vector<float> r_h(hidden_size_);
        for (int k = 0; k < hidden_size_; ++k) r_h[k] = r_gate[k] * h_[k];
        std::vector<float> zx_n = matvec(gates_[2].wx, hidden_size_, input_size_, input);
        std::vector<float> zh_n = matvec(gates_[2].wh, hidden_size_, hidden_size_, r_h);
        std::vector<float> n_gate(hidden_size_);
        for (int k = 0; k < hidden_size_; ++k) n_gate[k] = std::tanh(zx_n[k] + zh_n[k] + gates_[2].b[k]);
        std::vector<float> h(hidden_size_);
        for (int k = 0; k < hidden_size_; ++k) h[k] = (1.0f - z_gate[k]) * n_gate[k] + z_gate[k] * h_[k];
        ts.gates = {z_gate, r_gate, n_gate};
        ts.h = h;
        h_ = h;
    }
    cache_.push_back(std::move(ts));
    return h_;
}

std::vector<std::vector<float>> RecurrentCell::backward_through_time(
    const std::vector<std::vector<float>>& dh_targets) {
    if (dh_targets.size() != cache_.size())
        throw std::runtime_error("RecurrentCell::backward_through_time: dh_targets size mismatch");
    const int T = static_cast<int>(cache_.size());
    std::vector<std::vector<float>> dinputs(T, std::vector<float>(input_size_, 0.0f));
    std::vector<float> dh_carry(hidden_size_, 0.0f);
    std::vector<float> dc_carry(hidden_size_, 0.0f);

    for (int t = T - 1; t >= 0; --t) {
        const Timestep& ts = cache_[static_cast<std::size_t>(t)];
        std::vector<float> dh(hidden_size_);
        for (int k = 0; k < hidden_size_; ++k) dh[k] = dh_targets[static_cast<std::size_t>(t)][k] + dh_carry[k];
        std::fill(dh_carry.begin(), dh_carry.end(), 0.0f);

        if (kind_ == RecurrentKind::rnn) {
            std::vector<float> dz(hidden_size_);
            for (int k = 0; k < hidden_size_; ++k) dz[k] = dh[k] * (1.0f - ts.h[k] * ts.h[k]);
            outer_accumulate(gates_[0].dwx, dz, hidden_size_, ts.input, input_size_);
            outer_accumulate(gates_[0].dwh, dz, hidden_size_, ts.h_prev, hidden_size_);
            for (int k = 0; k < hidden_size_; ++k) gates_[0].db[k] += dz[k];
            transpose_matvec_accumulate(dinputs[static_cast<std::size_t>(t)], gates_[0].wx, hidden_size_, input_size_, dz);
            transpose_matvec_accumulate(dh_carry, gates_[0].wh, hidden_size_, hidden_size_, dz);
        } else if (kind_ == RecurrentKind::lstm) {
            const auto& i_gate = ts.gates[0];
            const auto& f_gate = ts.gates[1];
            const auto& o_gate = ts.gates[2];
            const auto& g_gate = ts.gates[3];
            std::vector<float> dc(hidden_size_);
            std::vector<float> di(hidden_size_), df(hidden_size_), do_(hidden_size_), dg(hidden_size_);
            for (int k = 0; k < hidden_size_; ++k) {
                float tanh_c = std::tanh(ts.c[k]);
                do_[k] = dh[k] * tanh_c * o_gate[k] * (1.0f - o_gate[k]);
                dc[k] = dc_carry[k] + dh[k] * o_gate[k] * (1.0f - tanh_c * tanh_c);
                df[k] = dc[k] * ts.c_prev[k] * f_gate[k] * (1.0f - f_gate[k]);
                di[k] = dc[k] * g_gate[k] * i_gate[k] * (1.0f - i_gate[k]);
                dg[k] = dc[k] * i_gate[k] * (1.0f - g_gate[k] * g_gate[k]);
                dc_carry[k] = dc[k] * f_gate[k];
            }
            const std::vector<float>* pre[4] = {&di, &df, &do_, &dg};
            for (int g = 0; g < 4; ++g) {
                outer_accumulate(gates_[g].dwx, *pre[g], hidden_size_, ts.input, input_size_);
                outer_accumulate(gates_[g].dwh, *pre[g], hidden_size_, ts.h_prev, hidden_size_);
                for (int k = 0; k < hidden_size_; ++k) gates_[g].db[k] += (*pre[g])[k];
                transpose_matvec_accumulate(dinputs[static_cast<std::size_t>(t)], gates_[g].wx, hidden_size_, input_size_, *pre[g]);
                transpose_matvec_accumulate(dh_carry, gates_[g].wh, hidden_size_, hidden_size_, *pre[g]);
            }
        } else {  // gru
            const auto& z_gate = ts.gates[0];
            const auto& r_gate = ts.gates[1];
            const auto& n_gate = ts.gates[2];
            std::vector<float> dz(hidden_size_), dn(hidden_size_);
            for (int k = 0; k < hidden_size_; ++k) {
                dz[k] = dh[k] * (ts.h_prev[k] - n_gate[k]) * z_gate[k] * (1.0f - z_gate[k]);
                dn[k] = dh[k] * (1.0f - z_gate[k]) * (1.0f - n_gate[k] * n_gate[k]);
            }
            std::vector<float> d_rh(hidden_size_, 0.0f);
            transpose_matvec_accumulate(d_rh, gates_[2].wh, hidden_size_, hidden_size_, dn);
            std::vector<float> dr(hidden_size_);
            for (int k = 0; k < hidden_size_; ++k) dr[k] = d_rh[k] * ts.h_prev[k] * r_gate[k] * (1.0f - r_gate[k]);

            std::vector<float> r_h_prev(hidden_size_);
            for (int k = 0; k < hidden_size_; ++k) r_h_prev[k] = r_gate[k] * ts.h_prev[k];

            outer_accumulate(gates_[0].dwx, dz, hidden_size_, ts.input, input_size_);
            outer_accumulate(gates_[0].dwh, dz, hidden_size_, ts.h_prev, hidden_size_);
            outer_accumulate(gates_[1].dwx, dr, hidden_size_, ts.input, input_size_);
            outer_accumulate(gates_[1].dwh, dr, hidden_size_, ts.h_prev, hidden_size_);
            outer_accumulate(gates_[2].dwx, dn, hidden_size_, ts.input, input_size_);
            outer_accumulate(gates_[2].dwh, dn, hidden_size_, r_h_prev, hidden_size_);
            for (int k = 0; k < hidden_size_; ++k) {
                gates_[0].db[k] += dz[k];
                gates_[1].db[k] += dr[k];
                gates_[2].db[k] += dn[k];
            }
            transpose_matvec_accumulate(dinputs[static_cast<std::size_t>(t)], gates_[0].wx, hidden_size_, input_size_, dz);
            transpose_matvec_accumulate(dinputs[static_cast<std::size_t>(t)], gates_[1].wx, hidden_size_, input_size_, dr);
            transpose_matvec_accumulate(dinputs[static_cast<std::size_t>(t)], gates_[2].wx, hidden_size_, input_size_, dn);

            for (int k = 0; k < hidden_size_; ++k) dh_carry[k] += dh[k] * z_gate[k] + d_rh[k] * r_gate[k];
            transpose_matvec_accumulate(dh_carry, gates_[0].wh, hidden_size_, hidden_size_, dz);
            transpose_matvec_accumulate(dh_carry, gates_[1].wh, hidden_size_, hidden_size_, dr);
        }
    }
    return dinputs;
}

void RecurrentCell::zero_gradients() {
    for (Gate& gate : gates_) {
        std::fill(gate.dwx.begin(), gate.dwx.end(), 0.0f);
        std::fill(gate.dwh.begin(), gate.dwh.end(), 0.0f);
        std::fill(gate.db.begin(), gate.db.end(), 0.0f);
    }
}

void RecurrentCell::sgd_step(float lr) {
    for (Gate& gate : gates_) {
        for (std::size_t i = 0; i < gate.wx.size(); ++i) gate.wx[i] -= lr * gate.dwx[i];
        for (std::size_t i = 0; i < gate.wh.size(); ++i) gate.wh[i] -= lr * gate.dwh[i];
        for (std::size_t i = 0; i < gate.b.size(); ++i) gate.b[i] -= lr * gate.db[i];
    }
    zero_gradients();
}

std::vector<float*> RecurrentCell::parameters() {
    std::vector<float*> result;
    for (Gate& gate : gates_) {
        for (float& w : gate.wx) result.push_back(&w);
        for (float& w : gate.wh) result.push_back(&w);
        for (float& w : gate.b) result.push_back(&w);
    }
    return result;
}

std::vector<float*> RecurrentCell::gradients() {
    std::vector<float*> result;
    for (Gate& gate : gates_) {
        for (float& w : gate.dwx) result.push_back(&w);
        for (float& w : gate.dwh) result.push_back(&w);
        for (float& w : gate.db) result.push_back(&w);
    }
    return result;
}

}  // namespace detector
