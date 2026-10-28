#include "detector/classifier_trainer.hpp"
#include "detector/l2norm.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace detector {

/** Validates a sequential classifier graph and its SGD hyperparameters. */
ClassificationTrainer::ClassificationTrainer(Network& network, float learning_rate, float weight_decay)
    : network_(network), learning_rate_(learning_rate), weight_decay_(weight_decay) {
    if (!std::isfinite(learning_rate) || learning_rate <= 0 ||
        !std::isfinite(weight_decay) || weight_decay < 0)
        throw std::runtime_error("invalid classification SGD parameters");
    const auto& layers = network_.layers_;
    if (layers.empty() || layers.back().kind != Kind::softmax)
        throw std::runtime_error("classifier requires final [softmax]");
    int recurrent_count = 0;
    for (const auto& layer : layers) {
        if (layer.kind == Kind::rnn || layer.kind == Kind::lstm || layer.kind == Kind::gru) { ++recurrent_count; continue; }
        if (layer.kind != Kind::convolutional && layer.kind != Kind::maxpool &&
            layer.kind != Kind::connected && layer.kind != Kind::softmax && layer.kind != Kind::l2norm)
            throw std::runtime_error("classifier trainer supports convolutional, maxpool, connected, softmax, "
                                      "l2norm, and (via train_sequence()) one rnn/lstm/gru layer");
    }
    if (recurrent_count > 1) throw std::runtime_error("classifier trainer supports at most one rnn/lstm/gru layer");
}

/** Applies one sample's cross-entropy SGD update to a nonrecurrent classifier. */
ClassificationResult ClassificationTrainer::step(const std::vector<float>& input, int class_id) {
    if (input.size() != network_.input_.elements())
        throw std::runtime_error("classification input shape mismatch");
    auto& layers = network_.layers_;
    for (const auto& layer : layers)
        if (layer.kind == Kind::rnn || layer.kind == Kind::lstm || layer.kind == Kind::gru)
            throw std::runtime_error("cfg has a recurrent layer; use train_sequence() instead of step()");
    const int classes = layers.back().output.c;
    if (class_id < 0 || class_id >= classes)
        throw std::runtime_error("classification target outside class range");

    std::vector<std::vector<float>> output(layers.size());
    std::vector<std::vector<float>> preactivation(layers.size());
    std::vector<std::vector<std::size_t>> max_index(layers.size());
    std::vector<std::vector<float>> gradient(layers.size());
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto& layer = layers[index];
        const auto& previous = index ? output[index - 1] : input;
        auto& result = output[index];
        result.resize(layer.output.elements());
        gradient[index].assign(result.size(), 0.0f);

        if (layer.kind == Kind::convolutional) {
            const std::size_t fan_in = static_cast<std::size_t>(layer.input.c) * layer.size * layer.size;
            if (layer.biases.size() != static_cast<std::size_t>(layer.filters) ||
                layer.weights.size() != static_cast<std::size_t>(layer.filters) * fan_in)
                throw std::runtime_error("initialize or load classifier weights before training");
            preactivation[index].resize(result.size());
            const int in_plane = layer.input.h * layer.input.w;
            const int out_plane = layer.output.h * layer.output.w;
            for (int oc = 0; oc < layer.output.c; ++oc) {
                float scale = layer.batch_normalize ?
                    layer.scales[oc] / std::sqrt(layer.variances[oc] + 1e-5f) : 1.0f;
                float offset = layer.batch_normalize ?
                    layer.biases[oc] - layer.means[oc] * scale : layer.biases[oc];
                for (int oy = 0; oy < layer.output.h; ++oy)
                    for (int ox = 0; ox < layer.output.w; ++ox) {
                        float sum = 0.0f;
                        for (int ic = 0; ic < layer.input.c; ++ic)
                            for (int ky = 0; ky < layer.size; ++ky)
                                for (int kx = 0; kx < layer.size; ++kx) {
                                    int iy = oy * layer.stride + ky - layer.padding;
                                    int ix = ox * layer.stride + kx - layer.padding;
                                    if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                                    std::size_t wi = ((static_cast<std::size_t>(oc) * layer.input.c + ic) *
                                        layer.size + ky) * layer.size + kx;
                                    sum += previous[static_cast<std::size_t>(ic) * in_plane +
                                        iy * layer.input.w + ix] * layer.weights[wi];
                                }
                        auto pos = static_cast<std::size_t>(oc) * out_plane + oy * layer.output.w + ox;
                        float value = sum * scale + offset;
                        preactivation[index][pos] = value;
                        result[pos] = layer.leaky && value < 0 ? value * 0.1f : value;
                    }
            }
        } else if (layer.kind == Kind::maxpool) {
            max_index[index].resize(result.size());
            const int pad = (layer.size - 1) / 2;
            const int in_plane = layer.input.h * layer.input.w;
            const int out_plane = layer.output.h * layer.output.w;
            for (int c = 0; c < layer.input.c; ++c)
                for (int oy = 0; oy < layer.output.h; ++oy)
                    for (int ox = 0; ox < layer.output.w; ++ox) {
                        float best = -std::numeric_limits<float>::infinity();
                        std::size_t best_pos = 0;
                        for (int ky = 0; ky < layer.size; ++ky)
                            for (int kx = 0; kx < layer.size; ++kx) {
                                int iy = oy * layer.stride + ky - pad;
                                int ix = ox * layer.stride + kx - pad;
                                if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                                std::size_t pos = static_cast<std::size_t>(c) * in_plane +
                                    iy * layer.input.w + ix;
                                if (previous[pos] > best) { best = previous[pos]; best_pos = pos; }
                            }
                        std::size_t pos = static_cast<std::size_t>(c) * out_plane +
                            oy * layer.output.w + ox;
                        result[pos] = best;
                        max_index[index][pos] = best_pos;
                    }
        } else if (layer.kind == Kind::connected) {
            if (layer.biases.size() != static_cast<std::size_t>(layer.filters) ||
                layer.weights.size() != static_cast<std::size_t>(layer.filters) * previous.size())
                throw std::runtime_error("initialize or load classifier weights before training");
            preactivation[index].resize(result.size());
            for (int oc = 0; oc < layer.filters; ++oc) {
                float value = layer.biases[static_cast<std::size_t>(oc)];
                for (std::size_t ic = 0; ic < previous.size(); ++ic)
                    value += previous[ic] * layer.weights[static_cast<std::size_t>(oc) *
                        previous.size() + ic];
                preactivation[index][static_cast<std::size_t>(oc)] = value;
                result[static_cast<std::size_t>(oc)] =
                    layer.leaky && value < 0 ? value * 0.1f : value;
            }
        } else if (layer.kind == Kind::l2norm) {
            preactivation[index] = {l2norm_forward(previous.data(), result.data(), result.size())};
        } else {
            float maximum = *std::max_element(previous.begin(), previous.end());
            float total = 0.0f;
            for (std::size_t j = 0; j < previous.size(); ++j) {
                result[j] = std::exp(previous[j] - maximum);
                total += result[j];
            }
            for (float& value : result) value /= total;
        }
    }

    const auto& probabilities = output.back();
    float probability = probabilities[static_cast<std::size_t>(class_id)];
    float loss = -std::log(std::max(probability, std::numeric_limits<float>::min()));
    bool correct = static_cast<int>(std::max_element(probabilities.begin(), probabilities.end()) -
                                    probabilities.begin()) == class_id;
    gradient.back() = probabilities;
    gradient.back()[static_cast<std::size_t>(class_id)] -= 1.0f;

    for (std::size_t reverse = layers.size(); reverse-- > 0;) {
        auto& layer = layers[reverse];
        const auto& previous = reverse ? output[reverse - 1] : input;
        auto& grad = gradient[reverse];
        if (layer.kind == Kind::softmax) {
            if (reverse == 0) throw std::runtime_error("softmax requires preceding layer");
            for (std::size_t j = 0; j < grad.size(); ++j)
                gradient[reverse - 1][j] += grad[j];
        } else if (layer.kind == Kind::maxpool) {
            if (reverse)
                for (std::size_t j = 0; j < grad.size(); ++j)
                    gradient[reverse - 1][max_index[reverse][j]] += grad[j];
        } else if (layer.kind == Kind::l2norm) {
            if (reverse)
                l2norm_backward(output[reverse].data(), grad.data(), gradient[reverse - 1].data(),
                                grad.size(), preactivation[reverse][0]);
        } else if (layer.kind == Kind::connected) {
            std::vector<float> weight_grad(layer.weights.size(), 0.0f);
            std::vector<float> bias_grad(layer.biases.size(), 0.0f);
            for (int oc = 0; oc < layer.filters; ++oc) {
                std::size_t o = static_cast<std::size_t>(oc);
                float g = grad[o] * (layer.leaky && preactivation[reverse][o] < 0 ? 0.1f : 1.0f);
                bias_grad[o] += g;
                for (std::size_t ic = 0; ic < previous.size(); ++ic) {
                    std::size_t wi = o * previous.size() + ic;
                    weight_grad[wi] += g * previous[ic];
                    if (reverse) gradient[reverse - 1][ic] += g * layer.weights[wi];
                }
            }
            for (std::size_t j = 0; j < layer.weights.size(); ++j)
                layer.weights[j] -= learning_rate_ * (weight_grad[j] + weight_decay_ * layer.weights[j]);
            for (std::size_t j = 0; j < layer.biases.size(); ++j)
                layer.biases[j] -= learning_rate_ * bias_grad[j];
        } else {
            const int in_plane = layer.input.h * layer.input.w;
            const int out_plane = layer.output.h * layer.output.w;
            std::vector<float> weight_grad(layer.weights.size(), 0.0f);
            std::vector<float> bias_grad(layer.biases.size(), 0.0f);
            for (int oc = 0; oc < layer.output.c; ++oc) {
                float bn_scale = layer.batch_normalize ?
                    layer.scales[oc] / std::sqrt(layer.variances[oc] + 1e-5f) : 1.0f;
                for (int oy = 0; oy < layer.output.h; ++oy)
                    for (int ox = 0; ox < layer.output.w; ++ox) {
                        auto out_pos = static_cast<std::size_t>(oc) * out_plane +
                            oy * layer.output.w + ox;
                        float g = grad[out_pos] *
                            (layer.leaky && preactivation[reverse][out_pos] < 0 ? 0.1f : 1.0f);
                        bias_grad[static_cast<std::size_t>(oc)] += g;
                        float scaled = g * bn_scale;
                        for (int ic = 0; ic < layer.input.c; ++ic)
                            for (int ky = 0; ky < layer.size; ++ky)
                                for (int kx = 0; kx < layer.size; ++kx) {
                                    int iy = oy * layer.stride + ky - layer.padding;
                                    int ix = ox * layer.stride + kx - layer.padding;
                                    if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                                    std::size_t wi = ((static_cast<std::size_t>(oc) * layer.input.c + ic) *
                                        layer.size + ky) * layer.size + kx;
                                    std::size_t source = static_cast<std::size_t>(ic) * in_plane +
                                        iy * layer.input.w + ix;
                                    weight_grad[wi] += scaled * previous[source];
                                    if (reverse) gradient[reverse - 1][source] += scaled * layer.weights[wi];
                                }
                    }
            }
            for (std::size_t j = 0; j < layer.weights.size(); ++j)
                layer.weights[j] -= learning_rate_ * (weight_grad[j] + weight_decay_ * layer.weights[j]);
            for (std::size_t j = 0; j < layer.biases.size(); ++j)
                layer.biases[j] -= learning_rate_ * bias_grad[j];
        }
    }
    if (!std::isfinite(loss)) throw std::runtime_error("non-finite classification loss");
    return {loss, correct};
}

/** Trains one truncated-BPTT chunk while carrying recurrent state into the next chunk. */
std::vector<ClassificationResult> ClassificationTrainer::train_sequence(
        const std::vector<std::vector<float>>& inputs, const std::vector<int>& class_ids) {
    if (inputs.empty() || inputs.size() != class_ids.size())
        throw std::runtime_error("train_sequence requires matching, non-empty inputs/class_ids");
    auto& layers = network_.layers_;
    int recurrent_index = -1;
    for (std::size_t i = 0; i < layers.size(); ++i)
        if (layers[i].kind == Kind::rnn || layers[i].kind == Kind::lstm || layers[i].kind == Kind::gru)
            recurrent_index = static_cast<int>(i);
    if (recurrent_index < 0)
        throw std::runtime_error("train_sequence requires a cfg with one rnn/lstm/gru layer; use step() otherwise");
    const auto r = static_cast<std::size_t>(recurrent_index);
    if (!network_.cells_[r]) throw std::runtime_error("initialize or load weights before training");
    const std::size_t T = inputs.size();
    const int classes = layers.back().output.c;
    for (int c : class_ids) if (c < 0 || c >= classes) throw std::runtime_error("classification target outside class range");
    for (const auto& in : inputs) if (in.size() != network_.input_.elements()) throw std::runtime_error("classification input shape mismatch");

    // Per-timestep, per-layer tensors (mirrors step()'s per-layer tensors, with an extra
    // leading timestep dimension); the recurrent layer's cell.step() calls below run in
    // ascending t order for a fixed layer index, which is what carries its hidden state
    // across timesteps - every other layer kind is independent per timestep.
    std::vector<std::vector<std::vector<float>>> output(T), preactivation(T), gradient(T);
    std::vector<std::vector<std::vector<std::size_t>>> max_index(T);
    for (std::size_t t = 0; t < T; ++t) {
        output[t].resize(layers.size());
        preactivation[t].resize(layers.size());
        gradient[t].resize(layers.size());
        max_index[t].resize(layers.size());
    }

    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto& layer = layers[index];
        for (std::size_t t = 0; t < T; ++t) {
            const auto& previous = index ? output[t][index - 1] : inputs[t];
            auto& result = output[t][index];
            result.resize(layer.output.elements());
            gradient[t][index].assign(result.size(), 0.0f);

            if (layer.kind == Kind::convolutional) {
                if (layer.biases.size() != static_cast<std::size_t>(layer.filters) ||
                    layer.weights.size() != static_cast<std::size_t>(layer.filters) *
                        static_cast<std::size_t>(layer.input.c) * layer.size * layer.size)
                    throw std::runtime_error("initialize or load classifier weights before training");
                preactivation[t][index].resize(result.size());
                const int in_plane = layer.input.h * layer.input.w;
                const int out_plane = layer.output.h * layer.output.w;
                for (int oc = 0; oc < layer.output.c; ++oc) {
                    float scale = layer.batch_normalize ?
                        layer.scales[oc] / std::sqrt(layer.variances[oc] + 1e-5f) : 1.0f;
                    float offset = layer.batch_normalize ?
                        layer.biases[oc] - layer.means[oc] * scale : layer.biases[oc];
                    for (int oy = 0; oy < layer.output.h; ++oy)
                        for (int ox = 0; ox < layer.output.w; ++ox) {
                            float sum = 0.0f;
                            for (int ic = 0; ic < layer.input.c; ++ic)
                                for (int ky = 0; ky < layer.size; ++ky)
                                    for (int kx = 0; kx < layer.size; ++kx) {
                                        int iy = oy * layer.stride + ky - layer.padding;
                                        int ix = ox * layer.stride + kx - layer.padding;
                                        if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                                        std::size_t wi = ((static_cast<std::size_t>(oc) * layer.input.c + ic) *
                                            layer.size + ky) * layer.size + kx;
                                        sum += previous[static_cast<std::size_t>(ic) * in_plane +
                                            iy * layer.input.w + ix] * layer.weights[wi];
                                    }
                            auto pos = static_cast<std::size_t>(oc) * out_plane + oy * layer.output.w + ox;
                            float value = sum * scale + offset;
                            preactivation[t][index][pos] = value;
                            result[pos] = layer.leaky && value < 0 ? value * 0.1f : value;
                        }
                }
            } else if (layer.kind == Kind::maxpool) {
                max_index[t][index].resize(result.size());
                const int pad = (layer.size - 1) / 2;
                const int in_plane = layer.input.h * layer.input.w;
                const int out_plane = layer.output.h * layer.output.w;
                for (int c = 0; c < layer.input.c; ++c)
                    for (int oy = 0; oy < layer.output.h; ++oy)
                        for (int ox = 0; ox < layer.output.w; ++ox) {
                            float best = -std::numeric_limits<float>::infinity();
                            std::size_t best_pos = 0;
                            for (int ky = 0; ky < layer.size; ++ky)
                                for (int kx = 0; kx < layer.size; ++kx) {
                                    int iy = oy * layer.stride + ky - pad;
                                    int ix = ox * layer.stride + kx - pad;
                                    if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                                    std::size_t pos = static_cast<std::size_t>(c) * in_plane +
                                        iy * layer.input.w + ix;
                                    if (previous[pos] > best) { best = previous[pos]; best_pos = pos; }
                                }
                            std::size_t pos = static_cast<std::size_t>(c) * out_plane + oy * layer.output.w + ox;
                            result[pos] = best;
                            max_index[t][index][pos] = best_pos;
                        }
            } else if (layer.kind == Kind::connected) {
                if (layer.biases.size() != static_cast<std::size_t>(layer.filters) ||
                    layer.weights.size() != static_cast<std::size_t>(layer.filters) * previous.size())
                    throw std::runtime_error("initialize or load classifier weights before training");
                preactivation[t][index].resize(result.size());
                for (int oc = 0; oc < layer.filters; ++oc) {
                    float value = layer.biases[static_cast<std::size_t>(oc)];
                    for (std::size_t ic = 0; ic < previous.size(); ++ic)
                        value += previous[ic] * layer.weights[static_cast<std::size_t>(oc) * previous.size() + ic];
                    preactivation[t][index][static_cast<std::size_t>(oc)] = value;
                    result[static_cast<std::size_t>(oc)] = layer.leaky && value < 0 ? value * 0.1f : value;
                }
            } else if (layer.kind == Kind::rnn || layer.kind == Kind::lstm || layer.kind == Kind::gru) {
                result = network_.cells_[index]->step(previous);
            } else if (layer.kind == Kind::l2norm) {
                preactivation[t][index] = {l2norm_forward(previous.data(), result.data(), result.size())};
            } else {  // softmax
                float maximum = *std::max_element(previous.begin(), previous.end());
                float total = 0.0f;
                for (std::size_t j = 0; j < previous.size(); ++j) {
                    result[j] = std::exp(previous[j] - maximum);
                    total += result[j];
                }
                for (float& value : result) value /= total;
            }
        }
    }

    std::vector<ClassificationResult> results(T);
    for (std::size_t t = 0; t < T; ++t) {
        const auto& probabilities = output[t].back();
        float probability = probabilities[static_cast<std::size_t>(class_ids[t])];
        float loss = -std::log(std::max(probability, std::numeric_limits<float>::min()));
        if (!std::isfinite(loss)) throw std::runtime_error("non-finite classification loss");
        bool correct = static_cast<int>(std::max_element(probabilities.begin(), probabilities.end()) -
                                         probabilities.begin()) == class_ids[t];
        gradient[t].back() = probabilities;
        gradient[t].back()[static_cast<std::size_t>(class_ids[t])] -= 1.0f;
        results[t] = {loss, correct};
    }

    const float inv_t = 1.0f / static_cast<float>(T);
    for (std::size_t reverse = layers.size(); reverse-- > 0;) {
        auto& layer = layers[reverse];
        if (layer.kind == Kind::softmax) {
            if (reverse == 0) throw std::runtime_error("softmax requires preceding layer");
            for (std::size_t t = 0; t < T; ++t)
                for (std::size_t j = 0; j < gradient[t][reverse].size(); ++j)
                    gradient[t][reverse - 1][j] += gradient[t][reverse][j];
        } else if (layer.kind == Kind::maxpool) {
            if (reverse)
                for (std::size_t t = 0; t < T; ++t)
                    for (std::size_t j = 0; j < gradient[t][reverse].size(); ++j)
                        gradient[t][reverse - 1][max_index[t][reverse][j]] += gradient[t][reverse][j];
        } else if (layer.kind == Kind::l2norm) {
            if (reverse)
                for (std::size_t t = 0; t < T; ++t)
                    l2norm_backward(output[t][reverse].data(), gradient[t][reverse].data(),
                                    gradient[t][reverse - 1].data(), output[t][reverse].size(),
                                    preactivation[t][reverse][0]);
        } else if (layer.kind == Kind::connected) {
            std::vector<float> weight_grad(layer.weights.size(), 0.0f);
            std::vector<float> bias_grad(layer.biases.size(), 0.0f);
            for (std::size_t t = 0; t < T; ++t) {
                const auto& previous = reverse ? output[t][reverse - 1] : inputs[t];
                auto& grad = gradient[t][reverse];
                for (int oc = 0; oc < layer.filters; ++oc) {
                    std::size_t o = static_cast<std::size_t>(oc);
                    float g = grad[o] * (layer.leaky && preactivation[t][reverse][o] < 0 ? 0.1f : 1.0f);
                    bias_grad[o] += g;
                    for (std::size_t ic = 0; ic < previous.size(); ++ic) {
                        std::size_t wi = o * previous.size() + ic;
                        weight_grad[wi] += g * previous[ic];
                        if (reverse) gradient[t][reverse - 1][ic] += g * layer.weights[wi];
                    }
                }
            }
            for (std::size_t j = 0; j < layer.weights.size(); ++j)
                layer.weights[j] -= learning_rate_ * (weight_grad[j] * inv_t + weight_decay_ * layer.weights[j]);
            for (std::size_t j = 0; j < layer.biases.size(); ++j)
                layer.biases[j] -= learning_rate_ * bias_grad[j] * inv_t;
        } else if (layer.kind == Kind::rnn || layer.kind == Kind::lstm || layer.kind == Kind::gru) {
            auto& cell = *network_.cells_[reverse];
            std::vector<std::vector<float>> dh_targets(T);
            for (std::size_t t = 0; t < T; ++t) dh_targets[t] = gradient[t][reverse];
            cell.zero_gradients();
            auto dinputs = cell.backward_through_time(dh_targets);
            if (reverse)
                for (std::size_t t = 0; t < T; ++t)
                    for (std::size_t p = 0; p < dinputs[t].size(); ++p) gradient[t][reverse - 1][p] += dinputs[t][p];
            cell.sgd_step(learning_rate_);
        } else {
            // Convolution (and, for batch_normalize layers, the fixed-statistics affine epilogue
            // already folded into forward above - same simplification step() uses).
            const int in_plane = layer.input.h * layer.input.w;
            const int out_plane = layer.output.h * layer.output.w;
            std::vector<float> weight_grad(layer.weights.size(), 0.0f);
            std::vector<float> bias_grad(layer.biases.size(), 0.0f);
            for (std::size_t t = 0; t < T; ++t) {
                const auto& previous = reverse ? output[t][reverse - 1] : inputs[t];
                auto& grad = gradient[t][reverse];
                for (int oc = 0; oc < layer.output.c; ++oc) {
                    float bn_scale = layer.batch_normalize ?
                        layer.scales[oc] / std::sqrt(layer.variances[oc] + 1e-5f) : 1.0f;
                    for (int oy = 0; oy < layer.output.h; ++oy)
                        for (int ox = 0; ox < layer.output.w; ++ox) {
                            auto out_pos = static_cast<std::size_t>(oc) * out_plane + oy * layer.output.w + ox;
                            float g = grad[out_pos] *
                                (layer.leaky && preactivation[t][reverse][out_pos] < 0 ? 0.1f : 1.0f);
                            bias_grad[static_cast<std::size_t>(oc)] += g;
                            float scaled = g * bn_scale;
                            for (int ic = 0; ic < layer.input.c; ++ic)
                                for (int ky = 0; ky < layer.size; ++ky)
                                    for (int kx = 0; kx < layer.size; ++kx) {
                                        int iy = oy * layer.stride + ky - layer.padding;
                                        int ix = ox * layer.stride + kx - layer.padding;
                                        if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                                        std::size_t wi = ((static_cast<std::size_t>(oc) * layer.input.c + ic) *
                                            layer.size + ky) * layer.size + kx;
                                        std::size_t source = static_cast<std::size_t>(ic) * in_plane +
                                            iy * layer.input.w + ix;
                                        weight_grad[wi] += scaled * previous[source];
                                        if (reverse) gradient[t][reverse - 1][source] += scaled * layer.weights[wi];
                                    }
                        }
                }
            }
            for (std::size_t j = 0; j < layer.weights.size(); ++j)
                layer.weights[j] -= learning_rate_ * (weight_grad[j] * inv_t + weight_decay_ * layer.weights[j]);
            for (std::size_t j = 0; j < layer.biases.size(); ++j)
                layer.biases[j] -= learning_rate_ * bias_grad[j] * inv_t;
        }
    }
    return results;
}

}  // namespace detector
