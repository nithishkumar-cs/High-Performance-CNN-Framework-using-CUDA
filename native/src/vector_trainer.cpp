#include "detector/classifier_trainer.hpp"
#include "detector/l2norm.hpp"
#include "detector/normalization.hpp"
#include "detector/local.hpp"
#include "detector/deconvolutional.hpp"
#include "detector/avgpool.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace detector {

/** Validates a feedforward (no softmax, no recurrent layer) graph and its SGD hyperparameters. */
VectorTrainer::VectorTrainer(Network& network, VectorLoss loss, float learning_rate, float weight_decay)
    : network_(network), loss_(loss), learning_rate_(learning_rate), weight_decay_(weight_decay) {
    if (!std::isfinite(learning_rate) || learning_rate <= 0 ||
        !std::isfinite(weight_decay) || weight_decay < 0)
        throw std::runtime_error("invalid vector SGD parameters");
    const auto& layers = network_.layers_;
    if (layers.empty()) throw std::runtime_error("vector trainer requires at least one layer");
    for (const auto& layer : layers) {
        if (layer.kind == Kind::softmax)
            throw std::runtime_error("vector trainer cfg must not end in [softmax] - its loss is applied directly to the final layer's raw output");
        if (layer.kind == Kind::rnn || layer.kind == Kind::lstm || layer.kind == Kind::gru)
            throw std::runtime_error("vector trainer does not support recurrent layers");
        if (layer.kind != Kind::convolutional && layer.kind != Kind::local &&
            layer.kind != Kind::deconvolutional && layer.kind != Kind::maxpool && layer.kind != Kind::avgpool &&
            layer.kind != Kind::connected && layer.kind != Kind::l2norm && layer.kind != Kind::normalization)
            throw std::runtime_error("unsupported layer in vector trainer");
    }
    if (loss_ == VectorLoss::pixel_softmax_ce && layers.back().output.c < 2)
        throw std::runtime_error("pixel_softmax_ce requires at least two output channels");
}

/** Applies one sample's MSE, sigmoid-BCE, or pixelwise softmax-CE SGD update. */
VectorResult VectorTrainer::step(const std::vector<float>& input, const std::vector<float>& targets) {
    if (input.size() != network_.input_.elements())
        throw std::runtime_error("vector trainer input shape mismatch");
    auto& layers = network_.layers_;
    const auto final_shape = layers.back().output;
    const std::size_t pixels = static_cast<std::size_t>(final_shape.h) * final_shape.w;
    const std::size_t expected_targets = loss_ == VectorLoss::pixel_softmax_ce ?
        pixels : final_shape.elements();
    if (targets.size() != expected_targets)
        throw std::runtime_error("vector trainer target shape mismatch");

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
                throw std::runtime_error("initialize or load vector trainer weights before training");
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
        } else if (layer.kind == Kind::local) {
            if (layer.biases.size() != result.size() ||
                layer.weights.size() != result.size() *
                    static_cast<std::size_t>(layer.input.c) * layer.size * layer.size)
                throw std::runtime_error("initialize or load local vector trainer weights before training");
            local_forward(layer, previous.data(), result.data());
        } else if (layer.kind == Kind::deconvolutional) {
            if (layer.biases.size() != static_cast<std::size_t>(layer.filters) ||
                layer.weights.size() != static_cast<std::size_t>(layer.filters) *
                    layer.input.c * layer.size * layer.size)
                throw std::runtime_error("initialize or load deconvolutional vector trainer weights before training");
            deconvolutional_forward(layer, previous.data(), result.data());
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
        } else if (layer.kind == Kind::avgpool) {
            avgpool_forward(previous.data(), result.data(), layer.input.c, layer.input.h * layer.input.w);
        } else if (layer.kind == Kind::connected) {
            if (layer.biases.size() != static_cast<std::size_t>(layer.filters) ||
                layer.weights.size() != static_cast<std::size_t>(layer.filters) * previous.size())
                throw std::runtime_error("initialize or load vector trainer weights before training");
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
        } else {  // Kind::normalization
            auto& norms = preactivation[index];
            norms.resize(result.size());
            normalization_forward(previous.data(), result.data(), norms.data(), layer.input.c,
                layer.input.h * layer.input.w, layer.size, layer.norm_alpha, layer.norm_beta,
                layer.norm_kappa);
        }
    }

    const auto& raw = output.back();
    const std::size_t n = raw.size();
    const float inv_n = 1.0f / static_cast<float>(n);
    float loss = 0.0f;
    std::size_t valid_pixels = pixels;
    std::vector<float> reported(n);
    auto& output_gradient = gradient.back();
    if (loss_ == VectorLoss::mse) {
        for (std::size_t j = 0; j < n; ++j) {
            float diff = raw[j] - targets[j];
            loss += 0.5f * diff * diff;
            output_gradient[j] = diff * inv_n;
        }
        reported = raw;
    } else if (loss_ == VectorLoss::sigmoid_bce) {
        for (std::size_t j = 0; j < n; ++j) {
            if (targets[j] < 0.0f || targets[j] > 1.0f)
                throw std::runtime_error("sigmoid_bce targets must be in [0,1]");
            float p = 1.0f / (1.0f + std::exp(-raw[j]));
            float clamped = std::min(std::max(p, std::numeric_limits<float>::min()), 1.0f - 1e-7f);
            loss += -(targets[j] * std::log(clamped) + (1.0f - targets[j]) * std::log(1.0f - clamped));
            output_gradient[j] = (p - targets[j]) * inv_n;
            reported[j] = p;
        }
    } else {  // pixel_softmax_ce; targets are integer class IDs in row-major HxW order.
        valid_pixels = 0;
        for (float target : targets) {
            if (!std::isfinite(target) || (target != -1.0f &&
                (target < 0.0f || target >= final_shape.c || std::floor(target) != target)))
                throw std::runtime_error("pixel_softmax_ce targets must be integer class IDs or -1 for ignore");
            if (target != -1.0f) ++valid_pixels;
        }
        if (!valid_pixels) throw std::runtime_error("pixel_softmax_ce requires at least one labeled pixel");
        const float inv_pixels = 1.0f / static_cast<float>(valid_pixels);
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const float target = targets[pixel];
            const bool ignored = target == -1.0f;
            const std::size_t truth = ignored ? 0 : static_cast<std::size_t>(target);
            float maximum = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < final_shape.c; ++c)
                maximum = std::max(maximum, raw[static_cast<std::size_t>(c) * pixels + pixel]);
            double sum = 0.0;
            for (int c = 0; c < final_shape.c; ++c) {
                const std::size_t pos = static_cast<std::size_t>(c) * pixels + pixel;
                const float value = std::exp(raw[pos] - maximum);
                reported[pos] = value;
                sum += value;
            }
            if (!ignored)
                loss += static_cast<float>(std::log(sum) + maximum - raw[truth * pixels + pixel]);
            for (int c = 0; c < final_shape.c; ++c) {
                const std::size_t pos = static_cast<std::size_t>(c) * pixels + pixel;
                reported[pos] = static_cast<float>(reported[pos] / sum);
                output_gradient[pos] = ignored ? 0.0f :
                    (reported[pos] - (static_cast<std::size_t>(c) == truth ? 1.0f : 0.0f)) * inv_pixels;
            }
        }
    }
    if (loss_ != VectorLoss::pixel_softmax_ce) loss *= inv_n;
    else loss /= static_cast<float>(valid_pixels);
    if (!std::isfinite(loss)) throw std::runtime_error("non-finite vector trainer loss");

    for (std::size_t reverse = layers.size(); reverse-- > 0;) {
        auto& layer = layers[reverse];
        const auto& previous = reverse ? output[reverse - 1] : input;
        auto& grad = gradient[reverse];
        if (layer.kind == Kind::maxpool) {
            if (reverse)
                for (std::size_t j = 0; j < grad.size(); ++j)
                    gradient[reverse - 1][max_index[reverse][j]] += grad[j];
        } else if (layer.kind == Kind::avgpool) {
            if (reverse)
                avgpool_backward(grad.data(), gradient[reverse - 1].data(),
                                 layer.input.c, layer.input.h * layer.input.w);
        } else if (layer.kind == Kind::l2norm) {
            if (reverse)
                l2norm_backward(output[reverse].data(), grad.data(), gradient[reverse - 1].data(),
                                grad.size(), preactivation[reverse][0]);
        } else if (layer.kind == Kind::normalization) {
            if (reverse)
                normalization_backward(previous.data(), output[reverse].data(),
                    preactivation[reverse].data(), grad.data(), gradient[reverse - 1].data(),
                    layer.input.c, layer.input.h * layer.input.w, layer.size,
                    layer.norm_alpha, layer.norm_beta);
        } else if (layer.kind == Kind::local) {
            std::vector<float> weight_grad(layer.weights.size(), 0.0f);
            std::vector<float> bias_grad(layer.biases.size(), 0.0f);
            local_backward(layer, previous.data(), output[reverse].data(), grad.data(),
                reverse ? gradient[reverse - 1].data() : nullptr,
                weight_grad.data(), bias_grad.data());
            for (std::size_t j = 0; j < layer.weights.size(); ++j)
                layer.weights[j] -= learning_rate_ * (weight_grad[j] + weight_decay_ * layer.weights[j]);
            for (std::size_t j = 0; j < layer.biases.size(); ++j)
                layer.biases[j] -= learning_rate_ * bias_grad[j];
        } else if (layer.kind == Kind::deconvolutional) {
            std::vector<float> weight_grad(layer.weights.size(), 0.0f);
            std::vector<float> bias_grad(layer.biases.size(), 0.0f);
            deconvolutional_backward(layer, previous.data(), output[reverse].data(), grad.data(),
                reverse ? gradient[reverse - 1].data() : nullptr,
                weight_grad.data(), bias_grad.data());
            for (std::size_t j = 0; j < layer.weights.size(); ++j)
                layer.weights[j] -= learning_rate_ * (weight_grad[j] + weight_decay_ * layer.weights[j]);
            for (std::size_t j = 0; j < layer.biases.size(); ++j)
                layer.biases[j] -= learning_rate_ * bias_grad[j];
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
        } else {  // Kind::convolutional
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
    return {loss, std::move(reported)};
}

}  // namespace detector
