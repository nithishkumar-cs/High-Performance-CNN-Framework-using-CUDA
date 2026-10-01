#include "detector/trainer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace detector {
namespace {

float sigmoid(float x) {
    if (x >= 0) return 1.0f / (1.0f + std::exp(-x));
    float e = std::exp(x);
    return e / (1.0f + e);
}

float binary_cross_entropy(float logit, float target) {
    return std::max(logit, 0.0f) - logit * target + std::log1p(std::exp(-std::abs(logit)));
}

float anchor_iou(float aw, float ah, float bw, float bh) {
    float intersection = std::min(aw, bw) * std::min(ah, bh);
    return intersection / (aw * ah + bw * bh - intersection);
}

void write_floats(std::ofstream& file, const std::vector<float>& values) {
    if (values.empty()) throw std::runtime_error("cannot save uninitialized convolution weights");
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

struct Assignment {
    int anchor = 0;
    int cell = 0;
    int class_id = 0;
    float offset_x = 0;
    float offset_y = 0;
    float log_width = 0;
    float log_height = 0;
};

}  // namespace

void Network::initialize_random(std::uint32_t seed) {
    std::mt19937 rng(seed);
    for (auto& layer : layers_) {
        if (layer.kind != Kind::convolutional && layer.kind != Kind::connected) continue;
        const std::size_t count = static_cast<std::size_t>(layer.filters);
        layer.biases.assign(count, 0.0f);
        if (layer.batch_normalize) {
            layer.scales.assign(count, 1.0f);
            layer.means.assign(count, 0.0f);
            layer.variances.assign(count, 1.0f);
        }
        const auto fan_in = layer.kind == Kind::connected ? static_cast<int>(layer.input.elements()) :
            layer.input.c * layer.size * layer.size;
        std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / fan_in));
        layer.weights.resize(count * static_cast<std::size_t>(fan_in));
        for (auto& weight : layer.weights) weight = distribution(rng);
    }
}

void Network::save_weights(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot write weights: " + path);
    const std::int32_t version[3] = {0, 2, 0};
    const std::uint64_t seen = 0;
    file.write(reinterpret_cast<const char*>(version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&seen), sizeof(seen));
    for (const auto& layer : layers_) {
        if (layer.kind != Kind::convolutional && layer.kind != Kind::connected) continue;
        write_floats(file, layer.biases);
        if (layer.batch_normalize) {
            write_floats(file, layer.scales);
            write_floats(file, layer.means);
            write_floats(file, layer.variances);
        }
        write_floats(file, layer.weights);
    }
    if (!file) throw std::runtime_error("failed writing weights: " + path);
}

float LearningRateSchedule::at(int iteration) const {
    if (burn_in > 0 && iteration < burn_in)
        return base_lr * std::pow((iteration + 1) / static_cast<float>(burn_in), burn_in_power);
    float lr = base_lr;
    for (std::size_t i = 0; i < steps.size(); ++i) if (iteration >= steps[i]) lr *= scales[i];
    return lr;
}

Trainer::Trainer(Network& network, LearningRateSchedule schedule, int batch_size, float weight_decay, float bn_momentum)
    : network_(network), schedule_(std::move(schedule)), batch_size_(batch_size), weight_decay_(weight_decay),
      bn_momentum_(bn_momentum) {
    if (!std::isfinite(schedule_.base_lr) || schedule_.base_lr <= 0 || schedule_.burn_in < 0 ||
        schedule_.steps.size() != schedule_.scales.size())
        throw std::runtime_error("invalid learning rate schedule");
    if (!std::isfinite(weight_decay) || weight_decay < 0) throw std::runtime_error("invalid weight decay");
    if (batch_size < 1) throw std::runtime_error("invalid batch size");
    if (!std::isfinite(bn_momentum) || bn_momentum <= 0 || bn_momentum >= 1) throw std::runtime_error("invalid batch-norm momentum");
    weight_grad_.resize(network_.layers_.size());
    bias_grad_.resize(network_.layers_.size());
    scale_grad_.resize(network_.layers_.size());
}

TrainResult Trainer::step(const std::vector<float>& input, const std::vector<Target>& targets) {
    if (input.size() != network_.input_.elements()) throw std::runtime_error("training input shape mismatch");
    auto& layers = network_.layers_;
    std::vector<std::vector<float>> output(layers.size());
    std::vector<std::vector<float>> preactivation(layers.size());
    std::vector<std::vector<float>> gradient(layers.size());
    // Per-sample batch-norm statistics (over this sample's H*W positions per channel) and the
    // pre-normalization conv sum, kept only for batch_normalize conv layers, needed by backward.
    std::vector<std::vector<float>> raw(layers.size());
    std::vector<std::vector<float>> batch_mean(layers.size());
    std::vector<std::vector<float>> batch_var(layers.size());
    constexpr float bn_eps = 1e-5f;

    // CPU forward reference, retaining tensors needed for reverse-mode gradients.
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto& layer = layers[index];
        const auto& previous = index ? output[index - 1] : input;
        auto& result = output[index];
        result.resize(layer.output.elements());
        gradient[index].assign(result.size(), 0.0f);
        if (layer.kind == Kind::convolutional) {
            if (layer.biases.size() != static_cast<std::size_t>(layer.filters) || layer.weights.empty())
                throw std::runtime_error("initialize or load weights before training");
            preactivation[index].resize(result.size());
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            auto conv_sum = [&](int oc, int oy, int ox) {
                float sum = 0;
                for (int ic = 0; ic < layer.input.c; ++ic)
                    for (int ky = 0; ky < layer.size; ++ky)
                        for (int kx = 0; kx < layer.size; ++kx) {
                            int iy = oy * layer.stride + ky - layer.padding;
                            int ix = ox * layer.stride + kx - layer.padding;
                            if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                            std::size_t wi = ((static_cast<std::size_t>(oc) * layer.input.c + ic) * layer.size + ky) * layer.size + kx;
                            sum += previous[static_cast<std::size_t>(ic) * in_plane + iy * layer.input.w + ix] * layer.weights[wi];
                        }
                return sum;
            };
            if (layer.batch_normalize) {
                raw[index].resize(result.size());
                batch_mean[index].assign(static_cast<std::size_t>(layer.output.c), 0.0f);
                batch_var[index].assign(static_cast<std::size_t>(layer.output.c), 0.0f);
            }
            for (int oc = 0; oc < layer.output.c; ++oc) {
                if (layer.batch_normalize) {
                    float sum_z = 0;
                    for (int oy = 0; oy < layer.output.h; ++oy)
                        for (int ox = 0; ox < layer.output.w; ++ox) {
                            float z = conv_sum(oc, oy, ox);
                            auto pos = static_cast<std::size_t>(oc) * out_plane + oy * layer.output.w + ox;
                            raw[index][pos] = z;
                            sum_z += z;
                        }
                    float mean = sum_z / out_plane;
                    float var_sum = 0;
                    for (int p = 0; p < out_plane; ++p) {
                        float d = raw[index][static_cast<std::size_t>(oc) * out_plane + p] - mean;
                        var_sum += d * d;
                    }
                    float variance = var_sum / out_plane;
                    batch_mean[index][oc] = mean;
                    batch_var[index][oc] = variance;
                    float inv_std = 1.0f / std::sqrt(variance + bn_eps);
                    float gamma = layer.scales[oc];
                    float beta = layer.biases[oc];
                    for (int p = 0; p < out_plane; ++p) {
                        auto pos = static_cast<std::size_t>(oc) * out_plane + p;
                        float xhat = (raw[index][pos] - mean) * inv_std;
                        float value = gamma * xhat + beta;
                        preactivation[index][pos] = value;
                        result[pos] = layer.leaky && value < 0 ? value * 0.1f : value;
                    }
                    layer.means[oc] = bn_momentum_ * layer.means[oc] + (1 - bn_momentum_) * mean;
                    layer.variances[oc] = bn_momentum_ * layer.variances[oc] + (1 - bn_momentum_) * variance;
                } else {
                    for (int oy = 0; oy < layer.output.h; ++oy)
                        for (int ox = 0; ox < layer.output.w; ++ox) {
                            float value = conv_sum(oc, oy, ox) + layer.biases[oc];
                            auto pos = static_cast<std::size_t>(oc) * out_plane + oy * layer.output.w + ox;
                            preactivation[index][pos] = value;
                            result[pos] = layer.leaky && value < 0 ? value * 0.1f : value;
                        }
                }
            }
        } else if (layer.kind == Kind::shortcut) {
            const auto& branch = output[static_cast<std::size_t>(layer.sources.front())];
            preactivation[index].resize(result.size());
            for (std::size_t p = 0; p < result.size(); ++p) {
                float value = previous[p] + branch[p];
                preactivation[index][p] = value;
                result[p] = layer.leaky && value < 0 ? value * 0.1f : value;
            }
        } else if (layer.kind == Kind::maxpool) {
            int pad = (layer.size - 1) / 2;
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            for (int c = 0; c < layer.input.c; ++c)
                for (int oy = 0; oy < layer.output.h; ++oy)
                    for (int ox = 0; ox < layer.output.w; ++ox) {
                        float best = -std::numeric_limits<float>::infinity();
                        for (int ky = 0; ky < layer.size; ++ky)
                            for (int kx = 0; kx < layer.size; ++kx) {
                                int iy = oy * layer.stride + ky - pad;
                                int ix = ox * layer.stride + kx - pad;
                                if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                                best = std::max(best, previous[static_cast<std::size_t>(c) * in_plane + iy * layer.input.w + ix]);
                            }
                        result[static_cast<std::size_t>(c) * out_plane + oy * layer.output.w + ox] = best;
                    }
        } else if (layer.kind == Kind::route) {
            std::size_t offset = 0;
            for (int source : layer.sources) {
                const auto& branch = output[static_cast<std::size_t>(source)];
                std::copy(branch.begin(), branch.end(), result.begin() + static_cast<std::ptrdiff_t>(offset));
                offset += branch.size();
            }
        } else if (layer.kind == Kind::upsample) {
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            for (int c = 0; c < layer.input.c; ++c)
                for (int y = 0; y < layer.output.h; ++y)
                    for (int x = 0; x < layer.output.w; ++x)
                        result[static_cast<std::size_t>(c) * out_plane + y * layer.output.w + x] =
                            previous[static_cast<std::size_t>(c) * in_plane + (y / layer.stride) * layer.input.w + x / layer.stride];
        } else {
            result = previous;
        }
    }

    // Assign each target to the best active anchor across all YOLO heads.
    std::vector<std::vector<Assignment>> assignments(layers.size());
    for (const auto& target : targets) {
        if (!(target.cx >= 0 && target.cx <= 1 && target.cy >= 0 && target.cy <= 1 &&
              target.width > 0 && target.width <= 1 && target.height > 0 && target.height <= 1))
            throw std::runtime_error("target coordinates must be normalized to [0,1]");
        int best_layer = -1;
        int best_anchor = -1;
        float best_iou = -1;
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const auto& layer = layers[index];
            if (layer.kind != Kind::yolo) continue;
            if (target.class_id < 0 || target.class_id >= layer.classes) throw std::runtime_error("target class outside cfg classes");
            for (int slot = 0; slot < static_cast<int>(layer.mask.size()); ++slot) {
                const auto [aw, ah] = layer.anchors[static_cast<std::size_t>(layer.mask[slot])];
                float overlap = anchor_iou(aw, ah, target.width * network_.input_.w,
                                          target.height * network_.input_.h);
                if (overlap > best_iou) { best_iou = overlap; best_layer = static_cast<int>(index); best_anchor = slot; }
            }
        }
        if (best_layer < 0) throw std::runtime_error("no YOLO head for target");
        const auto& layer = layers[static_cast<std::size_t>(best_layer)];
        float grid_x = target.cx * layer.output.w;
        float grid_y = target.cy * layer.output.h;
        int cell_x = std::clamp(static_cast<int>(grid_x), 0, layer.output.w - 1);
        int cell_y = std::clamp(static_cast<int>(grid_y), 0, layer.output.h - 1);
        const auto [aw, ah] = layer.anchors[static_cast<std::size_t>(layer.mask[best_anchor])];
        assignments[static_cast<std::size_t>(best_layer)].push_back({best_anchor,
            cell_y * layer.output.w + cell_x, target.class_id,
            std::clamp(grid_x - cell_x, 0.0f, 1.0f), std::clamp(grid_y - cell_y, 0.0f, 1.0f),
            std::log(target.width * network_.input_.w / aw),
            std::log(target.height * network_.input_.h / ah)});
    }

    float loss = 0;
    std::size_t candidate_count = 0;
    for (const auto& layer : layers) if (layer.kind == Kind::yolo)
        candidate_count += layer.mask.size() * static_cast<std::size_t>(layer.output.h) * layer.output.w;
    const float negative_weight = 0.5f / std::max<std::size_t>(1, candidate_count);
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto& layer = layers[index];
        if (layer.kind != Kind::yolo) continue;
        int area = layer.output.h * layer.output.w;
        int attrs = layer.classes + 5;
        auto pos = [&](int anchor, int attribute, int cell) {
            return (static_cast<std::size_t>(anchor) * attrs + attribute) * area + cell;
        };
        std::vector<unsigned char> positive(layer.mask.size() * static_cast<std::size_t>(area), 0);
        for (const auto& match : assignments[index]) positive[static_cast<std::size_t>(match.anchor) * area + match.cell] = 1;
        for (int anchor = 0; anchor < static_cast<int>(layer.mask.size()); ++anchor)
            for (int cell = 0; cell < area; ++cell) {
                if (positive[static_cast<std::size_t>(anchor) * area + cell]) continue;
                auto p = pos(anchor, 4, cell);
                loss += negative_weight * binary_cross_entropy(output[index][p], 0.0f);
                gradient[index][p] += negative_weight * sigmoid(output[index][p]);
            }
        for (const auto& match : assignments[index]) {
            auto obj = pos(match.anchor, 4, match.cell);
            loss += binary_cross_entropy(output[index][obj], 1.0f);
            gradient[index][obj] += sigmoid(output[index][obj]) - 1.0f;
            for (int attr = 0; attr < 2; ++attr) {
                float wanted = attr == 0 ? match.offset_x : match.offset_y;
                auto p = pos(match.anchor, attr, match.cell);
                loss += binary_cross_entropy(output[index][p], wanted);
                gradient[index][p] += sigmoid(output[index][p]) - wanted;
            }
            for (int attr = 2; attr < 4; ++attr) {
                float wanted = attr == 2 ? match.log_width : match.log_height;
                auto p = pos(match.anchor, attr, match.cell);
                float error = output[index][p] - wanted;
                loss += 0.5f * error * error;
                gradient[index][p] += error;
            }
            for (int cls = 0; cls < layer.classes; ++cls) {
                float wanted = cls == match.class_id ? 1.0f : 0.0f;
                auto p = pos(match.anchor, 5 + cls, match.cell);
                loss += binary_cross_entropy(output[index][p], wanted);
                gradient[index][p] += sigmoid(output[index][p]) - wanted;
            }
        }
    }
    const float normalizer = static_cast<float>(std::max<std::size_t>(1, targets.size()));
    loss /= normalizer;
    for (auto& tensor : gradient) for (auto& value : tensor) value /= normalizer;

    // Reverse graph traversal: convolution weights/biases, and scale+bias (gamma/beta) for
    // batch_normalize layers, accumulate per-sample gradients into this trainer's minibatch buffers.
    for (std::size_t reverse = layers.size(); reverse-- > 0;) {
        auto& layer = layers[reverse];
        const auto& previous = reverse ? output[reverse - 1] : input;
        auto& grad = gradient[reverse];
        if (layer.kind == Kind::yolo) {
            if (reverse) for (std::size_t p = 0; p < grad.size(); ++p) gradient[reverse - 1][p] += grad[p];
        } else if (layer.kind == Kind::shortcut) {
            int source = layer.sources.front();
            for (std::size_t p = 0; p < grad.size(); ++p) {
                float g = grad[p] * (layer.leaky && preactivation[reverse][p] < 0 ? 0.1f : 1.0f);
                if (reverse) gradient[reverse - 1][p] += g;
                gradient[static_cast<std::size_t>(source)][p] += g;
            }
        } else if (layer.kind == Kind::route) {
            std::size_t offset = 0;
            for (int source : layer.sources) {
                auto& branch = gradient[static_cast<std::size_t>(source)];
                for (std::size_t p = 0; p < branch.size(); ++p) branch[p] += grad[offset + p];
                offset += branch.size();
            }
        } else if (layer.kind == Kind::upsample) {
            if (!reverse) continue;
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            for (int c = 0; c < layer.input.c; ++c)
                for (int y = 0; y < layer.output.h; ++y)
                    for (int x = 0; x < layer.output.w; ++x)
                        gradient[reverse - 1][static_cast<std::size_t>(c) * in_plane +
                            (y / layer.stride) * layer.input.w + x / layer.stride] +=
                            grad[static_cast<std::size_t>(c) * out_plane + y * layer.output.w + x];
        } else if (layer.kind == Kind::maxpool) {
            if (!reverse) continue;
            int pad = (layer.size - 1) / 2;
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
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
                                std::size_t p = static_cast<std::size_t>(c) * in_plane + iy * layer.input.w + ix;
                                if (previous[p] > best) { best = previous[p]; best_pos = p; }
                            }
                        gradient[reverse - 1][best_pos] += grad[static_cast<std::size_t>(c) * out_plane + oy * layer.output.w + ox];
                    }
        } else {
            // Convolution (and, for batch_normalize layers, batch-norm) backward. Gradients are
            // accumulated into this trainer's minibatch buffers; the SGD update is applied once
            // per batch_size samples, below.
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            if (weight_grad_[reverse].empty()) weight_grad_[reverse].assign(layer.weights.size(), 0.0f);
            if (bias_grad_[reverse].empty()) bias_grad_[reverse].assign(layer.biases.size(), 0.0f);
            if (layer.batch_normalize && scale_grad_[reverse].empty())
                scale_grad_[reverse].assign(static_cast<std::size_t>(layer.filters), 0.0f);
            std::vector<float> dz(static_cast<std::size_t>(out_plane));  // gradient w.r.t. pre-BN conv sum
            for (int oc = 0; oc < layer.output.c; ++oc) {
                if (layer.batch_normalize) {
                    float mean = batch_mean[reverse][oc];
                    float inv_std = 1.0f / std::sqrt(batch_var[reverse][oc] + bn_eps);
                    float gamma = layer.scales[oc];
                    float sum_dxhat = 0, sum_dxhat_xhat = 0;
                    std::vector<float> dxhat(static_cast<std::size_t>(out_plane));
                    for (int p = 0; p < out_plane; ++p) {
                        auto pos = static_cast<std::size_t>(oc) * out_plane + p;
                        float g = grad[pos] * (layer.leaky && preactivation[reverse][pos] < 0 ? 0.1f : 1.0f);
                        float xhat = (raw[reverse][pos] - mean) * inv_std;
                        scale_grad_[reverse][oc] += g * xhat;
                        bias_grad_[reverse][oc] += g;
                        dxhat[static_cast<std::size_t>(p)] = g * gamma;
                        sum_dxhat += dxhat[static_cast<std::size_t>(p)];
                        sum_dxhat_xhat += dxhat[static_cast<std::size_t>(p)] * xhat;
                    }
                    for (int p = 0; p < out_plane; ++p) {
                        auto pos = static_cast<std::size_t>(oc) * out_plane + p;
                        float xhat = (raw[reverse][pos] - mean) * inv_std;
                        dz[static_cast<std::size_t>(p)] = inv_std / out_plane *
                            (out_plane * dxhat[static_cast<std::size_t>(p)] - sum_dxhat - xhat * sum_dxhat_xhat);
                    }
                } else {
                    for (int p = 0; p < out_plane; ++p) {
                        auto pos = static_cast<std::size_t>(oc) * out_plane + p;
                        float g = grad[pos] * (layer.leaky && preactivation[reverse][pos] < 0 ? 0.1f : 1.0f);
                        bias_grad_[reverse][oc] += g;
                        dz[static_cast<std::size_t>(p)] = g;
                    }
                }
                for (int oy = 0; oy < layer.output.h; ++oy)
                    for (int ox = 0; ox < layer.output.w; ++ox) {
                        float gsum = dz[static_cast<std::size_t>(oy * layer.output.w + ox)];
                        for (int ic = 0; ic < layer.input.c; ++ic)
                            for (int ky = 0; ky < layer.size; ++ky)
                                for (int kx = 0; kx < layer.size; ++kx) {
                                    int iy = oy * layer.stride + ky - layer.padding;
                                    int ix = ox * layer.stride + kx - layer.padding;
                                    if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                                    std::size_t wi = ((static_cast<std::size_t>(oc) * layer.input.c + ic) * layer.size + ky) * layer.size + kx;
                                    auto source_pos = static_cast<std::size_t>(ic) * in_plane + iy * layer.input.w + ix;
                                    weight_grad_[reverse][wi] += gsum * previous[source_pos];
                                    if (reverse) gradient[reverse - 1][source_pos] += gsum * layer.weights[wi];
                                }
                    }
            }
        }
    }

    bool updated = false;
    if (++accumulated_ >= batch_size_) {
        float lr = schedule_.at(iteration_);
        for (std::size_t index = 0; index < layers.size(); ++index) {
            auto& layer = layers[index];
            if (layer.kind != Kind::convolutional && layer.kind != Kind::connected) continue;
            for (std::size_t p = 0; p < layer.weights.size(); ++p) {
                layer.weights[p] -= lr * (weight_grad_[index][p] / batch_size_ + weight_decay_ * layer.weights[p]);
                weight_grad_[index][p] = 0.0f;
            }
            for (std::size_t p = 0; p < layer.biases.size(); ++p) {
                layer.biases[p] -= lr * bias_grad_[index][p] / batch_size_;
                bias_grad_[index][p] = 0.0f;
            }
            if (layer.batch_normalize) {
                for (std::size_t p = 0; p < scale_grad_[index].size(); ++p) {
                    layer.scales[p] -= lr * scale_grad_[index][p] / batch_size_;
                    scale_grad_[index][p] = 0.0f;
                }
            }
        }
        accumulated_ = 0;
        ++iteration_;
        updated = true;
    }
    if (!std::isfinite(loss)) throw std::runtime_error("non-finite training loss");
    return {loss, targets.size(), updated};
}

}  // namespace detector
