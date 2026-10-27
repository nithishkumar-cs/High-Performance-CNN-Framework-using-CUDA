#include "detector/trainer.hpp"
#include "detector/yolo_loss.hpp"
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

void write_floats(std::ofstream& file, const std::vector<float>& values) {
    if (values.empty()) throw std::runtime_error("cannot save uninitialized convolution weights");
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

}  // namespace

void Network::initialize_random(std::uint32_t seed) {
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        if (layer.kind == Kind::rnn || layer.kind == Kind::lstm || layer.kind == Kind::gru) {
            RecurrentKind kind = layer.kind == Kind::rnn ? RecurrentKind::rnn :
                layer.kind == Kind::lstm ? RecurrentKind::lstm : RecurrentKind::gru;
            cells_[i].emplace(kind, static_cast<int>(layer.input.elements()), layer.filters,
                               static_cast<unsigned>(rng()));
            continue;
        }
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
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const auto& layer = layers_[i];
        if (layer.kind == Kind::rnn || layer.kind == Kind::lstm || layer.kind == Kind::gru) {
            if (!cells_[i]) throw std::runtime_error("cannot save uninitialized rnn/lstm/gru weights");
            write_floats(file, cells_[i]->parameter_values());
            continue;
        }
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

Trainer::Trainer(Network& network, LearningRateSchedule schedule, int batch_size, float weight_decay,
                  float momentum, float bn_momentum, int initial_iteration)
    : network_(network), schedule_(std::move(schedule)), batch_size_(batch_size), weight_decay_(weight_decay),
      momentum_(momentum), bn_momentum_(bn_momentum), iteration_(initial_iteration) {
    if (initial_iteration < 0) throw std::runtime_error("invalid initial iteration");
    if (!std::isfinite(schedule_.base_lr) || schedule_.base_lr <= 0 || schedule_.burn_in < 0 ||
        schedule_.steps.size() != schedule_.scales.size())
        throw std::runtime_error("invalid learning rate schedule");
    if (!std::isfinite(weight_decay) || weight_decay < 0) throw std::runtime_error("invalid weight decay");
    if (batch_size < 1) throw std::runtime_error("invalid batch size");
    if (!std::isfinite(momentum) || momentum < 0 || momentum >= 1) throw std::runtime_error("invalid momentum");
    if (!std::isfinite(bn_momentum) || bn_momentum <= 0 || bn_momentum >= 1) throw std::runtime_error("invalid batch-norm momentum");
    for (const auto& layer : network_.layers_)
        if (layer.kind == Kind::rnn || layer.kind == Kind::lstm || layer.kind == Kind::gru)
            throw std::runtime_error("the detector Trainer does not support rnn/lstm/gru layers; "
                                      "use ClassificationTrainer for sequence-classification cfgs");
    weight_grad_.resize(network_.layers_.size());
    bias_grad_.resize(network_.layers_.size());
    scale_grad_.resize(network_.layers_.size());
    weight_velocity_.resize(network_.layers_.size());
    bias_velocity_.resize(network_.layers_.size());
    scale_velocity_.resize(network_.layers_.size());
}

TrainResult Trainer::step(const std::vector<std::vector<float>>& inputs,
                           const std::vector<std::vector<Target>>& targets) {
    const std::size_t n = inputs.size();
    if (n == 0 || n > static_cast<std::size_t>(batch_size_))
        throw std::runtime_error("step() requires between 1 and batch_size samples");
    if (targets.size() != n) throw std::runtime_error("inputs/targets length mismatch");
    for (const auto& sample : inputs)
        if (sample.size() != network_.input_.elements()) throw std::runtime_error("training input shape mismatch");
    auto& layers = network_.layers_;
    const std::size_t num_layers = layers.size();
    // Per-sample, per-layer tensors; batch_mean/batch_var below are shared across every sample in
    // this call (true minibatch batch-norm statistics), not computed per sample.
    std::vector<std::vector<std::vector<float>>> output(n), preactivation(n), gradient(n), raw(n);
    for (std::size_t s = 0; s < n; ++s) {
        output[s].resize(num_layers);
        preactivation[s].resize(num_layers);
        gradient[s].resize(num_layers);
        raw[s].resize(num_layers);
    }
    std::vector<std::vector<float>> batch_mean(num_layers), batch_var(num_layers);
    constexpr float bn_eps = 1e-5f;

    // CPU forward reference, retaining tensors needed for reverse-mode gradients. Layer-major so
    // every sample's raw pre-BN conv output at a batch_normalize layer is available before that
    // layer's batch statistics (and hence every sample's normalized output) are finalized.
    for (std::size_t index = 0; index < num_layers; ++index) {
        auto& layer = layers[index];
        auto previous_of = [&](std::size_t s) -> const std::vector<float>& {
            return index ? output[s][index - 1] : inputs[s];
        };
        for (std::size_t s = 0; s < n; ++s) {
            output[s][index].resize(layer.output.elements());
            gradient[s][index].assign(output[s][index].size(), 0.0f);
        }
        if (layer.kind == Kind::convolutional) {
            if (layer.biases.size() != static_cast<std::size_t>(layer.filters) || layer.weights.empty())
                throw std::runtime_error("initialize or load weights before training");
            for (std::size_t s = 0; s < n; ++s) preactivation[s][index].resize(output[s][index].size());
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            auto conv_sum = [&](const std::vector<float>& previous, int oc, int oy, int ox) {
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
                for (std::size_t s = 0; s < n; ++s) raw[s][index].resize(output[s][index].size());
                batch_mean[index].assign(static_cast<std::size_t>(layer.output.c), 0.0f);
                batch_var[index].assign(static_cast<std::size_t>(layer.output.c), 0.0f);
                const double total = static_cast<double>(n) * out_plane;
                for (int oc = 0; oc < layer.output.c; ++oc) {
                    for (std::size_t s = 0; s < n; ++s) {
                        const auto& previous = previous_of(s);
                        for (int oy = 0; oy < layer.output.h; ++oy)
                            for (int ox = 0; ox < layer.output.w; ++ox) {
                                auto pos = static_cast<std::size_t>(oc) * out_plane + oy * layer.output.w + ox;
                                raw[s][index][pos] = conv_sum(previous, oc, oy, ox);
                            }
                    }
                    // True batch statistics: mean/variance over every sample's spatial positions.
                    double sum_z = 0;
                    for (std::size_t s = 0; s < n; ++s)
                        for (int p = 0; p < out_plane; ++p) sum_z += raw[s][index][static_cast<std::size_t>(oc) * out_plane + p];
                    float mean = static_cast<float>(sum_z / total);
                    double var_sum = 0;
                    for (std::size_t s = 0; s < n; ++s)
                        for (int p = 0; p < out_plane; ++p) {
                            double d = raw[s][index][static_cast<std::size_t>(oc) * out_plane + p] - mean;
                            var_sum += d * d;
                        }
                    float variance = static_cast<float>(var_sum / total);
                    batch_mean[index][oc] = mean;
                    batch_var[index][oc] = variance;
                    float inv_std = 1.0f / std::sqrt(variance + bn_eps);
                    float gamma = layer.scales[oc];
                    float beta = layer.biases[oc];
                    for (std::size_t s = 0; s < n; ++s)
                        for (int p = 0; p < out_plane; ++p) {
                            auto pos = static_cast<std::size_t>(oc) * out_plane + p;
                            float xhat = (raw[s][index][pos] - mean) * inv_std;
                            float value = gamma * xhat + beta;
                            preactivation[s][index][pos] = value;
                            output[s][index][pos] = layer.leaky && value < 0 ? value * 0.1f : value;
                        }
                    layer.means[oc] = bn_momentum_ * layer.means[oc] + (1 - bn_momentum_) * mean;
                    layer.variances[oc] = bn_momentum_ * layer.variances[oc] + (1 - bn_momentum_) * variance;
                }
            } else {
                for (std::size_t s = 0; s < n; ++s) {
                    const auto& previous = previous_of(s);
                    for (int oc = 0; oc < layer.output.c; ++oc)
                        for (int oy = 0; oy < layer.output.h; ++oy)
                            for (int ox = 0; ox < layer.output.w; ++ox) {
                                float value = conv_sum(previous, oc, oy, ox) + layer.biases[oc];
                                auto pos = static_cast<std::size_t>(oc) * out_plane + oy * layer.output.w + ox;
                                preactivation[s][index][pos] = value;
                                output[s][index][pos] = layer.leaky && value < 0 ? value * 0.1f : value;
                            }
                }
            }
        } else if (layer.kind == Kind::shortcut) {
            for (std::size_t s = 0; s < n; ++s) {
                const auto& previous = previous_of(s);
                const auto& branch = output[s][static_cast<std::size_t>(layer.sources.front())];
                auto& result = output[s][index];
                preactivation[s][index].resize(result.size());
                for (std::size_t p = 0; p < result.size(); ++p) {
                    float value = previous[p] + branch[p];
                    preactivation[s][index][p] = value;
                    result[p] = layer.leaky && value < 0 ? value * 0.1f : value;
                }
            }
        } else if (layer.kind == Kind::maxpool) {
            int pad = (layer.size - 1) / 2;
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            for (std::size_t s = 0; s < n; ++s) {
                const auto& previous = previous_of(s);
                auto& result = output[s][index];
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
            }
        } else if (layer.kind == Kind::route) {
            for (std::size_t s = 0; s < n; ++s) {
                auto& result = output[s][index];
                std::size_t offset = 0;
                for (int source : layer.sources) {
                    const auto& branch = output[s][static_cast<std::size_t>(source)];
                    std::copy(branch.begin(), branch.end(), result.begin() + static_cast<std::ptrdiff_t>(offset));
                    offset += branch.size();
                }
            }
        } else if (layer.kind == Kind::upsample) {
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            for (std::size_t s = 0; s < n; ++s) {
                const auto& previous = previous_of(s);
                auto& result = output[s][index];
                for (int c = 0; c < layer.input.c; ++c)
                    for (int y = 0; y < layer.output.h; ++y)
                        for (int x = 0; x < layer.output.w; ++x)
                            result[static_cast<std::size_t>(c) * out_plane + y * layer.output.w + x] =
                                previous[static_cast<std::size_t>(c) * in_plane + (y / layer.stride) * layer.input.w + x / layer.stride];
            }
        } else if (layer.kind == Kind::dropout) {
            // Inverted dropout: kept elements scaled by 1/(1-p) so the expectation matches
            // inference's unscaled passthrough; the per-element scale (0 or 1/(1-p)) is stashed in
            // `preactivation` (otherwise unused by this kind) so backward reuses the exact same
            // mask this forward pass drew, not a freshly re-rolled one.
            float keep_scale = 1.0f / (1.0f - layer.dropout_probability);
            std::bernoulli_distribution drop(layer.dropout_probability);
            for (std::size_t s = 0; s < n; ++s) {
                const auto& previous = previous_of(s);
                auto& result = output[s][index];
                preactivation[s][index].resize(result.size());
                for (std::size_t p = 0; p < result.size(); ++p) {
                    float scale = drop(dropout_rng_) ? 0.0f : keep_scale;
                    preactivation[s][index][p] = scale;
                    result[p] = previous[p] * scale;
                }
            }
        } else if (layer.kind == Kind::reorg) {
            // Space-to-depth rearrangement (see network.cpp's reorg_forward for the shared
            // convention); no learnable parameters, just a one-to-one position mapping.
            int in_plane = layer.input.h * layer.input.w;
            int out_w = layer.output.w, out_plane = layer.output.h * out_w;
            for (std::size_t s = 0; s < n; ++s) {
                const auto& previous = previous_of(s);
                auto& result = output[s][index];
                for (int ic = 0; ic < layer.input.c; ++ic)
                    for (int iy = 0; iy < layer.input.h; ++iy)
                        for (int ix = 0; ix < layer.input.w; ++ix) {
                            int oc = ic * layer.stride * layer.stride + (iy % layer.stride) * layer.stride + (ix % layer.stride);
                            result[static_cast<std::size_t>(oc) * out_plane + (iy / layer.stride) * out_w + (ix / layer.stride)] =
                                previous[static_cast<std::size_t>(ic) * in_plane + iy * layer.input.w + ix];
                        }
            }
        } else {
            for (std::size_t s = 0; s < n; ++s) output[s][index] = previous_of(s);
        }
    }

    // Shared with the CUDA trainer so both use identical target assignment and loss math; summed
    // (not averaged) across samples so train_main.cpp's epoch-level mean-loss-per-image stays
    // correct regardless of how many step() calls an epoch takes.
    float loss = 0;
    std::size_t positives = 0;
    for (std::size_t s = 0; s < n; ++s) {
        auto result = yolo_loss_and_gradient(network_, targets[s], output[s], gradient[s]);
        loss += result.loss;
        positives += result.positives;
    }

    // Reverse graph traversal: convolution weights/biases, and scale+bias (gamma/beta) for
    // batch_normalize layers, accumulate gradients summed over every sample in this minibatch
    // into this trainer's buffers; the SGD update (divided by `n`) is applied once below.
    for (std::size_t reverse = num_layers; reverse-- > 0;) {
        auto& layer = layers[reverse];
        if (layer.kind == Kind::yolo || layer.kind == Kind::region) {
            if (reverse)
                for (std::size_t s = 0; s < n; ++s)
                    for (std::size_t p = 0; p < gradient[s][reverse].size(); ++p) gradient[s][reverse - 1][p] += gradient[s][reverse][p];
        } else if (layer.kind == Kind::shortcut) {
            int source = layer.sources.front();
            for (std::size_t s = 0; s < n; ++s) {
                auto& grad = gradient[s][reverse];
                for (std::size_t p = 0; p < grad.size(); ++p) {
                    float g = grad[p] * (layer.leaky && preactivation[s][reverse][p] < 0 ? 0.1f : 1.0f);
                    if (reverse) gradient[s][reverse - 1][p] += g;
                    gradient[s][static_cast<std::size_t>(source)][p] += g;
                }
            }
        } else if (layer.kind == Kind::route) {
            for (std::size_t s = 0; s < n; ++s) {
                auto& grad = gradient[s][reverse];
                std::size_t offset = 0;
                for (int source : layer.sources) {
                    auto& branch = gradient[s][static_cast<std::size_t>(source)];
                    for (std::size_t p = 0; p < branch.size(); ++p) branch[p] += grad[offset + p];
                    offset += branch.size();
                }
            }
        } else if (layer.kind == Kind::upsample) {
            if (!reverse) continue;
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            for (std::size_t s = 0; s < n; ++s) {
                auto& grad = gradient[s][reverse];
                for (int c = 0; c < layer.input.c; ++c)
                    for (int y = 0; y < layer.output.h; ++y)
                        for (int x = 0; x < layer.output.w; ++x)
                            gradient[s][reverse - 1][static_cast<std::size_t>(c) * in_plane +
                                (y / layer.stride) * layer.input.w + x / layer.stride] +=
                                grad[static_cast<std::size_t>(c) * out_plane + y * layer.output.w + x];
            }
        } else if (layer.kind == Kind::maxpool) {
            if (!reverse) continue;
            int pad = (layer.size - 1) / 2;
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            for (std::size_t s = 0; s < n; ++s) {
                const auto& previous = output[s][reverse - 1];
                auto& grad = gradient[s][reverse];
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
                            gradient[s][reverse - 1][best_pos] += grad[static_cast<std::size_t>(c) * out_plane + oy * layer.output.w + ox];
                        }
            }
        } else if (layer.kind == Kind::dropout) {
            if (!reverse) continue;
            for (std::size_t s = 0; s < n; ++s) {
                auto& grad = gradient[s][reverse];
                for (std::size_t p = 0; p < grad.size(); ++p)
                    gradient[s][reverse - 1][p] += grad[p] * preactivation[s][reverse][p];
            }
        } else if (layer.kind == Kind::reorg) {
            if (!reverse) continue;
            int in_plane = layer.input.h * layer.input.w;
            int out_w = layer.output.w, out_plane = layer.output.h * out_w;
            for (std::size_t s = 0; s < n; ++s) {
                auto& grad = gradient[s][reverse];
                for (int ic = 0; ic < layer.input.c; ++ic)
                    for (int iy = 0; iy < layer.input.h; ++iy)
                        for (int ix = 0; ix < layer.input.w; ++ix) {
                            int oc = ic * layer.stride * layer.stride + (iy % layer.stride) * layer.stride + (ix % layer.stride);
                            gradient[s][reverse - 1][static_cast<std::size_t>(ic) * in_plane + iy * layer.input.w + ix] +=
                                grad[static_cast<std::size_t>(oc) * out_plane + (iy / layer.stride) * out_w + (ix / layer.stride)];
                        }
            }
        } else {
            // Convolution (and, for batch_normalize layers, true batch-norm) backward.
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            if (weight_grad_[reverse].empty()) weight_grad_[reverse].assign(layer.weights.size(), 0.0f);
            if (bias_grad_[reverse].empty()) bias_grad_[reverse].assign(layer.biases.size(), 0.0f);
            if (layer.batch_normalize && scale_grad_[reverse].empty())
                scale_grad_[reverse].assign(static_cast<std::size_t>(layer.filters), 0.0f);
            // dz[s] ends this block holding sample s's gradient w.r.t. its pre-BN conv sum.
            std::vector<std::vector<float>> dz(n, std::vector<float>(static_cast<std::size_t>(out_plane)));
            const double total = static_cast<double>(n) * out_plane;
            for (int oc = 0; oc < layer.output.c; ++oc) {
                if (layer.batch_normalize) {
                    float mean = batch_mean[reverse][oc];
                    float inv_std = 1.0f / std::sqrt(batch_var[reverse][oc] + bn_eps);
                    float gamma = layer.scales[oc];
                    double sum_dxhat = 0, sum_dxhat_xhat = 0;
                    std::vector<std::vector<float>> dxhat(n, std::vector<float>(static_cast<std::size_t>(out_plane)));
                    for (std::size_t s = 0; s < n; ++s)
                        for (int p = 0; p < out_plane; ++p) {
                            auto pos = static_cast<std::size_t>(oc) * out_plane + p;
                            float g = gradient[s][reverse][pos] * (layer.leaky && preactivation[s][reverse][pos] < 0 ? 0.1f : 1.0f);
                            float xhat = (raw[s][reverse][pos] - mean) * inv_std;
                            scale_grad_[reverse][oc] += g * xhat;
                            bias_grad_[reverse][oc] += g;
                            dxhat[s][static_cast<std::size_t>(p)] = g * gamma;
                            sum_dxhat += dxhat[s][static_cast<std::size_t>(p)];
                            sum_dxhat_xhat += dxhat[s][static_cast<std::size_t>(p)] * xhat;
                        }
                    for (std::size_t s = 0; s < n; ++s)
                        for (int p = 0; p < out_plane; ++p) {
                            auto pos = static_cast<std::size_t>(oc) * out_plane + p;
                            float xhat = (raw[s][reverse][pos] - mean) * inv_std;
                            dz[s][static_cast<std::size_t>(p)] = static_cast<float>(inv_std / total *
                                (total * dxhat[s][static_cast<std::size_t>(p)] - sum_dxhat - xhat * sum_dxhat_xhat));
                        }
                } else {
                    for (std::size_t s = 0; s < n; ++s)
                        for (int p = 0; p < out_plane; ++p) {
                            auto pos = static_cast<std::size_t>(oc) * out_plane + p;
                            float g = gradient[s][reverse][pos] * (layer.leaky && preactivation[s][reverse][pos] < 0 ? 0.1f : 1.0f);
                            bias_grad_[reverse][oc] += g;
                            dz[s][static_cast<std::size_t>(p)] = g;
                        }
                }
                for (std::size_t s = 0; s < n; ++s) {
                    const auto& previous = reverse ? output[s][reverse - 1] : inputs[s];
                    for (int oy = 0; oy < layer.output.h; ++oy)
                        for (int ox = 0; ox < layer.output.w; ++ox) {
                            float gsum = dz[s][static_cast<std::size_t>(oy * layer.output.w + ox)];
                            for (int ic = 0; ic < layer.input.c; ++ic)
                                for (int ky = 0; ky < layer.size; ++ky)
                                    for (int kx = 0; kx < layer.size; ++kx) {
                                        int iy = oy * layer.stride + ky - layer.padding;
                                        int ix = ox * layer.stride + kx - layer.padding;
                                        if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                                        std::size_t wi = ((static_cast<std::size_t>(oc) * layer.input.c + ic) * layer.size + ky) * layer.size + kx;
                                        auto source_pos = static_cast<std::size_t>(ic) * in_plane + iy * layer.input.w + ix;
                                        weight_grad_[reverse][wi] += gsum * previous[source_pos];
                                        if (reverse) gradient[s][reverse - 1][source_pos] += gsum * layer.weights[wi];
                                    }
                        }
                }
            }
        }
    }

    // SGD with classic momentum: velocity = momentum*velocity + this step's averaged gradient
    // (plus weight decay for weights), then param -= lr*velocity. Velocity buffers persist
    // across step() calls and start at zero, so the first call on a fresh Trainer is identical
    // to plain SGD regardless of `momentum_`.
    float lr = schedule_.at(iteration_);
    float inv_n = 1.0f / static_cast<float>(n);
    for (std::size_t index = 0; index < num_layers; ++index) {
        auto& layer = layers[index];
        if (layer.kind != Kind::convolutional && layer.kind != Kind::connected) continue;
        if (weight_velocity_[index].empty()) weight_velocity_[index].assign(layer.weights.size(), 0.0f);
        if (bias_velocity_[index].empty()) bias_velocity_[index].assign(layer.biases.size(), 0.0f);
        for (std::size_t p = 0; p < layer.weights.size(); ++p) {
            float& v = weight_velocity_[index][p];
            v = momentum_ * v + weight_grad_[index][p] * inv_n + weight_decay_ * layer.weights[p];
            layer.weights[p] -= lr * v;
            weight_grad_[index][p] = 0.0f;
        }
        for (std::size_t p = 0; p < layer.biases.size(); ++p) {
            float& v = bias_velocity_[index][p];
            v = momentum_ * v + bias_grad_[index][p] * inv_n;
            layer.biases[p] -= lr * v;
            bias_grad_[index][p] = 0.0f;
        }
        if (layer.batch_normalize) {
            if (scale_velocity_[index].empty()) scale_velocity_[index].assign(scale_grad_[index].size(), 0.0f);
            for (std::size_t p = 0; p < scale_grad_[index].size(); ++p) {
                float& v = scale_velocity_[index][p];
                v = momentum_ * v + scale_grad_[index][p] * inv_n;
                layer.scales[p] -= lr * v;
                scale_grad_[index][p] = 0.0f;
            }
        }
    }
    ++iteration_;
    if (!std::isfinite(loss)) throw std::runtime_error("non-finite training loss");
    return {loss, positives};
}

}  // namespace detector
