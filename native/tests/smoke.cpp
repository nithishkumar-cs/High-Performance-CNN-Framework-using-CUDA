#include "detector/image.hpp"
#include "detector/deconvolutional.hpp"
#include "detector/local.hpp"
#include "detector/network.hpp"
#include "detector/normalization.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef DETECTOR_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void floats(std::ofstream& file, std::size_t count, float value = 0.0f) {
    for (std::size_t i = 0; i < count; ++i) file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_weights(const std::filesystem::path& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot create smoke-test weights");
    const std::int32_t version[3] = {0, 2, 0};
    const std::uint64_t seen = 0;
    file.write(reinterpret_cast<const char*>(version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&seen), sizeof(seen));
    // First 1x1 conv: six channels, batch normalization, three input channels.
    floats(file, 6); floats(file, 6, 1.0f); floats(file, 6); floats(file, 6, 1.0f); floats(file, 18);
    // Second 1x1 conv: six outputs, six input channels.
    floats(file, 6); floats(file, 36);
    // Output conv: zero weights; only objectness and class logits are high.
    floats(file, 4); floats(file, 2, 5.0f); floats(file, 72);
    if (!file) throw std::runtime_error("failed writing smoke-test weights");
}

void write_shortcut_weights(const std::filesystem::path& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot create shortcut-test weights");
    const std::int32_t version[3] = {0, 2, 0};
    const std::uint64_t seen = 0;
    file.write(reinterpret_cast<const char*>(version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&seen), sizeof(seen));
    // Layer 0: identity-like 1x1 conv (no batch norm) broadcasting the single input channel to six.
    floats(file, 6, 0.0f); floats(file, 6, 1.0f);
    // Layer 1: constant 1x1 conv (no batch norm, zero weights) producing a known bias per channel.
    const float bias[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    file.write(reinterpret_cast<const char*>(bias), sizeof(bias));
    floats(file, 36, 0.0f);
    if (!file) throw std::runtime_error("failed writing shortcut-test weights");
}

}  // namespace

/** Exercises cfg parsing, inference, weight loading and CPU/CUDA parity on small graphs. */
int main() {
    try {
        const auto cfg = std::filesystem::path(__FILE__).parent_path() / "mini.cfg";
        auto net = detector::Network::from_cfg(cfg.string());
        expect(net.layers().size() == 7, "incorrect layer count");
        expect(net.layers()[1].output.h == 2 && net.layers()[1].output.w == 2, "maxpool shape");
        expect(net.layers()[4].output.c == 12 && net.layers()[4].output.h == 4, "route shape");
        expect(net.layers()[6].output.c == 6, "YOLO head shape");
        const auto weights = std::filesystem::current_path() / "detector_smoke.weights";
        write_weights(weights);
        net.load_weights(weights.string());
        std::filesystem::remove(weights);
        std::vector<float> input(net.input_shape().elements(), 0.25f);
        auto heads = net.forward(input, detector::Backend::cpu);
        expect(heads.size() == 1 && heads[0].values.size() == 96, "CPU head output");
        expect(std::abs(heads[0].values[4 * 16] - 5.0f) < 1e-5f, "objectness logit");
        auto boxes = detector::decode(net, heads, 0.5f, 0.45f);
        expect(boxes.size() == 16, "YOLO decode count");
        expect(boxes[0].score > 0.98f && boxes[0].score < 0.99f, "YOLO score");
        auto norm_net = detector::Network::from_cfg((std::filesystem::path(__FILE__).parent_path() / "l2norm.cfg").string());
        expect(norm_net.layers()[0].output.elements() == 3, "l2norm shape");
        auto norm_cpu = norm_net.forward({3.0f, 4.0f, 0.0f}, detector::Backend::cpu);
        const float norm_logits[3] = {0.6f, 0.8f, 0.0f};
        float norm_exp = std::exp(norm_logits[0]) + std::exp(norm_logits[1]) + std::exp(norm_logits[2]);
        for (int i = 0; i < 3; ++i)
            expect(std::abs(norm_cpu[0].values[static_cast<std::size_t>(i)] -
                            std::exp(norm_logits[i]) / norm_exp) < 1e-6f, "l2norm CPU output");
        auto norm_zero = norm_net.forward({0.0f, 0.0f, 0.0f}, detector::Backend::cpu);
        for (float probability : norm_zero[0].values)
            expect(std::abs(probability - 1.0f / 3.0f) < 1e-6f, "l2norm zero vector");
        auto average_net = detector::Network::from_cfg((std::filesystem::path(__FILE__).parent_path() / "avgpool.cfg").string());
        expect(average_net.layers()[0].output.c == 2 && average_net.layers()[0].output.h == 1 &&
               average_net.layers()[0].output.w == 1, "global avgpool shape");
        std::vector<float> average_input{1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 4.0f, 6.0f, 8.0f};
        auto average_cpu = average_net.forward(average_input, detector::Backend::cpu);
        float average_expected = std::exp(5.0f) / (std::exp(2.5f) + std::exp(5.0f));
        expect(std::abs(average_cpu[0].values[1] - average_expected) < 1e-6f,
               "global avgpool channel means");
        auto response_net = detector::Network::from_cfg(
            (std::filesystem::path(__FILE__).parent_path() / "normalization.cfg").string());
        const std::vector<float> response_input{1.0f, 2.0f, 3.0f, 4.0f};
        const float expected_response[4] = {1.0f / 2.5f, 2.0f / 3.4f, 3.0f / 4.9f, 4.0f / 4.5f};
        auto response_cpu = response_net.forward(response_input, detector::Backend::cpu);
        float response_sum = 0.0f;
        for (float value : expected_response) response_sum += std::exp(value);
        for (int i = 0; i < 4; ++i)
            expect(std::abs(response_cpu[0].values[static_cast<std::size_t>(i)] -
                std::exp(expected_response[i]) / response_sum) < 1e-6f,
                "normalization channel window");
        const float response_upstream[4] = {0.3f, -0.7f, 1.1f, 0.4f};
        float response_output[4], response_norms[4], response_gradient[4] = {};
        detector::normalization_forward(response_input.data(), response_output, response_norms,
            4, 1, 3, 0.1f, 1.0f, 2.0f);
        detector::normalization_backward(response_input.data(), response_output, response_norms,
            response_upstream, response_gradient, 4, 1, 3, 0.1f, 1.0f);
        for (int i = 0; i < 4; ++i) {
            auto perturbed = response_input;
            perturbed[static_cast<std::size_t>(i)] += 1e-3f;
            float plus[4];
            detector::normalization_forward(perturbed.data(), plus, nullptr, 4, 1, 3, 0.1f, 1.0f, 2.0f);
            perturbed[static_cast<std::size_t>(i)] -= 2e-3f;
            float minus[4];
            detector::normalization_forward(perturbed.data(), minus, nullptr, 4, 1, 3, 0.1f, 1.0f, 2.0f);
            float numeric = 0.0f;
            for (int j = 0; j < 4; ++j)
                numeric += response_upstream[j] * (plus[j] - minus[j]) / 2e-3f;
            expect(std::abs(numeric - response_gradient[i]) < 1e-4f,
                   "normalization input gradient");
        }
        auto local_net = detector::Network::from_cfg(
            (std::filesystem::path(__FILE__).parent_path() / "local.cfg").string());
        local_net.initialize_random(5);
        auto& local_layer = local_net.layer(0);
        expect(local_layer.weights.size() == 4 && local_layer.biases.size() == 4,
               "local layer keeps one filter and bias per output location");
        local_layer.weights = {2.0f, -1.0f, 0.5f, 3.0f};
        local_layer.biases = {0.5f, 0.0f, -1.0f, 0.25f};
        auto& local_dense = local_net.layer(1);
        std::fill(local_dense.weights.begin(), local_dense.weights.end(), 0.0f);
        std::fill(local_dense.biases.begin(), local_dense.biases.end(), 0.0f);
        for (int i = 0; i < 4; ++i) local_dense.weights[static_cast<std::size_t>(i * 4 + i)] = 1.0f;
        const std::vector<float> local_input{1.0f, 2.0f, 3.0f, 4.0f};
        const float local_logits[4] = {2.5f, -2.0f, 0.5f, 12.25f};
        float local_exp = 0.0f;
        for (float logit : local_logits) local_exp += std::exp(logit);
        auto local_cpu = local_net.forward(local_input, detector::Backend::cpu);
        for (int i = 0; i < 4; ++i)
            expect(std::abs(local_cpu[0].values[static_cast<std::size_t>(i)] -
                            std::exp(local_logits[i]) / local_exp) < 1e-6f, "local unshared weights");
#ifdef DETECTOR_HAS_CUDA
        auto local_gpu = local_net.forward(local_input, detector::Backend::cuda);
        for (int i = 0; i < 4; ++i)
            expect(std::abs(local_gpu[0].values[static_cast<std::size_t>(i)] -
                            local_cpu[0].values[static_cast<std::size_t>(i)]) < 1e-5f,
                   "local CPU/CUDA parity");
#endif
        {
            detector::Layer probe;
            probe.kind = detector::Kind::local;
            probe.input = {2, 3, 3}; probe.output = {2, 2, 2};
            probe.filters = 2; probe.size = 2; probe.stride = 1; probe.padding = 0;
            probe.logistic = true;
            probe.weights.resize(4 * 2 * 2 * 2 * 2); probe.biases.resize(8);
            for (std::size_t i = 0; i < probe.weights.size(); ++i)
                probe.weights[i] = 0.3f * std::sin(0.7f * static_cast<float>(i));
            for (std::size_t i = 0; i < probe.biases.size(); ++i)
                probe.biases[i] = 0.1f * static_cast<float>(i) - 0.3f;
            std::vector<float> x(18), up{0.3f, -0.7f, 1.1f, 0.4f, -0.2f, 0.9f, 0.5f, -1.0f};
            for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::cos(0.9f * static_cast<float>(i));
            auto loss = [&](const detector::Layer& layer, const std::vector<float>& in) {
                std::vector<float> out(8);
                detector::local_forward(layer, in.data(), out.data());
                float total = 0.0f;
                for (int i = 0; i < 8; ++i) total += up[static_cast<std::size_t>(i)] * out[static_cast<std::size_t>(i)];
                return total;
            };
            std::vector<float> out(8), gx(18), gw(probe.weights.size()), gb(8);
            detector::local_forward(probe, x.data(), out.data());
            detector::local_backward(probe, x.data(), out.data(), up.data(), gx.data(), gw.data(), gb.data());
            const float eps = 1e-3f;
            for (std::size_t i = 0; i < x.size(); ++i) {
                auto hi = x, lo = x; hi[i] += eps; lo[i] -= eps;
                expect(std::abs((loss(probe, hi) - loss(probe, lo)) / (2 * eps) - gx[i]) < 2e-3f,
                       "local input gradient");
            }
            for (std::size_t i = 0; i < probe.weights.size(); ++i) {
                auto hi = probe, lo = probe; hi.weights[i] += eps; lo.weights[i] -= eps;
                expect(std::abs((loss(hi, x) - loss(lo, x)) / (2 * eps) - gw[i]) < 2e-3f,
                       "local weight gradient");
            }
            for (std::size_t i = 0; i < probe.biases.size(); ++i) {
                auto hi = probe, lo = probe; hi.biases[i] += eps; lo.biases[i] -= eps;
                expect(std::abs((loss(hi, x) - loss(lo, x)) / (2 * eps) - gb[i]) < 2e-3f,
                       "local bias gradient");
            }
        }
        {
            detector::Layer probe;
            probe.kind = detector::Kind::deconvolutional;
            probe.input = {1, 2, 2}; probe.output = {1, 3, 3};
            probe.filters = 1; probe.size = 2; probe.stride = 1; probe.padding = 0;
            probe.weights = {1.0f, 10.0f, 100.0f, 1000.0f};
            probe.biases = {0.5f};
            const std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f};
            const float expected[9] = {1, 12, 20, 103, 1234, 2040, 300, 3400, 4000};
            std::vector<float> out(9);
            detector::deconvolutional_forward(probe, x.data(), out.data());
            for (int i = 0; i < 9; ++i) expect(std::abs(out[static_cast<std::size_t>(i)] - expected[i] - 0.5f) < 1e-3f,
                                               "deconvolutional transposed-convolution output");
            auto strided = detector::Network::from_cfg(
                (std::filesystem::path(__FILE__).parent_path() / "deconv-train.cfg").string());
            expect(strided.layers()[0].output.h == 3 && strided.layers()[1].output.h == 4,
                   "deconvolutional output shape");

            probe.stride = 2; probe.padding = 1; probe.input = {1, 2, 2}; probe.output = {1, 2, 2};
            probe.logistic = true; probe.size = 2;
            probe.biases = {0.1f};
            probe.weights = {0.3f, -0.5f, 0.7f, 0.2f};
            const std::vector<float> up{0.3f, -0.7f, 1.1f, 0.4f};
            auto loss = [&](const detector::Layer& layer, const std::vector<float>& in) {
                std::vector<float> o(4);
                detector::deconvolutional_forward(layer, in.data(), o.data());
                float total = 0.0f;
                for (int i = 0; i < 4; ++i) total += up[static_cast<std::size_t>(i)] * o[static_cast<std::size_t>(i)];
                return total;
            };
            std::vector<float> o(4), gx(4), gw(4), gb(1);
            detector::deconvolutional_forward(probe, x.data(), o.data());
            detector::deconvolutional_backward(probe, x.data(), o.data(), up.data(), gx.data(), gw.data(), gb.data());
            const float eps = 1e-3f;
            for (std::size_t i = 0; i < 4; ++i) {
                auto hi = x, lo = x; hi[i] += eps; lo[i] -= eps;
                expect(std::abs((loss(probe, hi) - loss(probe, lo)) / (2 * eps) - gx[i]) < 2e-3f,
                       "deconvolutional input gradient");
                auto wh = probe, wl = probe; wh.weights[i] += eps; wl.weights[i] -= eps;
                expect(std::abs((loss(wh, x) - loss(wl, x)) / (2 * eps) - gw[i]) < 2e-3f,
                       "deconvolutional weight gradient");
            }
            auto parity = detector::Network::from_cfg(
                (std::filesystem::path(__FILE__).parent_path() / "deconv-classifier.cfg").string());
            parity.initialize_random(9);
            std::vector<float> parity_input(parity.input_shape().elements());
            for (std::size_t i = 0; i < parity_input.size(); ++i) parity_input[i] = 0.2f * static_cast<float>(i % 5) - 0.3f;
            auto parity_cpu = parity.forward(parity_input, detector::Backend::cpu);
#ifdef DETECTOR_HAS_CUDA
            auto parity_gpu = parity.forward(parity_input, detector::Backend::cuda);
            for (std::size_t i = 0; i < parity_cpu[0].values.size(); ++i)
                expect(std::abs(parity_gpu[0].values[i] - parity_cpu[0].values[i]) < 1e-5f,
                       "deconvolutional CPU/CUDA parity");
#endif
            auto bh = probe, bl = probe; bh.biases[0] += eps; bl.biases[0] -= eps;
            expect(std::abs((loss(bh, x) - loss(bl, x)) / (2 * eps) - gb[0]) < 2e-3f,
                   "deconvolutional bias gradient");
        }
        std::vector<detector::Detection> duplicates = {
            {0, 0.9f, 0, 0, 2, 2}, {0, 0.8f, 0, 0, 2, 2}, {1, 0.7f, 0, 0, 2, 2}
        };
        expect(detector::non_max_suppression(duplicates, 0.45f).size() == 2, "class-aware NMS");
        detector::Image image{2, 1, {255, 0, 0, 0, 255, 0}};
        auto transform = detector::letterbox(image, {3, 4, 4});
        expect(transform.resized_width == 4 && transform.resized_height == 2 && transform.top == 1, "letterbox geometry");
        std::vector<detector::Detection> mapped = {{0, 1, 0, 1, 4, 3}};
        detector::undo_letterbox(mapped, transform);
        expect(std::abs(mapped[0].x2 - 2) < 1e-5f && std::abs(mapped[0].y2 - 1) < 1e-5f, "unletterbox geometry");
#ifdef DETECTOR_HAS_CUDA
        int devices = 0;
        if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0) {
            auto gpu = net.forward(input, detector::Backend::cuda);
            expect(gpu.size() == heads.size(), "CUDA head count");
            for (std::size_t i = 0; i < heads[0].values.size(); ++i)
                expect(std::abs(gpu[0].values[i] - heads[0].values[i]) < 1e-4f, "CPU/CUDA parity");
            auto norm_gpu = norm_net.forward({3.0f, 4.0f, 0.0f}, detector::Backend::cuda);
            for (std::size_t i = 0; i < norm_cpu[0].values.size(); ++i)
                expect(std::abs(norm_gpu[0].values[i] - norm_cpu[0].values[i]) < 1e-5f,
                       "l2norm CPU/CUDA parity");
            auto average_gpu = average_net.forward(average_input, detector::Backend::cuda);
            for (std::size_t i = 0; i < average_cpu[0].values.size(); ++i)
                expect(std::abs(average_gpu[0].values[i] - average_cpu[0].values[i]) < 1e-5f,
                        "global avgpool CPU/CUDA parity");
            auto response_gpu = response_net.forward(response_input, detector::Backend::cuda);
            for (std::size_t i = 0; i < response_cpu[0].values.size(); ++i)
                expect(std::abs(response_gpu[0].values[i] - response_cpu[0].values[i]) < 1e-5f,
                       "normalization CPU/CUDA parity");
        }
#endif

        // Shortcut (residual add) layer: output must equal layer1's constant bias plus layer0's
        // broadcast input, verifying it reads the correct "from" branch rather than just passing through.
        const auto shortcut_cfg = std::filesystem::path(__FILE__).parent_path() / "shortcut.cfg";
        auto shortcut_net = detector::Network::from_cfg(shortcut_cfg.string());
        expect(shortcut_net.layers().size() == 4 && shortcut_net.layers()[2].output.c == 6, "shortcut layer shape");
        const auto shortcut_weights = std::filesystem::current_path() / "shortcut_smoke.weights";
        write_shortcut_weights(shortcut_weights);
        shortcut_net.load_weights(shortcut_weights.string());
        std::filesystem::remove(shortcut_weights);
        std::vector<float> pixel = {0.25f};
        auto shortcut_heads = shortcut_net.forward(pixel, detector::Backend::cpu);
        expect(shortcut_heads.size() == 1 && shortcut_heads[0].values.size() == 6, "shortcut head shape");
        for (int c = 0; c < 6; ++c)
            expect(std::abs(shortcut_heads[0].values[static_cast<std::size_t>(c)] - (c + 1 + 0.25f)) < 1e-5f,
                   "shortcut residual add");
#ifdef DETECTOR_HAS_CUDA
        if (devices > 0) {
            auto shortcut_gpu = shortcut_net.forward(pixel, detector::Backend::cuda);
            expect(shortcut_gpu.size() == shortcut_heads.size(), "shortcut CUDA head count");
            for (std::size_t i = 0; i < shortcut_heads[0].values.size(); ++i)
                expect(std::abs(shortcut_gpu[0].values[i] - shortcut_heads[0].values[i]) < 1e-4f,
                       "shortcut CPU/CUDA parity");
        }
#endif

        // Partial (pretrained-backbone) load: Darknet's `partial N` transfer loads only the first
        // N layers from a weights file sized for a larger network and leaves the rest random-init.
        auto partial_net = detector::Network::from_cfg(cfg.string());
        partial_net.initialize_random(7);
        std::vector<float> layer2_before = partial_net.layer(2).weights;
        const auto partial_weights = std::filesystem::current_path() / "detector_partial_smoke.weights";
        write_weights(partial_weights);  // full 3-conv-block file; layer_limit=1 should read only block 0.
        partial_net.load_weights(partial_weights.string(), 1);
        std::filesystem::remove(partial_weights);
        expect(partial_net.layer(0).biases.size() == 6 && partial_net.layer(0).scales[0] == 1.0f,
               "partial load applied layer 0 from file");
        expect(partial_net.layer(2).weights == layer2_before,
               "partial load left layer 2 at its pre-call (random) value");
        const auto strict_weights = std::filesystem::current_path() / "detector_strict_smoke.weights";
        write_weights(strict_weights);
        // Append one stray byte so a full (non-partial) load must reject it as trailing data.
        { std::ofstream append(strict_weights, std::ios::binary | std::ios::app); char pad = 0; append.write(&pad, 1); }
        bool full_load_threw = false;
        try {
            detector::Network::from_cfg(cfg.string()).load_weights(strict_weights.string());
        } catch (const std::runtime_error&) { full_load_threw = true; }
        std::filesystem::remove(strict_weights);
        expect(full_load_threw, "full load must reject trailing data that a partial load would ignore");

        // Network::resize (Darknet-style multiscale training): recomputes every layer's shape for
        // a new input size while leaving every learned parameter untouched. mini.cfg's one maxpool
        // (stride 2) and one upsample (stride 2) exactly cancel, so doubling 4x4 to 8x8 keeps the
        // route layer's two branches' spatial shapes matched, just like at the original size.
        auto resize_net = detector::Network::from_cfg(cfg.string());
        resize_net.initialize_random(11);
        auto layer0_weights_before = resize_net.layer(0).weights;
        resize_net.resize(8, 8);
        expect(resize_net.input_shape().h == 8 && resize_net.input_shape().w == 8, "resize: input shape");
        expect(resize_net.layers()[0].output.h == 8 && resize_net.layers()[0].output.w == 8, "resize: conv0 shape");
        expect(resize_net.layers()[1].output.h == 4 && resize_net.layers()[1].output.w == 4, "resize: maxpool shape");
        expect(resize_net.layers()[4].output.c == 12 && resize_net.layers()[4].output.h == 8, "resize: route shape");
        expect(resize_net.layer(0).weights == layer0_weights_before, "resize: weights untouched");
        std::vector<float> resized_input(resize_net.input_shape().elements(), 0.25f);
        auto resized_heads = resize_net.forward(resized_input, detector::Backend::cpu);
        expect(resized_heads.size() == 1 && resized_heads[0].values.size() == 6 * 8 * 8, "resize: forward output shape");

        // Reorg (space-to-depth) layer: input channel `ic`'s pixel (iy,ix) is value ic*100+iy*4+ix
        // (c=3,h=4,w=4); stride=2 must route it to output channel ic*4+(iy%2)*2+(ix%2), spatial
        // (iy/2,ix/2) - these spot checks are independently hand-derived from that formula, not
        // just re-calling it, plus a permutation invariant (same multiset of values) over the rest.
        const auto reorg_cfg = std::filesystem::path(__FILE__).parent_path() / "reorg.cfg";
        auto reorg_net = detector::Network::from_cfg(reorg_cfg.string());
        expect(reorg_net.layers()[0].output.c == 12 && reorg_net.layers()[0].output.h == 2, "reorg layer shape");
        std::vector<float> reorg_input(48);
        for (int ic = 0; ic < 3; ++ic)
            for (int iy = 0; iy < 4; ++iy)
                for (int ix = 0; ix < 4; ++ix)
                    reorg_input[static_cast<std::size_t>(ic * 16 + iy * 4 + ix)] = static_cast<float>(ic * 100 + iy * 4 + ix);
        auto reorg_heads = reorg_net.forward(reorg_input, detector::Backend::cpu);
        expect(reorg_heads.size() == 1 && reorg_heads[0].values.size() == 48, "reorg head shape");
        auto check_reorg = [&](int ic, int iy, int ix) {
            int oc = ic * 4 + (iy % 2) * 2 + (ix % 2), sp = (iy / 2) * 2 + ix / 2;
            float expected = static_cast<float>(ic * 100 + iy * 4 + ix);
            expect(std::abs(reorg_heads[0].values[static_cast<std::size_t>(oc * 4 + sp)] - expected) < 1e-6f,
                   "reorg space-to-depth mapping");
        };
        check_reorg(0, 0, 0); check_reorg(0, 3, 3); check_reorg(1, 2, 1); check_reorg(2, 1, 3); check_reorg(2, 3, 0);
        float reorg_sum_in = 0, reorg_sum_out = 0;
        for (float v : reorg_input) reorg_sum_in += v;
        for (float v : reorg_heads[0].values) reorg_sum_out += v;
        expect(std::abs(reorg_sum_in - reorg_sum_out) < 1e-3f, "reorg must be a pure permutation (sum-preserving)");
#ifdef DETECTOR_HAS_CUDA
        if (devices > 0) {
            auto reorg_gpu = reorg_net.forward(reorg_input, detector::Backend::cuda);
            for (std::size_t i = 0; i < reorg_heads[0].values.size(); ++i)
                expect(std::abs(reorg_gpu[0].values[i] - reorg_heads[0].values[i]) < 1e-4f, "reorg CPU/CUDA parity");
        }
#endif

        // Route layer spatial-shape validation must compare branches to each other, not to the
        // running sequential shape: a standalone passthrough route (Darknet's YOLOv2-style
        // `[route] layers=-9` pattern, jumping back to an earlier layer at a different spatial
        // resolution than the layer immediately before the route) is standard and must not be
        // rejected. route-passthrough.cfg's lone route (layers=-2) resolves to the pre-maxpool
        // conv (4x4), which differs from the route's immediately-preceding maxpool output (2x2).
        const auto route_cfg = std::filesystem::path(__FILE__).parent_path() / "route-passthrough.cfg";
        auto route_net = detector::Network::from_cfg(route_cfg.string());
        expect(route_net.layers()[2].output.c == 2 && route_net.layers()[2].output.h == 4 &&
                   route_net.layers()[2].output.w == 4,
               "passthrough route resolves to its source's shape, not the preceding layer's");

        // Dropout layer: a no-op at inference (Darknet only applies it during training).
        const auto dropout_cfg = std::filesystem::path(__FILE__).parent_path() / "dropout.cfg";
        auto dropout_net = detector::Network::from_cfg(dropout_cfg.string());
        std::vector<float> dropout_input{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
        auto dropout_heads = dropout_net.forward(dropout_input, detector::Backend::cpu);
        expect(dropout_heads[0].values.size() == 6, "dropout head shape");
        for (std::size_t i = 0; i < dropout_input.size(); ++i)
            expect(std::abs(dropout_heads[0].values[i] - dropout_input[i]) < 1e-6f, "dropout must be a no-op at inference");
#ifdef DETECTOR_HAS_CUDA
        if (devices > 0) {
            auto dropout_gpu = dropout_net.forward(dropout_input, detector::Backend::cuda);
            for (std::size_t i = 0; i < dropout_input.size(); ++i)
                expect(std::abs(dropout_gpu[0].values[i] - dropout_input[i]) < 1e-4f, "dropout CUDA must also be a no-op at inference");
        }
#endif

        // Region (YOLOv2-style) head: unlike yolo's independent per-class sigmoid, class score is
        // objectness * softmax(class logits). anchor 0/cell 0 gets a confident box (high
        // objectness, class0 logit=2 vs class1 logit=0); every other (anchor, cell) stays at very
        // negative objectness so only this one cell's class0 clears the 0.5 threshold (class1's
        // softmax share is ~0.12, below it).
        const auto region_cfg = std::filesystem::path(__FILE__).parent_path() / "region.cfg";
        auto region_net = detector::Network::from_cfg(region_cfg.string());
        expect(region_net.layers()[0].output.c == 14, "region layer shape");
        std::vector<float> region_raw(224, -10.0f);
        auto region_pos = [](int anchor, int attr, int cell) { return static_cast<std::size_t>((anchor * 7 + attr) * 16 + cell); };
        region_raw[region_pos(0, 0, 0)] = 0.0f;   // x -> sigmoid(0)=0.5
        region_raw[region_pos(0, 1, 0)] = 0.0f;   // y -> sigmoid(0)=0.5
        region_raw[region_pos(0, 2, 0)] = 0.0f;   // w -> exp(0)*anchor_w=2
        region_raw[region_pos(0, 3, 0)] = 0.0f;   // h -> exp(0)*anchor_h=2
        region_raw[region_pos(0, 4, 0)] = 10.0f;  // objectness -> sigmoid(10) ~ 0.99995
        region_raw[region_pos(0, 5, 0)] = 2.0f;   // class0 logit
        region_raw[region_pos(0, 6, 0)] = 0.0f;   // class1 logit
        auto region_heads = region_net.forward(region_raw, detector::Backend::cpu);
        expect(region_heads.size() == 1 && region_heads[0].values.size() == 224, "region head shape");
        auto region_boxes = detector::decode(region_net, region_heads, 0.5f, 0.45f);
        expect(region_boxes.size() == 1, "region decode: only class0 at (anchor0, cell0) should clear threshold");
        expect(region_boxes[0].class_id == 0, "region decode class id");
        expect(region_boxes[0].score > 0.87f && region_boxes[0].score < 0.89f, "region softmax*objectness score");
        expect(std::abs(region_boxes[0].x1 - (-0.5f)) < 1e-4f && std::abs(region_boxes[0].x2 - 1.5f) < 1e-4f,
               "region decoded box geometry");
#ifdef DETECTOR_HAS_CUDA
        if (devices > 0) {
            auto region_gpu = region_net.forward(region_raw, detector::Backend::cuda);
            for (std::size_t i = 0; i < region_heads[0].values.size(); ++i)
                expect(std::abs(region_gpu[0].values[i] - region_heads[0].values[i]) < 1e-4f, "region CPU/CUDA parity");
        }
#endif

        std::cout << "detector smoke tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "smoke test failed: " << error.what() << '\n';
        return 1;
    }
}
