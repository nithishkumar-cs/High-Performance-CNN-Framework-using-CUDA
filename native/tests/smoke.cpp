#include "detector/image.hpp"
#include "detector/network.hpp"
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

        std::cout << "detector smoke tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "smoke test failed: " << error.what() << '\n';
        return 1;
    }
}
