#include "detector/instance_segmentation.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/** Verifies per-box crop coordinates, mask projection, and CPU/CUDA mask inference parity. */
int main() {
    try {
        const auto cfg = std::filesystem::path(__FILE__).parent_path() / "instance-mask.cfg";
        auto network = detector::Network::from_cfg(cfg.string());
        network.initialize_random(19);
        auto& head = network.layer(0);
        head.weights = {10.0f, 0.0f, 0.0f};
        head.biases = {-5.0f};
        detector::Image image;
        image.width = 4;
        image.height = 4;
        image.rgb.assign(4 * 4 * 3, 0);
        for (int y = 0; y < 4; ++y)
            for (int x = 2; x < 4; ++x)
                image.rgb[(y * 4 + x) * 3] = 255;
        const std::vector<detector::Detection> boxes{
            {0, 0.9f, 0.0f, 0.0f, 2.0f, 4.0f},
            {0, 0.8f, 2.0f, 0.0f, 5.0f, 4.0f}};
        auto masks = detector::predict_instance_masks(network, image, boxes, detector::Backend::cpu);
        if (masks.size() != 2 || masks[0].x != 0 || masks[1].x != 2 ||
            masks[0].width != 2 || masks[1].width != 2 ||
            masks[0].height != 4 || masks[1].height != 4 ||
            !std::all_of(masks[0].pixels.begin(), masks[0].pixels.end(),
                         [](std::uint8_t value) { return value == 0; }) ||
            !std::all_of(masks[1].pixels.begin(), masks[1].pixels.end(),
                         [](std::uint8_t value) { return value == 255; }))
            throw std::runtime_error("instance crop or mask projection mismatch");
#ifdef DETECTOR_HAS_CUDA
        auto cuda = detector::predict_instance_masks(network, image, boxes, detector::Backend::cuda);
        if (cuda.size() != masks.size() || cuda[0].pixels != masks[0].pixels ||
            cuda[1].pixels != masks[1].pixels)
            throw std::runtime_error("instance mask CPU/CUDA inference mismatch");
#endif
        try {
            detector::predict_instance_masks(network, image, boxes, detector::Backend::cpu, -0.1f);
            throw std::runtime_error("invalid mask threshold accepted");
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()) != "mask threshold must be in [0,1]") throw;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
