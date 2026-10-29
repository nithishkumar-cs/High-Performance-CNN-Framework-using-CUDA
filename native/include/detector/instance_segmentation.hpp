#pragma once

#include "detector/image.hpp"
#include "detector/network.hpp"
#include <cstdint>
#include <vector>

namespace detector {

/** One detection's mask in its cropped original-image rectangle. */
struct InstanceMask {
    Detection detection;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    // Row-major binary mask for this detection's crop, not a full-image allocation.
    std::vector<std::uint8_t> pixels;
};

/** Runs a binary mask head independently on each original-image detection crop.
 * Returns one cropped mask per valid box; coordinates remain in original-image pixels.
 */
std::vector<InstanceMask> predict_instance_masks(const Network& mask_network, const Image& image,
    const std::vector<Detection>& detections, Backend backend, float mask_threshold = 0.5f);

}  // namespace detector
