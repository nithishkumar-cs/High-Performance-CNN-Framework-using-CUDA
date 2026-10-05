#pragma once

#include "detector/network.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace detector {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
};

struct Letterbox {
    std::vector<float> chw;
    int original_width = 0;
    int original_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int left = 0;
    int top = 0;
};

Image read_image(const std::string& path);
Letterbox letterbox(const Image& image, Shape target);
void undo_letterbox(std::vector<Detection>& boxes, const Letterbox& transform);

}  // namespace detector
