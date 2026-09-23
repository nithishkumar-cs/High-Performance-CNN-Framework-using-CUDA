#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace yolo_cuda {

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    std::int32_t class_id;
};

struct PostprocessConfig {
    std::int32_t class_count = 80;
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.45f;
    std::int32_t image_width = 640;
    std::int32_t image_height = 640;
};

// Decode rows laid out as [cx, cy, width, height, objectness, class scores...].
// The input pointer must refer to device memory. Returned boxes are in pixel
// coordinates and have already passed confidence filtering and NMS.
std::vector<Detection> decode_and_nms(
    const float* device_output,
    std::size_t row_count,
    const PostprocessConfig& config);

}  // namespace yolo_cuda
