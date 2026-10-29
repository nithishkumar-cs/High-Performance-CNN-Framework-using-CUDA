#include "detector/instance_segmentation.hpp"
#include "detector/cuda_session.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace detector {

/** Applies the mask network to every box, reusing CUDA buffers, and maps logits onto source crops. */
std::vector<InstanceMask> predict_instance_masks(const Network& mask_network, const Image& image,
    const std::vector<Detection>& detections, Backend backend, float mask_threshold) {
    if (!std::isfinite(mask_threshold) || mask_threshold < 0.0f || mask_threshold > 1.0f)
        throw std::runtime_error("mask threshold must be in [0,1]");
    if (image.width <= 0 || image.height <= 0 ||
        image.rgb.size() != static_cast<std::size_t>(image.width) * image.height * 3)
        throw std::runtime_error("invalid instance source image");
    if (mask_network.input_shape().c != 3 || mask_network.layers().empty() ||
        mask_network.layers().back().output.c != 1)
        throw std::runtime_error("instance mask network needs RGB input and one output channel");
    const Shape input_shape = mask_network.input_shape();
    const Shape output_shape = mask_network.layers().back().output;
#ifdef DETECTOR_HAS_CUDA
    std::unique_ptr<CudaSession> cuda_session;
    if (backend == Backend::cuda && !detections.empty())
        cuda_session = std::make_unique<CudaSession>(mask_network);
#endif
    std::vector<InstanceMask> masks;
    masks.reserve(detections.size());
    for (const auto& detection : detections) {
        if (!std::isfinite(detection.x1) || !std::isfinite(detection.y1) ||
            !std::isfinite(detection.x2) || !std::isfinite(detection.y2))
            throw std::runtime_error("non-finite instance box");
        const int x0 = static_cast<int>(std::floor(std::clamp(detection.x1, 0.0f, static_cast<float>(image.width))));
        const int y0 = static_cast<int>(std::floor(std::clamp(detection.y1, 0.0f, static_cast<float>(image.height))));
        const int x1 = static_cast<int>(std::ceil(std::clamp(detection.x2, 0.0f, static_cast<float>(image.width))));
        const int y1 = static_cast<int>(std::ceil(std::clamp(detection.y2, 0.0f, static_cast<float>(image.height))));
        if (x1 <= x0 || y1 <= y0) continue;
        Image roi = crop(image, x0, y0, x1 - x0, y1 - y0);
        Letterbox transform = letterbox(roi, input_shape);
        std::vector<Head> heads;
#ifdef DETECTOR_HAS_CUDA
        if (cuda_session) heads = cuda_session->forward(transform.chw);
        else
#endif
            heads = mask_network.forward(transform.chw, backend);
        if (heads.empty() || heads.back().values.size() != output_shape.elements())
            throw std::runtime_error("instance mask head output missing");
        InstanceMask mask;
        mask.detection = detection;
        mask.x = x0;
        mask.y = y0;
        mask.width = roi.width;
        mask.height = roi.height;
        mask.pixels.resize(static_cast<std::size_t>(roi.width) * roi.height);
        for (int y = 0; y < roi.height; ++y) {
            const float network_y = transform.top +
                (y + 0.5f) * transform.resized_height / roi.height;
            const int my = std::clamp(static_cast<int>(network_y * output_shape.h / input_shape.h),
                                      0, output_shape.h - 1);
            for (int x = 0; x < roi.width; ++x) {
                const float network_x = transform.left +
                    (x + 0.5f) * transform.resized_width / roi.width;
                const int mx = std::clamp(static_cast<int>(network_x * output_shape.w / input_shape.w),
                                          0, output_shape.w - 1);
                const float logit = heads.back().values[static_cast<std::size_t>(my) * output_shape.w + mx];
                const float probability = 1.0f / (1.0f + std::exp(-logit));
                mask.pixels[static_cast<std::size_t>(y) * roi.width + x] =
                    probability >= mask_threshold ? 255 : 0;
            }
        }
        masks.push_back(std::move(mask));
    }
    return masks;
}

}  // namespace detector
