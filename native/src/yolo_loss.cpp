#include "detector/yolo_loss.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

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

// Standard IoU of two centered boxes (same units for both), used by the ignore_thresh near-miss
// check; unlike anchor_iou above this accounts for box position, not just width/height.
float box_iou(float cx1, float cy1, float w1, float h1, float cx2, float cy2, float w2, float h2) {
    float ix = std::max(0.0f, std::min(cx1 + w1 / 2, cx2 + w2 / 2) - std::max(cx1 - w1 / 2, cx2 - w2 / 2));
    float iy = std::max(0.0f, std::min(cy1 + h1 / 2, cy2 + h2 / 2) - std::max(cy1 - h1 / 2, cy2 - h2 / 2));
    float intersection = ix * iy;
    float denom = w1 * h1 + w2 * h2 - intersection;
    return denom > 0 ? intersection / denom : 0.0f;
}

struct Assignment {
    int anchor = 0;
    int cell = 0;
    int class_id = 0;
    float offset_x = 0;
    float offset_y = 0;
    float log_width = 0;
    float log_height = 0;
    // Darknet's (2 - truth.w*truth.h) coordinate-loss scale (truth fractions of the network
    // input); up-weights small boxes relative to large ones. Applied to x/y/w/h only.
    float box_scale = 1.0f;
};

}  // namespace

// Compute the per-image YOLO loss and raw-head gradients for both training backends.
YoloLossResult yolo_loss_and_gradient(const Network& network, const std::vector<Target>& targets,
                                       const std::vector<std::vector<float>>& output,
                                       std::vector<std::vector<float>>& gradient) {
    const auto& layers = network.layers();
    Shape input = network.input_shape();

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
                float overlap = anchor_iou(aw, ah, target.width * input.w, target.height * input.h);
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
            std::log(target.width * input.w / aw),
            std::log(target.height * input.h / ah),
            2.0f - target.width * target.height});
    }

    float loss = 0;
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
                if (layer.ignore_thresh < 1.0f && !targets.empty()) {
                    // Decode this (anchor, cell)'s predicted box (same math as network.cpp's
                    // decode()) and skip the negative push if it already overlaps some ground
                    // truth by >= ignore_thresh: it is a near-miss the model should keep refining,
                    // not be told is definitely background (Darknet's ignore_thresh rule).
                    int x = cell % layer.output.w, y = cell / layer.output.w;
                    auto [aw, ah] = layer.anchors[static_cast<std::size_t>(layer.mask[static_cast<std::size_t>(anchor)])];
                    float px = (sigmoid(output[index][pos(anchor, 0, cell)]) + x) * input.w / layer.output.w;
                    float py = (sigmoid(output[index][pos(anchor, 1, cell)]) + y) * input.h / layer.output.h;
                    float pw = std::exp(std::clamp(output[index][pos(anchor, 2, cell)], -20.0f, 20.0f)) * aw;
                    float ph = std::exp(std::clamp(output[index][pos(anchor, 3, cell)], -20.0f, 20.0f)) * ah;
                    float best = 0;
                    for (const auto& target : targets)
                        best = std::max(best, box_iou(px, py, pw, ph, target.cx * input.w, target.cy * input.h,
                                                        target.width * input.w, target.height * input.h));
                    if (best >= layer.ignore_thresh) continue;
                }
                auto p = pos(anchor, 4, cell);
                loss += binary_cross_entropy(output[index][p], 0.0f);
                gradient[index][p] += sigmoid(output[index][p]);
            }
        for (const auto& match : assignments[index]) {
            auto obj = pos(match.anchor, 4, match.cell);
            loss += binary_cross_entropy(output[index][obj], 1.0f);
            gradient[index][obj] += sigmoid(output[index][obj]) - 1.0f;
            for (int attr = 0; attr < 2; ++attr) {
                float wanted = attr == 0 ? match.offset_x : match.offset_y;
                auto p = pos(match.anchor, attr, match.cell);
                loss += match.box_scale * binary_cross_entropy(output[index][p], wanted);
                gradient[index][p] += match.box_scale * (sigmoid(output[index][p]) - wanted);
            }
            for (int attr = 2; attr < 4; ++attr) {
                float wanted = attr == 2 ? match.log_width : match.log_height;
                auto p = pos(match.anchor, attr, match.cell);
                float error = output[index][p] - wanted;
                loss += 0.5f * match.box_scale * error * error;
                gradient[index][p] += match.box_scale * error;
            }
            for (int cls = 0; cls < layer.classes; ++cls) {
                float wanted = cls == match.class_id ? 1.0f : 0.0f;
                auto p = pos(match.anchor, 5 + cls, match.cell);
                loss += binary_cross_entropy(output[index][p], wanted);
                gradient[index][p] += sigmoid(output[index][p]) - wanted;
            }
        }
    }
    // No per-target normalization (matching Darknet: every target's full-scale delta is summed
    // as-is; only the trainer's SGD update averages across the minibatch).
    std::size_t positives = targets.size();
    return {loss, positives};
}

}  // namespace detector
