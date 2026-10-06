#include "detector/network.hpp"
#include "detector/trainer.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Axis-aligned IoU in pixel space, matching decode()'s Detection coordinates.
float box_iou(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2) {
    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float area_a = (ax2 - ax1) * (ay2 - ay1);
    float area_b = (bx2 - bx1) * (by2 - by1);
    float un = area_a + area_b - inter;
    return un > 0 ? inter / un : 0.0f;
}

// Finds the highest-confidence decoded box of `class_id` and checks it recovers the ground-truth
// box (cx,cy,w,h normalized to [0,1]) with high confidence and IoU. Throws via `expect` on failure.
void check_recovered(const std::vector<detector::Detection>& boxes, detector::Shape input, int class_id,
                      float cx, float cy, float w, float h, const char* label) {
    const detector::Detection* best = nullptr;
    for (const auto& box : boxes) if (box.class_id == class_id && (!best || box.score > best->score)) best = &box;
    expect(best != nullptr, (std::string(label) + ": no detection of the target class survived").c_str());
    float gx1 = (cx - w / 2) * input.w, gy1 = (cy - h / 2) * input.h;
    float gx2 = (cx + w / 2) * input.w, gy2 = (cy + h / 2) * input.h;
    float overlap = box_iou(best->x1, best->y1, best->x2, best->y2, gx1, gy1, gx2, gy2);
    std::cout << label << " score=" << best->score << " iou=" << overlap << '\n';
    expect(best->score > 0.9f, (std::string(label) + ": confidence too low after overfitting").c_str());
    expect(overlap > 0.8f, (std::string(label) + ": did not localize the ground-truth box").c_str());
}

}  // namespace

// Diagnostic/regression test for the from-scratch YOLO anchor-assignment/loss/decode round trip:
// full-scale VOC retrains (CPU and GPU, see DESIGN.yaml) converge loss but leave mAP near zero,
// which cannot distinguish "undertrained" from "encode/decode pipeline bug" because both look the
// same at that scale. Overfitting tiny networks on a handful of FIXED images/boxes for many steps
// removes the capacity/budget/dataset-noise variables: if yolo_loss's target encoding, the backward
// pass, and network.cpp's decode are mutually consistent, this must drive decoded detections onto
// the ground-truth boxes with high confidence. Failure would pinpoint a real bug; passing on both a
// single-head/single-class case and a two-head/multi-target/many-class case (mirroring the actual
// failing architecture's anchor layout) rules the encode/decode/backward path out as the cause.
int main() {
    try {
        auto directory = std::filesystem::path(__FILE__).parent_path();

        // Scenario 1: smallest possible graph (mini.cfg: conv+BN -> maxpool -> conv -> upsample ->
        // route -> conv -> yolo, 1 anchor, 1 class) with one target, isolating the basic encode/
        // decode/backward wiring from any multi-head or multi-class assignment logic.
        {
            auto network = detector::Network::from_cfg((directory / "mini.cfg").string());
            network.initialize_random(21);
            std::vector<float> input(network.input_shape().elements());
            for (std::size_t i = 0; i < input.size(); ++i) input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
            const float cx = 0.625f, cy = 0.625f, w = 0.4f, h = 0.4f;
            std::vector<detector::Target> targets{{0, cx, cy, w, h}};

            detector::Trainer trainer(network, detector::LearningRateSchedule{0.1f}, 1, 0.0f);
            for (int i = 0; i < 6000; ++i) trainer.step({input}, {targets});

            auto heads = network.forward(input, detector::Backend::cpu);
            auto boxes = detector::decode(network, heads, 0.1f, 0.45f);
            check_recovered(boxes, network.input_shape(), 0, cx, cy, w, h, "mini single-target");
        }

        // Scenario 2: the actual architecture used by the failing full-scale VOC retrains
        // (voc-smoke.cfg: two YOLO heads at different grid scales with different anchor sets, 20
        // classes), overfit on two fixed targets of different classes/scales that the anchor-IoU
        // assignment must route to different heads/cells. Exercises the harder parts the mini.cfg
        // case cannot: cross-head anchor selection and multi-target, multi-class assignment.
        {
            auto network = detector::Network::from_cfg((directory / "voc-smoke.cfg").string());
            network.initialize_random(5);
            std::vector<float> input(network.input_shape().elements());
            int side = network.input_shape().w;
            for (int c = 0; c < network.input_shape().c; ++c)
                for (int y = 0; y < side; ++y)
                    for (int x = 0; x < side; ++x)
                        input[static_cast<std::size_t>(c) * side * side + y * side + x] =
                            0.5f + 0.3f * std::sin(0.3f * x + 0.21f * y + 0.7f * c);

            // Small box (anchor (8,10) on the fine 8x8 head) and a large box (anchor (48,56) on the
            // coarse 4x4 head), in different cells so neither assignment can mask the other.
            detector::Target small{2, 0.2f, 0.2f, 10.0f / 64, 12.0f / 64};
            detector::Target large{15, 0.7f, 0.6f, 44.0f / 64, 50.0f / 64};
            std::vector<detector::Target> targets{small, large};

            detector::Trainer trainer(network, detector::LearningRateSchedule{0.05f}, 1, 0.0f);
            for (int i = 0; i < 4000; ++i) trainer.step({input}, {targets});

            auto heads = network.forward(input, detector::Backend::cpu);
            auto boxes = detector::decode(network, heads, 0.1f, 0.45f);
            check_recovered(boxes, network.input_shape(), small.class_id, small.cx, small.cy,
                             small.width, small.height, "voc-smoke small target");
            check_recovered(boxes, network.input_shape(), large.class_id, large.cx, large.cy,
                             large.width, large.height, "voc-smoke large target");
        }

        std::cout << "overfit smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "overfit smoke failed: " << error.what() << '\n';
        return 1;
    }
}
