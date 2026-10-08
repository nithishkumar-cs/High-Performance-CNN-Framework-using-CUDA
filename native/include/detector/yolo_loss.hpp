#pragma once

#include "detector/network.hpp"
#include "detector/trainer.hpp"
#include <cstddef>
#include <vector>

namespace detector {

struct YoloLossResult {
    float loss = 0;
    std::size_t positives = 0;
};

// Assigns each target to the best-IoU anchor across every [yolo] layer in `network`, then
// computes BCE objectness/offset/class loss and log-space width/height loss, adding dL/doutput
// into `gradient[i]` for every yolo layer i (caller must size gradient[i] to
// network.layers()[i].output.elements() and zero it first; non-yolo entries are untouched).
// Negative (non-assigned) anchor/cell objectness is skipped when its decoded predicted box
// already overlaps some target by >= that layer's ignore_thresh (cfg `[yolo]` field, default
// 0.7): a near-miss Darknet convention keeps unpenalized rather than pushed to "background".
// Every remaining negative uses unit BCE weight, regardless of the number of candidates. Every
// matched target's x/y/width/height (not objectness/class) loss and gradient are scaled by
// Darknet's (2 - truth.w*truth.h) box scale (truth fractions of the network input), which
// up-weights small boxes; see delta_yolo_box in pjreddie/darknet's yolo_layer.c. `output[i]` must
// hold that yolo layer's raw head values (any backend: CPU or CUDA). Neither the returned loss
// nor the written gradients are normalized by target count (matching Darknet, which sums every
// target's full-scale delta regardless of how many targets share an image; batch-size averaging
// happens once, in the trainer's SGD update). Shared by the CPU and CUDA trainers so both use
// identical target assignment and loss math.
YoloLossResult yolo_loss_and_gradient(const Network& network, const std::vector<Target>& targets,
                                       const std::vector<std::vector<float>>& output,
                                       std::vector<std::vector<float>>& gradient);

}  // namespace detector
