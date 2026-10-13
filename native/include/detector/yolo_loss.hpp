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

// Assigns each target to the best-IoU anchor across every [yolo]/[region] layer in `network`,
// then computes objectness/offset/class loss and log-space width/height loss, adding dL/doutput
// into `gradient[i]` for every such layer i (caller must size gradient[i] to
// network.layers()[i].output.elements() and zero it first; other entries are untouched).
// Negative (non-assigned) anchor/cell objectness is skipped when its decoded predicted box
// already overlaps some target by >= that layer's ignore_thresh (cfg `[yolo]` field, or
// `[region]`'s `thresh`; default 0.7/0.5 respectively): a near-miss Darknet convention keeps
// unpenalized rather than pushed to "background". Every matched target's x/y/width/height (not
// objectness/class) loss and gradient are scaled by Darknet's (2 - truth.w*truth.h) box scale
// (truth fractions of the network input) folded with the layer's coord_scale (1 for yolo), which
// up-weights small boxes; see delta_yolo_box/delta_region_box in pjreddie/darknet. `output[i]`
// must hold that layer's raw head values (any backend: CPU or CUDA). Neither the returned loss
// nor the written gradients are normalized by target count (matching Darknet, which sums every
// target's full-scale delta regardless of how many targets share an image; batch-size averaging
// happens once, in the trainer's SGD update). Shared by the CPU and CUDA trainers so both use
// identical target assignment and loss math.
//
// [yolo] layers use unit-weight BCE for objectness and independent per-class BCE (Darknet's
// yolo_layer.c convention). [region] layers (YOLOv2-style, Darknet's region_layer.c) instead use
// cfg-configurable object_scale/noobject_scale weights on positive/negative objectness (optionally
// rescored to the predicted box's IoU with its matched truth via `rescore`, instead of a flat 1),
// softmax+cross-entropy class loss scaled by class_scale (not independent per-class BCE), and all
// of a region layer's `anchors` apply to every cell (no [yolo]-style `mask` subset - `mask` is
// populated as the identity by the cfg parser so this shared code treats both kinds uniformly).
YoloLossResult yolo_loss_and_gradient(const Network& network, const std::vector<Target>& targets,
                                       const std::vector<std::vector<float>>& output,
                                       std::vector<std::vector<float>>& gradient);

}  // namespace detector
