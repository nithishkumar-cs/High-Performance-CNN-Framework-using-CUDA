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
// `output[i]` must hold that yolo layer's raw head values (any backend: CPU or CUDA). Both the
// returned loss and every written gradient are normalized by max(1, targets.size()). Shared by
// the CPU and CUDA trainers so both use identical target assignment and loss math.
YoloLossResult yolo_loss_and_gradient(const Network& network, const std::vector<Target>& targets,
                                       const std::vector<std::vector<float>>& output,
                                       std::vector<std::vector<float>>& gradient);

}  // namespace detector
