#pragma once

#include "detector/network.hpp"
#include <vector>

namespace detector {

struct GroundTruth {
    int image_id = 0;
    int class_id = 0;
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
};

struct ScoredPrediction {
    int image_id = 0;
    Detection box;
};

struct ClassResult {
    int class_id = 0;
    int ground_truths = 0;
    int predictions = 0;
    float average_precision = 0;
};

std::vector<ClassResult> evaluate_ap(const std::vector<GroundTruth>& truth,
                                     const std::vector<ScoredPrediction>& predictions,
                                     int classes, float iou_threshold = 0.5f);
float mean_ap_present_classes(const std::vector<ClassResult>& results);

}  // namespace detector
