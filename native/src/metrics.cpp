#include "detector/metrics.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace detector {
namespace {

float overlap(const GroundTruth& truth, const Detection& box) {
    float width = std::max(0.0f, std::min(truth.x2, box.x2) - std::max(truth.x1, box.x1));
    float height = std::max(0.0f, std::min(truth.y2, box.y2) - std::max(truth.y1, box.y1));
    float intersection = width * height;
    float truth_area = std::max(0.0f, truth.x2 - truth.x1) * std::max(0.0f, truth.y2 - truth.y1);
    float box_area = std::max(0.0f, box.x2 - box.x1) * std::max(0.0f, box.y2 - box.y1);
    float combined = truth_area + box_area - intersection;
    return combined > 0 ? intersection / combined : 0;
}

}  // namespace

std::vector<ClassResult> evaluate_ap(const std::vector<GroundTruth>& truth,
                                     const std::vector<ScoredPrediction>& predictions,
                                     int classes, float iou_threshold) {
    if (classes <= 0 || !(iou_threshold > 0 && iou_threshold <= 1)) throw std::runtime_error("invalid AP settings");
    std::vector<ClassResult> results;
    results.reserve(static_cast<std::size_t>(classes));
    for (int cls = 0; cls < classes; ++cls) {
        std::vector<const GroundTruth*> relevant_truth;
        std::vector<const ScoredPrediction*> relevant_predictions;
        for (const auto& item : truth) if (item.class_id == cls) relevant_truth.push_back(&item);
        for (const auto& item : predictions) if (item.box.class_id == cls) relevant_predictions.push_back(&item);
        std::sort(relevant_predictions.begin(), relevant_predictions.end(),
                  [](const auto* a, const auto* b) { return a->box.score > b->box.score; });
        ClassResult result{cls, static_cast<int>(relevant_truth.size()), static_cast<int>(relevant_predictions.size()), 0};
        if (relevant_truth.empty()) { results.push_back(result); continue; }
        std::vector<unsigned char> used(relevant_truth.size(), 0);
        std::vector<float> recall{0.0f};
        std::vector<float> precision{0.0f};
        int true_positives = 0;
        int false_positives = 0;
        for (const auto* prediction : relevant_predictions) {
            float best_overlap = iou_threshold;
            std::size_t best = relevant_truth.size();
            for (std::size_t i = 0; i < relevant_truth.size(); ++i) {
                if (used[i] || relevant_truth[i]->image_id != prediction->image_id) continue;
                float value = overlap(*relevant_truth[i], prediction->box);
                if (value >= best_overlap) { best_overlap = value; best = i; }
            }
            if (best < relevant_truth.size()) { used[best] = 1; ++true_positives; }
            else ++false_positives;
            recall.push_back(static_cast<float>(true_positives) / result.ground_truths);
            precision.push_back(static_cast<float>(true_positives) / (true_positives + false_positives));
        }
        recall.push_back(1.0f);
        precision.push_back(0.0f);
        for (std::size_t i = precision.size() - 1; i > 0; --i)
            precision[i - 1] = std::max(precision[i - 1], precision[i]);
        for (std::size_t i = 1; i < recall.size(); ++i)
            if (recall[i] > recall[i - 1]) result.average_precision +=
                (recall[i] - recall[i - 1]) * precision[i];
        results.push_back(result);
    }
    return results;
}

float mean_ap_present_classes(const std::vector<ClassResult>& results) {
    float sum = 0;
    int count = 0;
    for (const auto& item : results) if (item.ground_truths > 0) { sum += item.average_precision; ++count; }
    return count ? sum / count : 0.0f;
}

}  // namespace detector
