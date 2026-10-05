#include "detector/metrics.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        std::vector<detector::GroundTruth> truth{{0, 0, 0, 0, 2, 2}};
        std::vector<detector::ScoredPrediction> predictions{
            {0, {0, 0.9f, 5, 5, 7, 7}},
            {0, {0, 0.8f, 0, 0, 2, 2}},
            {0, {0, 0.7f, 0, 0, 2, 2}}
        };
        auto results = detector::evaluate_ap(truth, predictions, 2, 0.5f);
        if (results.size() != 2 || results[0].ground_truths != 1 ||
            std::abs(results[0].average_precision - 0.5f) > 1e-6f ||
            std::abs(detector::mean_ap_present_classes(results) - 0.5f) > 1e-6f)
            throw std::runtime_error("AP matching or precision envelope error");
        std::cout << "metrics smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metrics smoke failed: " << error.what() << '\n';
        return 1;
    }
}
