// Darknet's regressor.c example: trains a feedforward cfg (ends in [connected], no [softmax])
// against a per-sample scalar/vector regression target via mean-squared error, using
// VectorTrainer. Dataset format is this project's own fixed-record binary file (see
// vector_dataset.hpp) rather than Darknet's JPEG-directory-plus-text-label format, since this
// build has no OpenCV dependency.
#include "detector/classifier_trainer.hpp"
#include "detector/network.hpp"
#include "detector/vector_dataset.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    int epochs = 1;
    int limit = 0;
    float learning_rate = 0.01f;
    float decay = 0.0f;
    unsigned seed = 2026;
    std::string initial_weights;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 5; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("option requires a value");
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--epochs") options.epochs = std::stoi(value);
        else if (key == "--limit") options.limit = std::stoi(value);
        else if (key == "--lr") options.learning_rate = std::stof(value);
        else if (key == "--decay") options.decay = std::stof(value);
        else if (key == "--seed") options.seed = static_cast<unsigned>(std::stoul(value));
        else if (key == "--init") options.initial_weights = value;
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.epochs < 1 || options.limit < 0) throw std::runtime_error("invalid regressor options");
    return options;
}

void usage() {
    std::cerr << "Usage:\n"
              << "  cuda_regressor train <cfg> <output.weights> <data.vecbin>"
                 " [--epochs N] [--lr 0.01] [--decay 0] [--seed 2026] [--limit N] [--init weights]\n"
              << "  cuda_regressor eval <cfg> <weights> <data.vecbin> [--limit N]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5) { usage(); return 2; }
        const std::string command = argv[1];
        if (command != "train" && command != "eval") { usage(); return 2; }
        auto options = parse(argc, argv);
        auto network = detector::Network::from_cfg(argv[2]);
        const std::string weights = argv[3];
        const std::size_t input_size = network.input_shape().elements();
        const std::size_t target_size = network.layers().back().output.elements();
        detector::VectorDataset data(argv[4], input_size, target_size);
        std::size_t count = options.limit > 0 ?
            std::min(data.size(), static_cast<std::size_t>(options.limit)) : data.size();
        if (command == "train") {
            if (std::filesystem::exists(weights))
                throw std::runtime_error("refusing to overwrite existing output weights");
            if (options.initial_weights.empty()) network.initialize_random(options.seed);
            else network.load_weights(options.initial_weights);
            detector::VectorTrainer trainer(network, detector::VectorLoss::mse,
                options.learning_rate, options.decay);
            std::vector<std::size_t> order(count);
            std::iota(order.begin(), order.end(), 0);
            std::mt19937 rng(options.seed);
            for (int epoch = 1; epoch <= options.epochs; ++epoch) {
                std::shuffle(order.begin(), order.end(), rng);
                double loss_sum = 0.0;
                for (std::size_t i : order) loss_sum += trainer.step(data.input(i), data.target(i)).loss;
                network.save_weights(weights);
                std::cout << "epoch=" << epoch << " samples=" << count
                          << " mean_loss=" << std::fixed << std::setprecision(6) << loss_sum / count
                          << " checkpoint=" << weights << '\n';
            }
        } else {
            if (!options.initial_weights.empty()) throw std::runtime_error("--init is train-only");
            network.load_weights(weights);
            double squared_error_sum = 0.0;
            auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < count; ++i) {
                auto heads = network.forward(data.input(i), detector::Backend::cpu);
                if (heads.empty() || heads.back().layer != network.layers().size() - 1)
                    throw std::runtime_error("regressor output missing");
                const auto& prediction = heads.back().values;
                auto target = data.target(i);
                for (std::size_t j = 0; j < target_size; ++j) {
                    float diff = prediction[j] - target[j];
                    squared_error_sum += diff * diff;
                }
            }
            auto elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << "samples=" << count
                      << " mean_squared_error=" << std::fixed << std::setprecision(6)
                      << squared_error_sum / (count * target_size)
                      << " mean_ms=" << elapsed / count
                      << " scope=sample_conversion+forward (dataset file I/O and setup excluded)\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
