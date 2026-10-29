// Darknet's cifar.c example: a CIFAR-10 image classifier, using this project's existing
// classifier trainer/CUDA trainer unchanged - only the dataset format (CifarDataset) and CLI
// differ from cuda_classifier (IDX/MNIST), since CIFAR-10 ships 3-channel 32x32 images in its
// own single-file-per-batch binary format rather than IDX's separate image/label files.
#include "detector/classifier_trainer.hpp"
#include "detector/cifar.hpp"
#include "detector/cuda_session.hpp"
#include "detector/cuda_classifier_trainer.hpp"
#include "detector/network.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
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
    std::string backend = "cpu";
    std::vector<std::string> batch_paths;
};

Options parse(int argc, char** argv) {
    Options options;
    int i = 4;
    for (; i < argc && std::string(argv[i]).rfind("--", 0) != 0; ++i)
        options.batch_paths.push_back(argv[i]);
    for (; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("option requires a value");
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--epochs") options.epochs = std::stoi(value);
        else if (key == "--limit") options.limit = std::stoi(value);
        else if (key == "--lr") options.learning_rate = std::stof(value);
        else if (key == "--decay") options.decay = std::stof(value);
        else if (key == "--seed") options.seed = static_cast<unsigned>(std::stoul(value));
        else if (key == "--init") options.initial_weights = value;
        else if (key == "--backend") options.backend = value;
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.batch_paths.empty()) throw std::runtime_error("at least one CIFAR batch file is required");
    if (options.epochs < 1 || options.limit < 0 ||
        (options.backend != "cpu" && options.backend != "cuda"))
        throw std::runtime_error("invalid cifar options");
    return options;
}

void usage() {
    std::cerr << "Usage:\n"
              << "  cuda_cifar train <cfg> <output.weights> <batch1.bin> [batch2.bin ...]"
                 " [--backend cpu|cuda] [--epochs N] [--lr 0.01] [--decay 0] [--seed 2026] [--limit N] [--init weights]\n"
              << "  cuda_cifar eval <cfg> <weights> <batch1.bin> [batch2.bin ...] [--backend cpu|cuda] [--limit N]\n";
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
        detector::CifarDataset data(options.batch_paths);
        auto shape = network.input_shape();
        if (shape.c != detector::CifarDataset::channels() || shape.h != detector::CifarDataset::height() ||
            shape.w != detector::CifarDataset::width())
            throw std::runtime_error("cfg input does not match CIFAR-10's 3x32x32 dimensions");
        if (network.layers().back().kind != detector::Kind::softmax)
            throw std::runtime_error("classifier cfg must end in [softmax]");
        const int classes = network.layers().back().output.c;
        std::size_t count = options.limit > 0 ?
            std::min(data.size(), static_cast<std::size_t>(options.limit)) : data.size();
        if (command == "train") {
            if (std::filesystem::exists(weights))
                throw std::runtime_error("refusing to overwrite existing output weights");
            if (options.initial_weights.empty()) network.initialize_random(options.seed);
            else network.load_weights(options.initial_weights);
            std::unique_ptr<detector::ClassificationTrainer> cpu_trainer;
#ifdef DETECTOR_HAS_CUDA
            std::unique_ptr<detector::CudaClassificationTrainer> cuda_trainer;
            if (options.backend == "cuda")
                cuda_trainer = std::make_unique<detector::CudaClassificationTrainer>(
                    network, options.learning_rate, options.decay);
            else
#else
            if (options.backend == "cuda")
                throw std::runtime_error("CUDA backend not built; use -DDETECTOR_CUDA=ON");
#endif
                cpu_trainer = std::make_unique<detector::ClassificationTrainer>(
                    network, options.learning_rate, options.decay);
            std::vector<std::size_t> order(count);
            std::iota(order.begin(), order.end(), 0);
            std::mt19937 rng(options.seed);
            for (int epoch = 1; epoch <= options.epochs; ++epoch) {
                std::shuffle(order.begin(), order.end(), rng);
                double loss_sum = 0.0;
                std::size_t correct = 0;
                for (std::size_t i : order) {
                    int label = data.label(i);
                    if (label >= classes) throw std::runtime_error("CIFAR label outside cfg class range");
                    detector::ClassificationResult result;
#ifdef DETECTOR_HAS_CUDA
                    if (cuda_trainer) result = cuda_trainer->step(data.image(i), label);
                    else
#endif
                        result = cpu_trainer->step(data.image(i), label);
                    loss_sum += result.loss;
                    correct += result.correct ? 1 : 0;
                }
#ifdef DETECTOR_HAS_CUDA
                if (cuda_trainer) cuda_trainer->sync_to_network();
#endif
                network.save_weights(weights);
                std::cout << "epoch=" << epoch << " samples=" << count
                          << " mean_loss=" << std::fixed << std::setprecision(6) << loss_sum / count
                          << " train_accuracy=" << static_cast<double>(correct) / count
                          << " backend=" << options.backend << " checkpoint=" << weights << '\n';
            }
        } else {
            if (!options.initial_weights.empty()) throw std::runtime_error("--init is train-only");
            network.load_weights(weights);
#ifdef DETECTOR_HAS_CUDA
            std::unique_ptr<detector::CudaSession> cuda;
            if (options.backend == "cuda") cuda = std::make_unique<detector::CudaSession>(network);
#else
            if (options.backend == "cuda")
                throw std::runtime_error("CUDA backend not built; use -DDETECTOR_CUDA=ON");
#endif
            std::size_t correct = 0;
            auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < count; ++i) {
                std::vector<float> probabilities;
#ifdef DETECTOR_HAS_CUDA
                if (cuda) {
                    auto heads = cuda->forward(data.image(i));
                    if (heads.empty() || heads.back().layer != network.layers().size() - 1)
                        throw std::runtime_error("CUDA classifier output missing");
                    probabilities = std::move(heads.back().values);
                } else
#endif
                    probabilities = network.classify(data.image(i), detector::Backend::cpu);
                int predicted = static_cast<int>(std::max_element(probabilities.begin(), probabilities.end()) -
                                                 probabilities.begin());
                correct += predicted == data.label(i) ? 1 : 0;
            }
            auto elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << "samples=" << count << " correct=" << correct
                      << " accuracy=" << std::fixed << std::setprecision(6)
                      << static_cast<double>(correct) / count
                      << " mean_ms=" << elapsed / count
                      << " backend=" << options.backend
                      << " scope=sample_conversion+forward (CIFAR file I/O and setup excluded)\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
