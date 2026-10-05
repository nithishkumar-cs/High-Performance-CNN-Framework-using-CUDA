#include "detector/classifier_trainer.hpp"
#include "detector/cuda_classifier_trainer.hpp"
#include "detector/network.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void exercise(const fs::path& cfg) {
    auto network = detector::Network::from_cfg(cfg.string());
    network.initialize_random(41);
    detector::ClassificationTrainer trainer(network, 0.03f);
    const std::vector<float> zeros(network.input_shape().elements(), 0.0f);
    const std::vector<float> ones(network.input_shape().elements(), 1.0f);
    double early = 0.0;
    double late = 0.0;
    for (int i = 0; i < 200; ++i) {
        auto result = trainer.step(i % 2 ? ones : zeros, i % 2);
        assert(std::isfinite(result.loss));
        if (i < 20) early += result.loss;
        if (i >= 180) late += result.loss;
    }
    if (!(late < early * 0.7))
        throw std::runtime_error("classification loss failed to decrease: " + cfg.string());
    for (int label = 0; label < 2; ++label) {
        auto probabilities = network.classify(label ? ones : zeros, detector::Backend::cpu);
        if (probabilities.size() != 2 || std::abs(probabilities[0] + probabilities[1] - 1.0f) > 1e-5f ||
            probabilities[static_cast<std::size_t>(label)] < 0.8f)
            throw std::runtime_error("classifier did not learn: " + cfg.string());
    }

    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path weights = fs::temp_directory_path() /
        ("cuda-framework-classifier-" + std::to_string(stamp) + ".weights");
    network.save_weights(weights.string());
    auto restored = detector::Network::from_cfg(cfg.string());
    restored.load_weights(weights.string());
    fs::remove(weights);
    for (int label = 0; label < 2; ++label) {
        const auto& input = label ? ones : zeros;
        auto original = network.classify(input, detector::Backend::cpu);
        auto roundtrip = restored.classify(input, detector::Backend::cpu);
        for (std::size_t i = 0; i < original.size(); ++i)
            if (std::abs(original[i] - roundtrip[i]) > 1e-6f)
                throw std::runtime_error("classifier checkpoint mismatch");
#ifdef DETECTOR_HAS_CUDA
        auto cuda = restored.classify(input, detector::Backend::cuda);
        for (std::size_t i = 0; i < original.size(); ++i)
            if (std::abs(original[i] - cuda[i]) > 1e-4f)
                throw std::runtime_error("classifier CPU/CUDA mismatch");
#endif
    }
    std::cout << cfg.filename().string() << " early_loss=" << early / 20
              << " late_loss=" << late / 20 << "\n";
}

#ifdef DETECTOR_HAS_CUDA
void exercise_single_step_parity(const fs::path& cfg) {
    auto cpu = detector::Network::from_cfg(cfg.string());
    auto cuda = detector::Network::from_cfg(cfg.string());
    cpu.initialize_random(71);
    cuda.initialize_random(71);
    std::vector<float> input(cpu.input_shape().elements());
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>((i * 3) % 11) / 10.0f;
    detector::ClassificationTrainer cpu_trainer(cpu, 0.001f);
    detector::CudaClassificationTrainer cuda_trainer(cuda, 0.001f);
    float cpu_loss = cpu_trainer.step(input, 1).loss;
    float cuda_loss = cuda_trainer.step(input, 1).loss;
    cuda_trainer.sync_to_network();
    if (std::abs(cpu_loss - cuda_loss) > 1e-4f)
        throw std::runtime_error("CPU/CUDA classifier loss mismatch: " + cfg.string());
    for (std::size_t index = 0; index < cpu.layers().size(); ++index) {
        const auto& a = cpu.layers()[index];
        const auto& b = cuda.layers()[index];
        if (a.weights.size() != b.weights.size() || a.biases.size() != b.biases.size())
            throw std::runtime_error("CPU/CUDA classifier parameter shape mismatch");
        for (std::size_t j = 0; j < a.weights.size(); ++j)
            if (std::abs(a.weights[j] - b.weights[j]) > 1e-4f)
                throw std::runtime_error("CPU/CUDA classifier weight update mismatch: " + cfg.string());
        for (std::size_t j = 0; j < a.biases.size(); ++j)
            if (std::abs(a.biases[j] - b.biases[j]) > 1e-4f)
                throw std::runtime_error("CPU/CUDA classifier bias update mismatch: " + cfg.string());
    }
}

void exercise_cuda_training(const fs::path& cfg) {
    auto network = detector::Network::from_cfg(cfg.string());
    network.initialize_random(41);
    detector::CudaClassificationTrainer trainer(network, 0.03f);
    const std::vector<float> zeros(network.input_shape().elements(), 0.0f);
    const std::vector<float> ones(network.input_shape().elements(), 1.0f);
    double early = 0.0;
    double late = 0.0;
    for (int i = 0; i < 200; ++i) {
        auto result = trainer.step(i % 2 ? ones : zeros, i % 2);
        if (!std::isfinite(result.loss)) throw std::runtime_error("non-finite CUDA loss");
        if (i < 20) early += result.loss;
        if (i >= 180) late += result.loss;
    }
    if (!(late < early * 0.7))
        throw std::runtime_error("CUDA classification loss failed to decrease: " + cfg.string());
    trainer.sync_to_network();
    for (int label = 0; label < 2; ++label) {
        const auto& input = label ? ones : zeros;
        auto cpu = network.classify(input, detector::Backend::cpu);
        auto cuda = network.classify(input, detector::Backend::cuda);
        if (cpu[static_cast<std::size_t>(label)] < 0.8f)
            throw std::runtime_error("CUDA classifier did not learn");
        for (std::size_t i = 0; i < cpu.size(); ++i)
            if (std::abs(cpu[i] - cuda[i]) > 1e-4f)
                throw std::runtime_error("CUDA learned weights disagree with CPU inference");
    }
    std::cout << cfg.filename().string() << " cuda_early_loss=" << early / 20
              << " cuda_late_loss=" << late / 20 << "\n";
}
#endif

int main() {
    try {
        fs::path directory = fs::path(__FILE__).parent_path();
        exercise(directory / "classifier-linear.cfg");
        exercise(directory / "classifier-conv.cfg");
        exercise(directory / "classifier-dense.cfg");
#ifdef DETECTOR_HAS_CUDA
        exercise_cuda_training(directory / "classifier-linear.cfg");
        exercise_cuda_training(directory / "classifier-dense.cfg");
        exercise_cuda_training(directory / "classifier-conv.cfg");
        exercise_single_step_parity(directory / "classifier-linear.cfg");
        exercise_single_step_parity(directory / "classifier-dense.cfg");
        exercise_single_step_parity(directory / "classifier-conv.cfg");
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
