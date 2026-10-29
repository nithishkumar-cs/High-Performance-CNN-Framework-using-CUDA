#include "detector/classifier_trainer.hpp"
#include "detector/cuda_classifier_trainer.hpp"
#include "detector/network.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
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

// Exercises Network/ClassificationTrainer's rnn/lstm/gru layer-kind integration (req-32): the
// task - classify every timestep of a length-T one-hot token sequence by the sequence's FIRST
// token - cannot be solved without the recurrent layer actually carrying hidden state across
// timesteps (an untrained or memoryless layer sees i.i.d. random tokens at t>0 and can do no
// better than chance on them), so convergence here is a real test of Network::reset_sequence(),
// ClassificationTrainer::train_sequence()'s chunked BPTT, and RecurrentCell's forward/backward.
void exercise_sequence(const fs::path& cfg) {
    auto network = detector::Network::from_cfg(cfg.string());
    network.initialize_random(7);
    detector::ClassificationTrainer trainer(network, 0.05f);
    constexpr int T = 4;
    std::mt19937 rng(123);
    std::bernoulli_distribution coin(0.5);
    auto one_hot = [](int token) { return std::vector<float>{token == 0 ? 1.0f : 0.0f, token == 1 ? 1.0f : 0.0f}; };

    double early = 0.0, late = 0.0;
    constexpr int sequences = 800;  // gru converges markedly slower than rnn/lstm on this task
    for (int s = 0; s < sequences; ++s) {
        int label = coin(rng) ? 1 : 0;
        std::vector<std::vector<float>> inputs(T);
        std::vector<int> class_ids(T, label);
        inputs[0] = one_hot(label);
        for (int t = 1; t < T; ++t) inputs[t] = one_hot(coin(rng) ? 1 : 0);
        network.reset_sequence();
        auto results = trainer.train_sequence(inputs, class_ids);
        double sequence_loss = 0.0;
        for (const auto& r : results) { assert(std::isfinite(r.loss)); sequence_loss += r.loss; }
        sequence_loss /= T;
        if (s < 40) early += sequence_loss;
        if (s >= sequences - 40) late += sequence_loss;
    }
    if (!(late < early * 0.7))
        throw std::runtime_error("sequence classification loss failed to decrease: " + cfg.string());

    // Checkpoint round trip: a reloaded network must classify identically (same hidden-state
    // trajectory) given the same sequence from a fresh reset_sequence().
    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path weights = fs::temp_directory_path() /
        ("cuda-framework-rnn-classifier-" + std::to_string(stamp) + ".weights");
    network.save_weights(weights.string());
    auto restored = detector::Network::from_cfg(cfg.string());
    restored.load_weights(weights.string());
    fs::remove(weights);
    for (int label = 0; label < 2; ++label) {
        std::vector<std::vector<float>> inputs = {one_hot(label), one_hot(1 - label), one_hot(label)};
        network.reset_sequence();
        restored.reset_sequence();
        for (const auto& input : inputs) {
            auto original = network.classify(input, detector::Backend::cpu);
            auto roundtrip = restored.classify(input, detector::Backend::cpu);
            for (std::size_t i = 0; i < original.size(); ++i)
                if (std::abs(original[i] - roundtrip[i]) > 1e-6f)
                    throw std::runtime_error("rnn classifier checkpoint mismatch");
        }
    }
    std::cout << cfg.filename().string() << " seq_early_loss=" << early / 40
              << " seq_late_loss=" << late / 40 << "\n";
}

#ifdef DETECTOR_HAS_CUDA
/** Checks stateful inference and two truncated BPTT updates against the CPU trainer. */
void exercise_cuda_sequence_parity(const fs::path& cfg) {
    auto cpu = detector::Network::from_cfg(cfg.string());
    auto cuda = detector::Network::from_cfg(cfg.string());
    cpu.initialize_random(71);
    cuda.initialize_random(71);
    const std::vector<std::vector<float>> tokens{{1.0f, 0.0f}, {0.0f, 1.0f},
                                                  {0.3f, -0.5f}, {-0.2f, 0.6f},
                                                  {1.0f, 0.0f}};
    std::vector<std::vector<float>> inputs(tokens.size(),
        std::vector<float>(cpu.input_shape().elements()));
    for (std::size_t t = 0; t < inputs.size(); ++t)
        for (std::size_t p = 0; p < inputs[t].size(); ++p)
            inputs[t][p] = tokens[t][p % 2] + static_cast<float>(p % 5) * 0.04f;
    cpu.reset_sequence(); cuda.reset_sequence();
    for (const auto& input : inputs) {
        auto a = cpu.classify(input, detector::Backend::cpu);
        auto b = cuda.classify(input, detector::Backend::cuda);
        for (std::size_t i = 0; i < a.size(); ++i)
            if (std::abs(a[i] - b[i]) > 2e-4f)
                throw std::runtime_error("CUDA recurrent inference mismatch: " + cfg.string());
    }
    cpu.reset_sequence(); cuda.reset_sequence();
    detector::ClassificationTrainer cpu_trainer(cpu, 0.01f);
    detector::CudaClassificationTrainer cuda_trainer(cuda, 0.01f);
    for (int chunk = 0; chunk < 2; ++chunk) {
        std::vector<std::vector<float>> samples = chunk ?
            std::vector<std::vector<float>>(inputs.begin() + 3, inputs.end()) :
            std::vector<std::vector<float>>(inputs.begin(), inputs.begin() + 3);
        std::vector<int> targets = chunk ? std::vector<int>{1, 0} : std::vector<int>{0, 0, 1};
        auto a = cpu_trainer.train_sequence(samples, targets);
        auto b = cuda_trainer.train_sequence(samples, targets);
        for (std::size_t t = 0; t < a.size(); ++t)
            if (std::abs(a[t].loss - b[t].loss) > 2e-4f)
                throw std::runtime_error("CUDA recurrent training loss mismatch: " + cfg.string());
        cuda_trainer.sync_to_network();
        for (std::size_t i = 0; i < cpu.layers().size(); ++i) {
            if (auto* cell = cpu.cell_at(i)) {
                auto expected = cell->parameter_values();
                auto actual = cuda.cell_at(i)->parameter_values();
                for (std::size_t p = 0; p < expected.size(); ++p)
                    if (std::abs(expected[p] - actual[p]) > 3e-4f)
                        throw std::runtime_error("CUDA recurrent parameter mismatch: " + cfg.string());
            } else {
                const auto& expected = cpu.layers()[i];
                const auto& actual = cuda.layers()[i];
                for (std::size_t p = 0; p < expected.weights.size(); ++p)
                    if (std::abs(expected.weights[p] - actual.weights[p]) > 3e-4f)
                        throw std::runtime_error("CUDA sequence dense weight mismatch: " + cfg.string());
                for (std::size_t p = 0; p < expected.biases.size(); ++p)
                    if (std::abs(expected.biases[p] - actual.biases[p]) > 3e-4f)
                        throw std::runtime_error("CUDA sequence dense bias mismatch: " + cfg.string());
            }
        }
    }
    cpu.reset_sequence(); cuda.reset_sequence();
    for (const auto& input : inputs) {
        auto a = cpu.classify(input, detector::Backend::cpu);
        auto b = cuda.classify(input, detector::Backend::cuda);
        for (std::size_t i = 0; i < a.size(); ++i)
            if (std::abs(a[i] - b[i]) > 3e-4f)
                throw std::runtime_error("CUDA trained recurrent inference mismatch: " + cfg.string());
    }
}

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

/** VectorTrainer (regressor.c-style MSE, or tag.c-style multi-label sigmoid_bce) learns to
 *  separate two fixed targets from a zeros/ones input, mirroring exercise()'s classifier check. */
void exercise_vector(const fs::path& cfg, detector::VectorLoss loss,
                      const std::vector<float>& target_a, const std::vector<float>& target_b) {
    auto network = detector::Network::from_cfg(cfg.string());
    network.initialize_random(41);
    detector::VectorTrainer trainer(network, loss, 0.1f);
    const std::vector<float> zeros(network.input_shape().elements(), 0.0f);
    const std::vector<float> ones(network.input_shape().elements(), 1.0f);
    double early = 0.0, late = 0.0;
    for (int i = 0; i < 300; ++i) {
        auto result = trainer.step(i % 2 ? ones : zeros, i % 2 ? target_b : target_a);
        if (!std::isfinite(result.loss)) throw std::runtime_error("non-finite vector trainer loss");
        if (loss == detector::VectorLoss::sigmoid_bce)
            for (float p : result.output)
                if (p < 0.0f || p > 1.0f) throw std::runtime_error("sigmoid_bce output outside [0,1]");
        if (i < 20) early += result.loss;
        if (i >= 280) late += result.loss;
    }
    if (!(late < early * 0.5))
        throw std::runtime_error("vector trainer loss failed to decrease: " + cfg.string());
    auto predicted_a = trainer.step(zeros, target_a).output;
    auto predicted_b = trainer.step(ones, target_b).output;
    for (std::size_t i = 0; i < target_a.size(); ++i) {
        if (std::abs(predicted_a[i] - target_a[i]) > 0.25f || std::abs(predicted_b[i] - target_b[i]) > 0.25f)
            throw std::runtime_error("vector trainer did not learn: " + cfg.string());
    }
}

/** Checks exact pixelwise CE gradients and learning on a two-class 1x1 segmentation model. */
void exercise_segmentation(const fs::path& cfg) {
    auto network = detector::Network::from_cfg(cfg.string());
    network.initialize_random(12);
    auto& head = network.layer(0);
    std::fill(head.weights.begin(), head.weights.end(), 0.0f);
    std::fill(head.biases.begin(), head.biases.end(), 0.0f);
    detector::VectorTrainer trainer(network, detector::VectorLoss::pixel_softmax_ce, 0.2f);
    const std::vector<float> input{1.0f, 0.0f, 1.0f, 0.0f};
    const std::vector<float> target{1.0f, 0.0f, 1.0f, 0.0f};
    auto first = trainer.step(input, target);
    if (std::abs(first.loss - std::log(2.0f)) > 1e-6f || first.output.size() != 8 ||
        std::abs(head.weights[0] + 0.05f) > 1e-6f ||
        std::abs(head.weights[1] - 0.05f) > 1e-6f)
        throw std::runtime_error("pixelwise CE loss or gradient mismatch");
    for (float probability : first.output)
        if (std::abs(probability - 0.5f) > 1e-6f)
            throw std::runtime_error("pixelwise CE probability mismatch");
    float last = first.loss;
    for (int i = 0; i < 100; ++i) last = trainer.step(input, target).loss;
    if (!(last < 0.2f)) throw std::runtime_error("segmentation loss did not converge");
    try {
        trainer.step(input, {1.0f, 0.0f, 2.0f, 0.0f});
        throw std::runtime_error("out-of-range segmentation target accepted");
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()) != "pixel_softmax_ce targets must be integer class IDs or -1 for ignore") throw;
    }
    auto before_ignore = head.weights;
    try {
        trainer.step(input, {-1.0f, -1.0f, -1.0f, -1.0f});
        throw std::runtime_error("all-ignored segmentation target accepted");
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()) != "pixel_softmax_ce requires at least one labeled pixel") throw;
    }
    if (head.weights != before_ignore ||
        !std::isfinite(trainer.step(input, {1.0f, -1.0f, 1.0f, -1.0f}).loss))
        throw std::runtime_error("ignored segmentation pixels affected training");
}

#ifdef DETECTOR_HAS_CUDA
/** Checks CPU and CUDA vector-loss trainers produce identical losses, outputs, and weights over several steps. */
void exercise_cuda_vector_parity(const fs::path& cfg, detector::VectorLoss loss,
                                 const std::vector<std::vector<float>>& inputs,
                                 const std::vector<std::vector<float>>& targets) {
    auto cpu = detector::Network::from_cfg(cfg.string());
    auto cuda = detector::Network::from_cfg(cfg.string());
    cpu.initialize_random(5);
    cuda.initialize_random(5);
    detector::VectorTrainer cpu_trainer(cpu, loss, 0.05f);
    detector::CudaClassificationTrainer cuda_trainer(cuda, 0.05f, 0.0f, loss);
    for (int step = 0; step < 6; ++step)
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            auto a = cpu_trainer.step(inputs[i], targets[i]);
            auto b = cuda_trainer.step_vector(inputs[i], targets[i]);
            if (std::abs(a.loss - b.loss) > 1e-5f || a.output.size() != b.output.size())
                throw std::runtime_error("CPU/CUDA vector trainer loss mismatch: " + cfg.string());
            for (std::size_t j = 0; j < a.output.size(); ++j)
                if (std::abs(a.output[j] - b.output[j]) > 1e-5f)
                    throw std::runtime_error("CPU/CUDA vector trainer output mismatch: " + cfg.string());
        }
    cuda_trainer.sync_to_network();
    for (std::size_t index = 0; index < cpu.layers().size(); ++index) {
        const auto& a = cpu.layers()[index];
        const auto& b = cuda.layers()[index];
        for (std::size_t j = 0; j < a.weights.size(); ++j)
            if (std::abs(a.weights[j] - b.weights[j]) > 1e-5f)
                throw std::runtime_error("CPU/CUDA vector trainer weight mismatch: " + cfg.string());
        for (std::size_t j = 0; j < a.biases.size(); ++j)
            if (std::abs(a.biases[j] - b.biases[j]) > 1e-5f)
                throw std::runtime_error("CPU/CUDA vector trainer bias mismatch: " + cfg.string());
    }
}
#endif

/** Exercises CPU/CUDA classifier training, sequence state, and checkpoint parity. */
int main() {
    try {
        fs::path directory = fs::path(__FILE__).parent_path();
        exercise(directory / "classifier-linear.cfg");
        exercise(directory / "classifier-conv.cfg");
        exercise(directory / "classifier-dense.cfg");
        exercise_sequence(directory / "rnn-classifier.cfg");
        exercise_sequence(directory / "lstm-classifier.cfg");
        exercise_sequence(directory / "gru-classifier.cfg");
        exercise_vector(directory / "regressor-vector.cfg", detector::VectorLoss::mse,
                         {0.1f, -0.5f, 0.2f}, {0.9f, 0.4f, -0.3f});
        exercise_vector(directory / "regressor-vector.cfg", detector::VectorLoss::sigmoid_bce,
                          {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 1.0f});
        exercise_segmentation(directory / "segmenter-vector.cfg");
#ifdef DETECTOR_HAS_CUDA
        exercise_cuda_training(directory / "classifier-linear.cfg");
        exercise_cuda_training(directory / "classifier-dense.cfg");
        exercise_cuda_training(directory / "classifier-conv.cfg");
        exercise_single_step_parity(directory / "classifier-linear.cfg");
        exercise_single_step_parity(directory / "classifier-dense.cfg");
        exercise_single_step_parity(directory / "classifier-conv.cfg");
        exercise_single_step_parity(directory / "l2norm-classifier.cfg");
        exercise_single_step_parity(directory / "avgpool-classifier.cfg");
        exercise_single_step_parity(directory / "normalization-classifier.cfg");
        exercise_single_step_parity(directory / "local-classifier.cfg");
        exercise_single_step_parity(directory / "deconv-classifier.cfg");
        exercise_cuda_sequence_parity(directory / "rnn-classifier.cfg");
        exercise_cuda_sequence_parity(directory / "lstm-classifier.cfg");
        exercise_cuda_sequence_parity(directory / "gru-classifier.cfg");
        exercise_cuda_sequence_parity(directory / "rnn-conv-classifier.cfg");
        exercise_cuda_sequence_parity(directory / "rnn-l2norm-classifier.cfg");
        exercise_cuda_sequence_parity(directory / "rnn-avgpool-classifier.cfg");
        exercise_cuda_sequence_parity(directory / "rnn-normalization-classifier.cfg");
        exercise_cuda_sequence_parity(directory / "rnn-local-classifier.cfg");
        exercise_cuda_sequence_parity(directory / "rnn-deconv-classifier.cfg");
        exercise_cuda_vector_parity(directory / "segmenter-vector.cfg", detector::VectorLoss::pixel_softmax_ce,
                                    {{1.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
                                    {{1.0f, 0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f, 1.0f}});
        exercise_cuda_vector_parity(directory / "regressor-vector.cfg", detector::VectorLoss::mse,
                                    {{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f}}, {{0.9f, 0.4f, -0.3f}});
        exercise_cuda_vector_parity(directory / "regressor-vector.cfg", detector::VectorLoss::sigmoid_bce,
                                    {{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f}}, {{1.0f, 0.0f, 1.0f}});
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
