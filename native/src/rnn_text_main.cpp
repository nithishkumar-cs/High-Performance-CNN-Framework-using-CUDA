// Byte-level char-sequence CLI (train/generate), mirroring Darknet's rnn.c tool but built on
// this project's own [rnn]/[lstm]/[gru] cfg layers and ClassificationTrainer::train_sequence -
// not a port of rnn.c. Vocabulary is the 256 possible byte values (no tokenizer), matching
// Darknet's own byte-level char-rnn; the cfg's [net] channels must be 256 and the final
// [softmax] must also have 256 outputs (next-byte prediction).
#include "detector/classifier_trainer.hpp"
#include "detector/cuda_classifier_trainer.hpp"
#include "detector/network.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kVocab = 256;

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open text file: " + path);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::vector<float> one_hot(unsigned char byte) {
    std::vector<float> encoded(kVocab, 0.0f);
    encoded[byte] = 1.0f;
    return encoded;
}

/** Raises if `network` isn't a byte-level sequence classifier: 256-wide one-hot input, exactly one recurrent layer, 256-class softmax output. */
void validate_byte_sequence_cfg(const detector::Network& network) {
    auto shape = network.input_shape();
    if (shape.c != kVocab || shape.h != 1 || shape.w != 1)
        throw std::runtime_error("cfg input must be channels=256 height=1 width=1 (byte one-hot)");
    if (network.layers().empty() || network.layers().back().kind != detector::Kind::softmax ||
        network.layers().back().output.c != kVocab)
        throw std::runtime_error("cfg must end in [softmax] with 256 outputs (next-byte prediction)");
    int recurrent = 0;
    for (const auto& layer : network.layers())
        if (layer.kind == detector::Kind::rnn || layer.kind == detector::Kind::lstm ||
            layer.kind == detector::Kind::gru)
            ++recurrent;
    if (recurrent != 1) throw std::runtime_error("cfg must contain exactly one [rnn]/[lstm]/[gru] layer");
}

struct TrainOptions {
    int epochs = 1;
    int time_steps = 32;
    float learning_rate = 0.01f;
    float decay = 0.0f;
    unsigned seed = 2026;
    std::string initial_weights;
    std::string backend = "cpu";
};

TrainOptions parse_train(int argc, char** argv) {
    TrainOptions options;
    for (int i = 5; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("option requires a value");
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--epochs") options.epochs = std::stoi(value);
        else if (key == "--time-steps") options.time_steps = std::stoi(value);
        else if (key == "--lr") options.learning_rate = std::stof(value);
        else if (key == "--decay") options.decay = std::stof(value);
        else if (key == "--seed") options.seed = static_cast<unsigned>(std::stoul(value));
        else if (key == "--init") options.initial_weights = value;
        else if (key == "--backend") options.backend = value;
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.epochs < 1 || options.time_steps < 1 ||
        (options.backend != "cpu" && options.backend != "cuda"))
        throw std::runtime_error("invalid train options");
    return options;
}

struct GenerateOptions {
    std::string seed_text = " ";
    int length = 200;
    float temperature = 1.0f;
    unsigned seed = 2026;
};

GenerateOptions parse_generate(int argc, char** argv) {
    GenerateOptions options;
    for (int i = 4; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("option requires a value");
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--seed-text") options.seed_text = value;
        else if (key == "--length") options.length = std::stoi(value);
        else if (key == "--temperature") options.temperature = std::stof(value);
        else if (key == "--seed") options.seed = static_cast<unsigned>(std::stoul(value));
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.seed_text.empty() || options.length < 1 || options.temperature <= 0.0f)
        throw std::runtime_error("invalid generate options");
    return options;
}

void usage() {
    std::cerr << "Usage:\n"
              << "  rnn_text train <cfg> <text_file> <output.weights>"
                 " [--backend cpu|cuda] [--epochs N] [--time-steps 32] [--lr 0.01] [--decay 0]"
                 " [--seed 2026] [--init weights]\n"
              << "  rnn_text generate <cfg> <weights>"
                 " [--seed-text \" \"] [--length 200] [--temperature 1.0] [--seed 2026]\n";
}

/** Samples a byte index from `probabilities` scaled by 1/temperature (1.0 = unscaled softmax draw). */
int sample(const std::vector<float>& probabilities, float temperature, std::mt19937& rng) {
    std::vector<double> weights(probabilities.size());
    double total = 0.0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        weights[i] = std::pow(static_cast<double>(std::max(probabilities[i], 1e-9f)),
                              1.0 / temperature);
        total += weights[i];
    }
    std::discrete_distribution<int> distribution(weights.begin(), weights.end());
    (void)total;
    return distribution(rng);
}

int run_train(int argc, char** argv) {
    auto options = parse_train(argc, argv);
    auto network = detector::Network::from_cfg(argv[2]);
    validate_byte_sequence_cfg(network);
    const std::string text = read_file(argv[3]);
    if (text.size() < 2) throw std::runtime_error("text file needs at least 2 bytes");
    const std::string weights = argv[4];
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

    const std::size_t predictable = text.size() - 1;  // last byte has no "next byte" target
    for (int epoch = 1; epoch <= options.epochs; ++epoch) {
        network.reset_sequence();
        double loss_sum = 0.0;
        std::size_t correct = 0;
        std::size_t position = 0;
        while (position < predictable) {
            const std::size_t chunk = std::min<std::size_t>(options.time_steps, predictable - position);
            std::vector<std::vector<float>> inputs(chunk);
            std::vector<int> targets(chunk);
            for (std::size_t t = 0; t < chunk; ++t) {
                inputs[t] = one_hot(static_cast<unsigned char>(text[position + t]));
                targets[t] = static_cast<unsigned char>(text[position + t + 1]);
            }
            std::vector<detector::ClassificationResult> results;
#ifdef DETECTOR_HAS_CUDA
            if (cuda_trainer) results = cuda_trainer->train_sequence(inputs, targets);
            else
#endif
                results = cpu_trainer->train_sequence(inputs, targets);
            for (const auto& result : results) {
                loss_sum += result.loss;
                correct += result.correct ? 1 : 0;
            }
            position += chunk;
        }
#ifdef DETECTOR_HAS_CUDA
        if (cuda_trainer) cuda_trainer->sync_to_network();
#endif
        network.save_weights(weights);
        std::cout << "epoch=" << epoch << " bytes=" << predictable
                  << " mean_loss=" << std::fixed << std::setprecision(6) << loss_sum / predictable
                  << " next_byte_accuracy=" << static_cast<double>(correct) / predictable
                  << " backend=" << options.backend << " checkpoint=" << weights << '\n';
    }
    return 0;
}

int run_generate(int argc, char** argv) {
    auto options = parse_generate(argc, argv);
    auto network = detector::Network::from_cfg(argv[2]);
    validate_byte_sequence_cfg(network);
    network.load_weights(argv[3]);
    network.reset_sequence();
    std::mt19937 rng(options.seed);

    std::vector<float> probabilities;
    for (unsigned char byte : options.seed_text)
        probabilities = network.classify(one_hot(byte), detector::Backend::cpu);
    std::string generated = options.seed_text;
    for (int i = 0; i < options.length; ++i) {
        int next = sample(probabilities, options.temperature, rng);
        generated.push_back(static_cast<char>(next));
        probabilities = network.classify(one_hot(static_cast<unsigned char>(next)), detector::Backend::cpu);
    }
    std::cout << generated << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 2; }
        const std::string command = argv[1];
        if (command == "train" && argc >= 5) return run_train(argc, argv);
        if (command == "generate" && argc >= 4) return run_generate(argc, argv);
        usage();
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
