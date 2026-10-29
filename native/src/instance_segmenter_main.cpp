#include "detector/classifier_trainer.hpp"
#include "detector/image.hpp"
#include "detector/instance_segmentation.hpp"
#include "detector/network.hpp"
#include "detector/vector_dataset.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/** Training, dataset-limit, and checkpoint options for the binary mask CLI. */
struct TrainOptions {
    int epochs = 1;
    int limit = 0;
    float learning_rate = 0.01f;
    float decay = 0.0f;
    unsigned seed = 2026;
    std::string initial_weights;
};

/** Parses mask-head training and evaluation options. */
TrainOptions parse_train_options(int argc, char** argv) {
    TrainOptions options;
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
    if (options.epochs < 1 || options.limit < 0) throw std::runtime_error("invalid mask trainer options");
    return options;
}

/** Trains a binary mask network from ROI records or reports mask-only IoU on those records. */
void train_or_eval(int argc, char** argv, bool train) {
    if (argc < 5) throw std::runtime_error("train/eval requires cfg, weights and ROI dataset");
    const auto options = parse_train_options(argc, argv);
    auto network = detector::Network::from_cfg(argv[2]);
    const auto shape = network.layers().back().output;
    if (shape.c != 1) throw std::runtime_error("instance mask head must have one output channel");
    detector::VectorDataset data(argv[4], network.input_shape().elements(), shape.elements());
    const std::size_t count = options.limit > 0 ?
        std::min(data.size(), static_cast<std::size_t>(options.limit)) : data.size();
    const std::string weights = argv[3];
    if (train) {
        if (std::filesystem::exists(weights))
            throw std::runtime_error("refusing to overwrite existing output weights");
        if (options.initial_weights.empty()) network.initialize_random(options.seed);
        else network.load_weights(options.initial_weights);
        detector::VectorTrainer trainer(network, detector::VectorLoss::sigmoid_bce,
                                        options.learning_rate, options.decay);
        std::vector<std::size_t> order(count);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(options.seed);
        for (int epoch = 1; epoch <= options.epochs; ++epoch) {
            std::shuffle(order.begin(), order.end(), rng);
            double loss_sum = 0.0;
            for (std::size_t i : order) loss_sum += trainer.step(data.input(i), data.target(i)).loss;
            network.save_weights(weights);
            std::cout << "epoch=" << epoch << " roi_samples=" << count
                      << " mean_mask_loss=" << std::fixed << std::setprecision(6) << loss_sum / count
                      << " checkpoint=" << weights << '\n';
        }
    } else {
        if (!options.initial_weights.empty()) throw std::runtime_error("--init is train-only");
        network.load_weights(weights);
        double iou_sum = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const auto heads = network.forward(data.input(i), detector::Backend::cpu);
            if (heads.empty() || heads.back().values.size() != shape.elements())
                throw std::runtime_error("instance mask head output missing");
            const auto target = data.target(i);
            std::size_t intersection = 0;
            std::size_t union_count = 0;
            for (std::size_t pixel = 0; pixel < target.size(); ++pixel) {
                if (!std::isfinite(target[pixel]) || target[pixel] < 0.0f || target[pixel] > 1.0f)
                    throw std::runtime_error("mask targets must be in [0,1]");
                const bool predicted = heads.back().values[pixel] >= 0.0f;
                const bool actual = target[pixel] >= 0.5f;
                if (predicted && actual) ++intersection;
                if (predicted || actual) ++union_count;
            }
            iou_sum += union_count ? static_cast<double>(intersection) / union_count : 1.0;
        }
        std::cout << "roi_samples=" << count << " mean_mask_iou=" << std::fixed
                  << std::setprecision(6) << iou_sum / count << '\n';
    }
}

/** Writes one full-image PGM mask while retaining only the cropped mask in memory. */
void write_mask(const std::filesystem::path& path, const detector::InstanceMask& mask,
                int image_width, int image_height) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot write mask: " + path.string());
    file << "P5\n" << image_width << ' ' << image_height << "\n255\n";
    std::vector<std::uint8_t> row(static_cast<std::size_t>(image_width), 0);
    for (int y = 0; y < image_height; ++y) {
        if (y >= mask.y && y < mask.y + mask.height) {
            std::copy_n(mask.pixels.data() + static_cast<std::size_t>(y - mask.y) * mask.width,
                        mask.width, row.data() + mask.x);
        }
        file.write(reinterpret_cast<const char*>(row.data()), image_width);
        if (y >= mask.y && y < mask.y + mask.height)
            std::fill(row.begin() + mask.x, row.begin() + mask.x + mask.width, 0);
    }
    if (!file) throw std::runtime_error("failed writing mask: " + path.string());
}

/** Detects boxes and saves one original-image-space binary mask per detected instance. */
void predict(int argc, char** argv) {
    if (argc < 8) throw std::runtime_error("predict requires detector cfg/weights, mask cfg/weights, image and output dir");
    detector::Backend backend = detector::Backend::cpu;
    float confidence = 0.25f;
    float mask_threshold = 0.5f;
    for (int i = 8; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("option requires a value");
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--backend") {
            if (value == "cpu") backend = detector::Backend::cpu;
            else if (value == "cuda") backend = detector::Backend::cuda;
            else throw std::runtime_error("backend must be cpu or cuda");
        } else if (key == "--confidence") confidence = std::stof(value);
        else if (key == "--mask-threshold") mask_threshold = std::stof(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (!std::isfinite(confidence) || confidence < 0.0f || confidence > 1.0f)
        throw std::runtime_error("confidence must be in [0,1]");
    auto detector_network = detector::Network::from_cfg(argv[2]);
    detector_network.load_weights(argv[3]);
    auto mask_network = detector::Network::from_cfg(argv[4]);
    mask_network.load_weights(argv[5]);
    auto image = detector::read_image(argv[6]);
    const auto transform = detector::letterbox(image, detector_network.input_shape());
    auto boxes = detector_network.detect(transform.chw, backend, confidence);
    detector::undo_letterbox(boxes, transform);
    auto masks = detector::predict_instance_masks(mask_network, image, boxes, backend, mask_threshold);
    const std::filesystem::path output(argv[7]);
    if (std::filesystem::exists(output) && !std::filesystem::is_empty(output))
        throw std::runtime_error("refusing to overwrite nonempty output directory");
    std::filesystem::create_directories(output);
    std::ofstream index(output / "instances.tsv");
    if (!index) throw std::runtime_error("cannot write instances.tsv");
    index << "mask\tclass_id\tscore\tx1\ty1\tx2\ty2\n";
    for (std::size_t i = 0; i < masks.size(); ++i) {
        const std::string filename = "instance_" + std::to_string(i) + ".pgm";
        write_mask(output / filename, masks[i], image.width, image.height);
        const auto& box = masks[i].detection;
        index << filename << '\t' << box.class_id << '\t' << box.score << '\t'
              << box.x1 << '\t' << box.y1 << '\t' << box.x2 << '\t' << box.y2 << '\n';
    }
    if (!index) throw std::runtime_error("failed writing instances.tsv");
    std::cout << "instances=" << masks.size() << " output=" << output.string() << '\n';
}

/** Prints mask-head training and two-stage instance prediction commands. */
void usage() {
    std::cerr << "Usage:\n"
              << "  cuda_instance_segmenter train <mask.cfg> <output.weights> <roi_masks.vecbin>"
                 " [--epochs N] [--lr 0.01] [--decay 0] [--seed 2026] [--limit N] [--init weights]\n"
              << "  cuda_instance_segmenter eval <mask.cfg> <weights> <roi_masks.vecbin> [--limit N]\n"
              << "  cuda_instance_segmenter predict <detector.cfg> <detector.weights>"
                 " <mask.cfg> <mask.weights> <image> <output_dir>"
                 " [--backend cpu|cuda] [--confidence 0.25] [--mask-threshold 0.5]\n"
              << "Each ROI record: normalized CHW float32 crop, then H*W float32 binary mask.\n";
}

}  // namespace

/** Dispatches mask-head training, mask-only evaluation, and detector plus mask inference. */
int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 2; }
        const std::string command = argv[1];
        if (command == "train" || command == "eval") train_or_eval(argc, argv, command == "train");
        else if (command == "predict") predict(argc, argv);
        else { usage(); return 2; }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
