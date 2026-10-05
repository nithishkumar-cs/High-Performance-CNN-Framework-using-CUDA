#include "detector/image.hpp"
#include "detector/metrics.hpp"
#include "detector/network.hpp"
#ifdef DETECTOR_HAS_CUDA
#include "detector/cuda_session.hpp"
#endif
#ifdef DETECTOR_HAS_TENSORRT
#include "detector/trt_session.hpp"
#endif
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

struct Options {
    detector::Backend backend = detector::Backend::cpu;
    float confidence = 0.01f;
    float nms_iou = 0.45f;
    int limit = 0;
};

Options options_from(int argc, char** argv) {
    Options options;
    for (int i = 5; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("option requires a value");
        std::string key = argv[i];
        std::string value = argv[i + 1];
        if (key == "--backend") {
            if (value == "cpu") options.backend = detector::Backend::cpu;
            else if (value == "cuda") options.backend = detector::Backend::cuda;
            else if (value == "tensorrt") options.backend = detector::Backend::tensorrt;
            else throw std::runtime_error("backend must be cpu, cuda, or tensorrt");
        } else if (key == "--confidence") options.confidence = std::stof(value);
        else if (key == "--nms-iou") options.nms_iou = std::stof(value);
        else if (key == "--limit") options.limit = std::stoi(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (!(options.confidence >= 0 && options.confidence <= 1 &&
          options.nms_iou >= 0 && options.nms_iou <= 1 && options.limit >= 0))
        throw std::runtime_error("invalid evaluator options");
    return options;
}

std::vector<fs::path> images_from(const fs::path& directory) {
    if (!fs::is_directory(directory)) throw std::runtime_error("not an image directory: " + directory.string());
    std::vector<fs::path> images;
    for (const auto& item : fs::directory_iterator(directory)) {
        if (!item.is_regular_file()) continue;
        std::string extension = item.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension == ".ppm" || extension == ".jpg" || extension == ".jpeg" || extension == ".png")
            images.push_back(item.path());
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) throw std::runtime_error("no images found in " + directory.string());
    return images;
}

void read_truth(const fs::path& path, int image_id, int width, int height,
                std::vector<detector::GroundTruth>& truth) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("missing label file: " + path.string());
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream stream(line);
        int cls;
        float cx, cy, w, h;
        std::string extra;
        if (!(stream >> cls >> cx >> cy >> w >> h) || (stream >> extra))
            throw std::runtime_error("malformed YOLO label in " + path.string());
        truth.push_back({image_id, cls,
                         std::max(0.0f, (cx - w / 2) * width),
                         std::max(0.0f, (cy - h / 2) * height),
                         std::min(static_cast<float>(width), (cx + w / 2) * width),
                         std::min(static_cast<float>(height), (cy + h / 2) * height)});
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5) {
            std::cerr << "Usage: cuda_detector_eval <cfg> <weights|engine> <images_dir> <labels_dir> "
                         "[--backend cpu|cuda|tensorrt] [--confidence 0.01] [--nms-iou 0.45] [--limit N]\n";
            return 2;
        }
        auto options = options_from(argc, argv);
        auto network = detector::Network::from_cfg(argv[1]);
        if (options.backend != detector::Backend::tensorrt) network.load_weights(argv[2]);
        auto images = images_from(argv[3]);
        if (options.limit > 0 && images.size() > static_cast<std::size_t>(options.limit))
            images.resize(options.limit);
        int classes = -1;
        for (const auto& layer : network.layers()) if (layer.kind == detector::Kind::yolo) {
            if (classes < 0) classes = layer.classes;
            else if (classes != layer.classes) throw std::runtime_error("YOLO heads disagree on class count");
        }
#ifdef DETECTOR_HAS_CUDA
        std::unique_ptr<detector::CudaSession> cuda;
        if (options.backend == detector::Backend::cuda)
            cuda = std::make_unique<detector::CudaSession>(network);
#endif
#ifdef DETECTOR_HAS_TENSORRT
        std::unique_ptr<detector::TrtSession> trt;
        if (options.backend == detector::Backend::tensorrt)
            trt = std::make_unique<detector::TrtSession>(network, argv[2]);
#else
        if (options.backend == detector::Backend::tensorrt)
            throw std::runtime_error("TensorRT backend not built; use -DDETECTOR_TENSORRT=ON");
#endif
        std::vector<detector::GroundTruth> truth;
        std::vector<detector::ScoredPrediction> predictions;
        fs::path labels_dir = argv[4];
        for (std::size_t i = 0; i < images.size(); ++i) {
            auto image = detector::read_image(images[i].string());
            read_truth(labels_dir / (images[i].stem().string() + ".txt"), static_cast<int>(i),
                       image.width, image.height, truth);
            auto box = detector::letterbox(image, network.input_shape());
            std::vector<detector::Detection> detections;
#ifdef DETECTOR_HAS_CUDA
            if (cuda) detections = detector::decode(network, cuda->forward(box.chw),
                                                     options.confidence, options.nms_iou);
            else
#endif
#ifdef DETECTOR_HAS_TENSORRT
            if (trt) detections = detector::decode(network, trt->forward(box.chw),
                                                    options.confidence, options.nms_iou);
            else
#endif
                detections = network.detect(box.chw, options.backend, options.confidence, options.nms_iou);
            detector::undo_letterbox(detections, box);
            for (const auto& detection : detections) predictions.push_back({static_cast<int>(i), detection});
        }
        auto results = detector::evaluate_ap(truth, predictions, classes, 0.5f);
        std::cout << "images=" << images.size() << " ground_truths=" << truth.size()
                  << " predictions=" << predictions.size() << " limited=" << (options.limit > 0 ? 1 : 0) << '\n';
        for (const auto& result : results) if (result.ground_truths > 0)
            std::cout << "class=" << result.class_id << " gt=" << result.ground_truths
                      << " predictions=" << result.predictions << " AP50=" << std::fixed
                      << std::setprecision(5) << result.average_precision << '\n';
        std::cout << "mAP50_present_classes=" << std::fixed << std::setprecision(5)
                  << detector::mean_ap_present_classes(results) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
