#include "detector/image.hpp"
#include "detector/network.hpp"
#ifdef DETECTOR_HAS_CUDA
#include "detector/cuda_session.hpp"
#endif
#ifdef DETECTOR_HAS_TENSORRT
#include "detector/trt_session.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace {

struct Options {
    detector::Backend backend = detector::Backend::cpu;
    float threshold = 0.25f;
    float iou = 0.45f;
    int warmup = 3;
    int iterations = 20;
    std::string names;
    std::string scope = "pipeline";
};

using Predict = std::function<std::vector<detector::Detection>(const std::vector<float>&)>;
using Forward = std::function<std::vector<detector::Head>(const std::vector<float>&)>;

void usage() {
    std::cerr << "Usage:\n"
              << "  cuda_detector inspect <network.cfg>\n"
              << "  cuda_detector detect <network.cfg> <model.weights|model.engine> <image> [options]\n"
              << "  cuda_detector benchmark <network.cfg> <model.weights|model.engine> <image> [options]\n"
              << "Options: --backend cpu|cuda|tensorrt --threshold 0.25 --iou 0.45 --names classes.txt\n"
              << "Benchmark options: --warmup 3 --iterations 20 --scope pipeline|network\n";
}

Options parse_options(int argc, char** argv, int start) {
    Options options;
    for (int i = start; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("option requires a value: " + std::string(argv[i]));
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--backend") {
            if (value == "cpu") options.backend = detector::Backend::cpu;
            else if (value == "cuda") options.backend = detector::Backend::cuda;
            else if (value == "tensorrt") options.backend = detector::Backend::tensorrt;
            else throw std::runtime_error("backend must be cpu, cuda, or tensorrt");
        } else if (key == "--threshold") options.threshold = std::stof(value);
        else if (key == "--iou") options.iou = std::stof(value);
        else if (key == "--warmup") options.warmup = std::stoi(value);
        else if (key == "--iterations") options.iterations = std::stoi(value);
        else if (key == "--names") options.names = value;
        else if (key == "--scope") options.scope = value;
        else throw std::runtime_error("unknown option: " + key);
    }
    if (!(options.threshold >= 0 && options.threshold <= 1 && options.iou >= 0 && options.iou <= 1))
        throw std::runtime_error("threshold and IoU must be in [0,1]");
    if (options.warmup < 0 || options.iterations < 1 || options.iterations > 100000)
        throw std::runtime_error("invalid benchmark iteration counts");
    if (options.scope != "pipeline" && options.scope != "network")
        throw std::runtime_error("scope must be pipeline or network");
    return options;
}

std::vector<std::string> read_names(const std::string& path) {
    std::vector<std::string> names;
    if (path.empty()) return names;
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open class names: " + path);
    std::string line;
    while (std::getline(file, line)) if (!line.empty()) names.push_back(line);
    return names;
}

/** Names a layer kind for inspect output. */
const char* kind_name(detector::Kind kind) {
    switch (kind) {
    case detector::Kind::convolutional: return "conv";
    case detector::Kind::maxpool: return "maxpool";
    case detector::Kind::avgpool: return "avgpool";
    case detector::Kind::route: return "route";
    case detector::Kind::upsample: return "upsample";
    case detector::Kind::yolo: return "yolo";
    case detector::Kind::connected: return "connected";
    case detector::Kind::softmax: return "softmax";
    case detector::Kind::shortcut: return "shortcut";
    case detector::Kind::dropout: return "dropout";
    case detector::Kind::reorg: return "reorg";
    case detector::Kind::region: return "region";
    case detector::Kind::rnn: return "rnn";
    case detector::Kind::lstm: return "lstm";
    case detector::Kind::gru: return "gru";
    case detector::Kind::l2norm: return "l2norm";
    }
    return "unknown";
}

void inspect(const detector::Network& net) {
    auto input = net.input_shape();
    std::cout << "input " << input.c << 'x' << input.h << 'x' << input.w << '\n';
    for (std::size_t i = 0; i < net.layers().size(); ++i) {
        const auto& layer = net.layers()[i];
        std::cout << std::setw(3) << i << ' ' << std::setw(8) << kind_name(layer.kind)
                  << "  " << layer.input.c << 'x' << layer.input.h << 'x' << layer.input.w
                  << " -> " << layer.output.c << 'x' << layer.output.h << 'x' << layer.output.w << '\n';
    }
}

std::vector<detector::Detection> run(const detector::Network& net, const detector::Image& image,
                                     const Predict& predict) {
    auto transform = detector::letterbox(image, net.input_shape());
    auto boxes = predict(transform.chw);
    detector::undo_letterbox(boxes, transform);
    return boxes;
}

void benchmark(const detector::Network& net, const detector::Image& image,
               const Options& options, const Predict& predict, const Forward& forward) {
    auto transform = detector::letterbox(image, net.input_shape());
    std::size_t detections = run(net, image, predict).size();
    auto timed_run = [&]() {
        if (options.scope == "network") forward(transform.chw);
        else detections = run(net, image, predict).size();
    };
    for (int i = 0; i < options.warmup; ++i) timed_run();
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(options.iterations));
    for (int i = 0; i < options.iterations; ++i) {
        auto begin = std::chrono::steady_clock::now();
        timed_run();
        auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    std::sort(times.begin(), times.end());
    auto percentile = [&](double fraction) { return times[static_cast<std::size_t>((times.size() - 1) * fraction)]; };
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << std::fixed << std::setprecision(3)
              << "backend=" << (options.backend == detector::Backend::tensorrt ? "tensorrt" :
                                 options.backend == detector::Backend::cuda ? "cuda" : "cpu")
              << " iterations=" << times.size() << " detections=" << detections
              << " mean_ms=" << mean << " p50_ms=" << percentile(0.50)
              << " p95_ms=" << percentile(0.95) << " min_ms=" << times.front()
              << " max_ms=" << times.back() << '\n';
    if (options.scope == "network")
        std::cout << "scope=network_forward (preprocessing, decode, NMS, file I/O, CUDA setup excluded)\n";
    else
        std::cout << "scope=letterbox+network+decode+NMS+unletterbox (image file I/O and CUDA setup excluded)\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) { usage(); return 2; }
        std::string command = argv[1];
        if (command == "inspect") {
            if (argc != 3) { usage(); return 2; }
            inspect(detector::Network::from_cfg(argv[2]));
            return 0;
        }
        if (command != "detect" && command != "benchmark") { usage(); return 2; }
        if (argc < 5) { usage(); return 2; }
        auto options = parse_options(argc, argv, 5);
        auto net = detector::Network::from_cfg(argv[2]);
        if (options.backend != detector::Backend::tensorrt) net.load_weights(argv[3]);
        auto image = detector::read_image(argv[4]);
#ifdef DETECTOR_HAS_CUDA
        std::unique_ptr<detector::CudaSession> cuda;
        if (options.backend == detector::Backend::cuda)
            cuda = std::make_unique<detector::CudaSession>(net);
#endif
#ifdef DETECTOR_HAS_TENSORRT
        std::unique_ptr<detector::TrtSession> trt;
        if (options.backend == detector::Backend::tensorrt)
            trt = std::make_unique<detector::TrtSession>(net, argv[3]);
#else
        if (options.backend == detector::Backend::tensorrt)
            throw std::runtime_error("TensorRT backend not built; use -DDETECTOR_TENSORRT=ON");
#endif
        Forward forward = [&](const std::vector<float>& input) {
#ifdef DETECTOR_HAS_CUDA
            if (cuda) return cuda->forward(input);
#endif
#ifdef DETECTOR_HAS_TENSORRT
            if (trt) return trt->forward(input);
#endif
            return net.forward(input, options.backend);
        };
        Predict predict = [&](const std::vector<float>& input) {
            return detector::decode(net, forward(input), options.threshold, options.iou);
        };
        if (command == "benchmark") { benchmark(net, image, options, predict, forward); return 0; }
        auto names = read_names(options.names);
        auto boxes = run(net, image, predict);
        for (const auto& box : boxes) {
            std::string name = box.class_id < static_cast<int>(names.size()) ? names[static_cast<std::size_t>(box.class_id)]
                                                                          : std::to_string(box.class_id);
            std::cout << name << ' ' << std::fixed << std::setprecision(4) << box.score << ' '
                      << box.x1 << ' ' << box.y1 << ' ' << box.x2 << ' ' << box.y2 << '\n';
        }
        std::cerr << "detections=" << boxes.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
