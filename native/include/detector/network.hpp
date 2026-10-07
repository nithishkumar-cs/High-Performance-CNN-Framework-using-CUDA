#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace detector {

struct Shape {
    int c = 0;
    int h = 0;
    int w = 0;
    std::size_t elements() const;
};

enum class Kind { convolutional, maxpool, route, upsample, yolo, connected, softmax, shortcut };
enum class Backend { cpu, cuda, tensorrt };

struct Layer {
    Kind kind{};
    Shape input;
    Shape output;
    int filters = 0;
    int size = 0;
    int stride = 1;
    int padding = 0;
    bool batch_normalize = false;
    bool leaky = false;
    int classes = 0;
    float ignore_thresh = 1.0f;  // yolo layers only: skip objectness-negative loss for predicted
                                  // boxes overlapping any ground truth by at least this IoU (Darknet's
                                  // near-miss "ignore" rule); 1.0 (unreachable) disables it.
    std::vector<int> sources;
    std::vector<int> mask;
    std::vector<std::pair<float, float>> anchors;
    std::vector<float> biases;
    std::vector<float> scales;
    std::vector<float> means;
    std::vector<float> variances;
    std::vector<float> weights;
};

struct Head {
    std::size_t layer = 0;
    std::vector<float> values;
};

struct Detection {
    int class_id = 0;
    float score = 0;
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
};

class Network {
public:
    static Network from_cfg(const std::string& path);
    // layer_limit < 0 (default) loads every layer and rejects trailing/short data (full-file load).
    // layer_limit >= 0 loads only layers [0, layer_limit) from `path` and leaves the rest of the
    // network untouched (e.g. after initialize_random()); this is Darknet's `partial N` transfer -
    // seeding a pretrained backbone whose later (shape-diverging) head layers are skipped - without
    // needing an intermediate truncated weights file, since it simply stops reading `path` early.
    void load_weights(const std::string& path, int layer_limit = -1);
    void initialize_random(std::uint32_t seed);
    void save_weights(const std::string& path) const;
    Shape input_shape() const { return input_; }
    const std::vector<Layer>& layers() const { return layers_; }
    // Mutable single-layer access for tests that perturb a trained parameter directly
    // (e.g. a finite-difference gradient check against Trainer's analytic backward).
    Layer& layer(std::size_t index) { return layers_.at(index); }
    std::vector<Head> forward(const std::vector<float>& input, Backend backend) const;
    std::vector<float> classify(const std::vector<float>& input, Backend backend) const;
    std::vector<Detection> detect(const std::vector<float>& input, Backend backend,
                                  float threshold = 0.25f, float iou_threshold = 0.45f) const;

private:
    friend class Trainer;
    friend class ClassificationTrainer;
    friend class CudaClassificationTrainer;
    Shape input_;
    std::vector<Layer> layers_;
};

std::vector<Detection> decode(const Network& network, const std::vector<Head>& heads,
                              float threshold, float iou_threshold);
std::vector<Detection> non_max_suppression(std::vector<Detection> boxes, float iou_threshold);

#ifdef DETECTOR_HAS_CUDA
std::vector<Head> forward_cuda(const Network& network, const std::vector<float>& input);
#endif

}  // namespace detector
