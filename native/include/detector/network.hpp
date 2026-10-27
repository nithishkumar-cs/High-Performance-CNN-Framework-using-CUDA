#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "detector/recurrent.hpp"

namespace detector {

struct Shape {
    int c = 0;
    int h = 0;
    int w = 0;
    std::size_t elements() const;
};

enum class Kind { convolutional, maxpool, route, upsample, yolo, connected, softmax, shortcut, dropout, reorg, region, rnn, lstm, gru };
enum class Backend { cpu, cuda, tensorrt };

struct Layer {
    Kind kind{};
    Shape input;
    Shape output;
    int filters = 0;  // also hidden/output size for rnn/lstm/gru layers (cfg `output=`), reusing
                       // the same field connected layers already use for their output unit count.
    int size = 0;
    int stride = 1;
    int padding = 0;
    bool batch_normalize = false;
    bool leaky = false;
    int classes = 0;
    float ignore_thresh = 1.0f;  // yolo layers only: skip objectness-negative loss for predicted
                                  // boxes overlapping any ground truth by at least this IoU (Darknet's
                                  // near-miss "ignore" rule); 1.0 (unreachable) disables it.
    float dropout_probability = 0.5f;  // dropout layers only: fraction of activations zeroed during
                                        // training (inverted dropout - kept activations scaled by
                                        // 1/(1-p)); a no-op at inference, matching Darknet.
    // region layers only (YOLOv2-style head: all `anchors` apply to every cell, no mask subset -
    // `mask` is populated as the identity [0..anchors.size()) at parse time so the shared
    // yolo/region assignment and decode code can treat both kinds uniformly). `rescore` makes the
    // positive objectness target the predicted box's IoU with its matched truth instead of a flat
    // 1 (Darknet's region_layer.c). The four scale fields default to 1 (neutral for `yolo` layers,
    // which never set them) and weight negative/positive objectness, class and coordinate loss
    // independently, matching Darknet's object_scale/noobject_scale/class_scale/coord_scale.
    bool rescore = false;
    float object_scale = 1.0f;
    float noobject_scale = 1.0f;
    float class_scale = 1.0f;
    float coord_scale = 1.0f;
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
    // Recomputes every layer's input/output Shape for a new network input width/height (Darknet's
    // periodic multiscale resize during training - see train_main.cpp's --multiscale options),
    // leaving every learned parameter (weights, batch-norm scale/mean/variance, biases) untouched.
    // Throws the same structural errors as from_cfg (e.g. a route/shortcut shape mismatch, or a
    // convolution filter exceeding the new input) if the new size is incompatible with this cfg.
    void resize(int width, int height);
    void initialize_random(std::uint32_t seed);
    void save_weights(const std::string& path) const;
    Shape input_shape() const { return input_; }
    const std::vector<Layer>& layers() const { return layers_; }
    // Mutable single-layer access for tests that perturb a trained parameter directly
    // (e.g. a finite-difference gradient check against Trainer's analytic backward).
    Layer& layer(std::size_t index) { return layers_.at(index); }
    // Mutable access to an rnn/lstm/gru layer's live cell (null for every other layer kind) -
    // callable through a const Network since cells_ is mutable (see its declaration below);
    // used by cpu_forward() to advance hidden state one timestep per call.
    RecurrentCell* cell_at(std::size_t index) const {
        return cells_.at(index) ? &*cells_[index] : nullptr;
    }
    std::vector<Head> forward(const std::vector<float>& input, Backend backend) const;
    std::vector<float> classify(const std::vector<float>& input, Backend backend) const;
    std::vector<Detection> detect(const std::vector<float>& input, Backend backend,
                                  float threshold = 0.25f, float iou_threshold = 0.45f) const;
    // Clears every rnn/lstm/gru layer's hidden (and, for lstm, cell) state - call between
    // independent sequences. Each forward()/classify() call is one timestep; state otherwise
    // persists across calls, matching Darknet's own rnn.c streaming usage (and this project's
    // BN running-stat persistence pattern) rather than resetting on every call. A no-op cfg with
    // no recurrent layers.
    void reset_sequence();

private:
    friend class Trainer;
    friend class ClassificationTrainer;
    friend class CudaClassificationTrainer;
    // Lazily built in from_cfg() for every rnn/lstm/gru layer (input/hidden size are fixed then);
    // initialize_random()/load_weights() only overwrite its learned parameters, never resize it.
    // mutable so const forward()/classify() can still advance each cell's persistent hidden
    // state one timestep per call - that state is runtime position in a sequence, not part of
    // the network's logical (weight) value, the same justification batch-norm running stats
    // would use if they were mutated outside Trainer's non-const step().
    mutable std::vector<std::optional<RecurrentCell>> cells_;
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
