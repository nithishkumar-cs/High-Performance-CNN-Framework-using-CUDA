#pragma once

#include "detector/network.hpp"
#include <vector>

namespace detector {

// YOLO box coordinates normalized to the network input after letterboxing.
struct Target {
    int class_id = 0;
    float cx = 0;
    float cy = 0;
    float width = 0;
    float height = 0;
};

struct TrainResult {
    float loss = 0;
    std::size_t positives = 0;
    bool updated = false;  // true once this call's minibatch reached batch_size and applied an SGD step
};

// Linear burn-in to `base_lr` over `burn_in` iterations, then `base_lr` scaled by the product of
// `scales` whose matching `steps` entry has been passed (Darknet's `burn_in`/`steps`/`scales` policy).
struct LearningRateSchedule {
    float base_lr = 0.001f;
    int burn_in = 0;
    float burn_in_power = 4.0f;
    std::vector<int> steps;
    std::vector<float> scales;
    float at(int iteration) const;
};

// CPU reference trainer for the cfg layer subset supported by Network, including shortcut
// (residual-add) branches. Convolution weights/biases and, for batch_normalize layers, the
// scale (gamma) and bias (beta) all receive SGD updates; batch-norm statistics are computed
// per sample over its spatial (H*W) positions per channel (this trainer has no literal batched
// tensor, so true cross-sample batch statistics are not computed) and the running mean/variance
// used by inference are updated by exponential moving average on every sample. Gradients from
// `batch_size` consecutive step() calls are accumulated and applied as one averaged SGD update,
// giving minibatch-size semantics without changing the single-sample forward/backward shapes.
// This is still not Darknet's exact loss or augmentation pipeline.
class Trainer {
public:
    explicit Trainer(Network& network, LearningRateSchedule schedule, int batch_size = 1,
                      float weight_decay = 0.0005f, float bn_momentum = 0.99f);
    TrainResult step(const std::vector<float>& input, const std::vector<Target>& targets);
    float current_learning_rate() const { return schedule_.at(iteration_); }

private:
    Network& network_;
    LearningRateSchedule schedule_;
    int batch_size_;
    float weight_decay_;
    float bn_momentum_;
    int iteration_ = 0;
    int accumulated_ = 0;
    std::vector<std::vector<float>> weight_grad_;
    std::vector<std::vector<float>> bias_grad_;   // beta grad when batch_normalize, else conv bias grad
    std::vector<std::vector<float>> scale_grad_;  // gamma grad, batch_normalize layers only
};

}  // namespace detector

