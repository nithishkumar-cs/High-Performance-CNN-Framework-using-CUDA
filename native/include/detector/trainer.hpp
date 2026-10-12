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

// Every step() call now applies exactly one SGD update over its samples, so there is no
// partial/pending state to report back (unlike the earlier cross-call gradient accumulation).
struct TrainResult {
    float loss = 0;
    std::size_t positives = 0;
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
// scale (gamma) and bias (beta) all receive SGD updates with classic momentum: a velocity buffer
// per parameter accumulates `velocity = momentum*velocity + averaged_grad(+weight_decay*param for
// weights)`, then `param -= lr*velocity` (matching Darknet's cfg momentum=0.9 default; velocity
// starts at zero so a trainer's very first step() call is unaffected by `momentum`). step() takes
// a true minibatch (one input/target-list per sample): batch-norm mean/variance are computed
// across every sample in the call (and each channel's spatial H*W positions), matching Darknet's
// batched statistics rather than per-sample ones, and the running mean/variance used by inference
// are updated by exponential moving average from those batch statistics once per call. Every
// step() call applies one SGD update averaged over its samples; `batch_size` only bounds how many
// samples a call may hold (a final, smaller partial batch is fine). This is still not Darknet's
// exact loss or augmentation pipeline.
class Trainer {
public:
    explicit Trainer(Network& network, LearningRateSchedule schedule, int batch_size = 1,
                      float weight_decay = 0.0005f, float momentum = 0.9f, float bn_momentum = 0.99f);
    // `inputs.size()` must be between 1 and the constructor's `batch_size` inclusive; `targets`
    // must have the same length, one target list per sample.
    TrainResult step(const std::vector<std::vector<float>>& inputs,
                      const std::vector<std::vector<Target>>& targets);
    float current_learning_rate() const { return schedule_.at(iteration_); }
    // No-op: Trainer's weights already live on `network`. Lets callers that are generic over
    // Trainer and CudaTrainer (which does need this to copy device weights back) call it unconditionally.
    void sync_to_network() {}
    // Darknet-style multiscale resize: Trainer keeps no shape-sized buffers of its own (step()
    // allocates its per-call activation buffers from the network's current shapes), so this is
    // just a forward to Network::resize - lets callers be generic over Trainer and CudaTrainer
    // (which does need to reallocate device buffers) like sync_to_network() above.
    void resize(int width, int height) { network_.resize(width, height); }

private:
    Network& network_;
    LearningRateSchedule schedule_;
    int batch_size_;
    float weight_decay_;
    float momentum_;
    float bn_momentum_;
    int iteration_ = 0;
    std::vector<std::vector<float>> weight_grad_;
    std::vector<std::vector<float>> bias_grad_;   // beta grad when batch_normalize, else conv bias grad
    std::vector<std::vector<float>> scale_grad_;  // gamma grad, batch_normalize layers only
    std::vector<std::vector<float>> weight_velocity_;
    std::vector<std::vector<float>> bias_velocity_;
    std::vector<std::vector<float>> scale_velocity_;
};

}  // namespace detector

