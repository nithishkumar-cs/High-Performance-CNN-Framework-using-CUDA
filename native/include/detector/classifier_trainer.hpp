#pragma once

#include "detector/network.hpp"

namespace detector {

struct ClassificationResult {
    float loss = 0;
    bool correct = false;
};

// Reference SGD trainer for [convolutional], [maxpool], [connected], [softmax] classifiers,
// optionally with one [rnn]/[lstm]/[gru] layer for sequence classification (req-32). Batch size
// is one timestep for step(); convolution batch-normalization statistics and scales are fixed.
// This is not a general CUDA/autograd trainer yet.
class ClassificationTrainer {
public:
    ClassificationTrainer(Network& network, float learning_rate, float weight_decay = 0.0f);
    // A cfg with no recurrent layer: one independent example, immediate backward+update.
    ClassificationResult step(const std::vector<float>& input, int class_id);
    // A cfg WITH a recurrent layer: trains one chunk of `inputs.size()` consecutive timesteps of
    // the same sequence (Darknet rnn.c's truncated-BPTT `time_steps`-per-batch pattern) - every
    // other layer kind's weight gradient is summed across the chunk into one averaged SGD
    // update (matching Trainer::step()'s minibatch averaging); the recurrent layer gets exactly
    // one backward_through_time() call over the whole chunk, then its own sgd_step(). Hidden
    // state persists into the next train_sequence()/step() call unless Network::reset_sequence()
    // is called first - call that before the first chunk of each independent sequence. Requires
    // the cfg to contain exactly one rnn/lstm/gru layer (use step() otherwise).
    std::vector<ClassificationResult> train_sequence(const std::vector<std::vector<float>>& inputs,
                                                       const std::vector<int>& class_ids);

private:
    Network& network_;
    float learning_rate_;
    float weight_decay_;
};

}  // namespace detector
