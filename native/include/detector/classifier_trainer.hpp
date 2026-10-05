#pragma once

#include "detector/network.hpp"

namespace detector {

struct ClassificationResult {
    float loss = 0;
    bool correct = false;
};

// Reference SGD trainer for [convolutional], [maxpool], [connected], [softmax]
// classifiers. Batch size is one; convolution batch-normalization statistics
// and scales are fixed. This is not a general CUDA/autograd trainer yet.
class ClassificationTrainer {
public:
    ClassificationTrainer(Network& network, float learning_rate, float weight_decay = 0.0f);
    ClassificationResult step(const std::vector<float>& input, int class_id);

private:
    Network& network_;
    float learning_rate_;
    float weight_decay_;
};

}  // namespace detector
