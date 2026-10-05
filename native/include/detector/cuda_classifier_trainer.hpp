#pragma once

#include "detector/classifier_trainer.hpp"
#include <memory>

namespace detector {

#ifdef DETECTOR_HAS_CUDA
// Batch-one CUDA SGD for sequential [convolutional], [maxpool], [connected]
// layers and final [softmax], without trainable batch normalization.
// sync_to_network() copies learned device parameters into Network for saving.
class CudaClassificationTrainer {
public:
    CudaClassificationTrainer(Network& network, float learning_rate, float weight_decay = 0.0f);
    ~CudaClassificationTrainer();
    CudaClassificationTrainer(const CudaClassificationTrainer&) = delete;
    CudaClassificationTrainer& operator=(const CudaClassificationTrainer&) = delete;
    ClassificationResult step(const std::vector<float>& input, int class_id);
    void sync_to_network();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace detector
