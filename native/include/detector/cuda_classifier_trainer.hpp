#pragma once

#include "detector/classifier_trainer.hpp"
#include <memory>
#include <optional>

namespace detector {

#ifdef DETECTOR_HAS_CUDA
// CUDA SGD for classifiers with a final [softmax], including one recurrent layer
// trained in bounded time chunks. Trainable batch normalization is unsupported.
class CudaClassificationTrainer {
public:
    /** Uploads classifier parameters and validates supported layer kinds. */
    CudaClassificationTrainer(Network& network, float learning_rate, float weight_decay = 0.0f,
                              std::optional<VectorLoss> vector_loss = std::nullopt);
    /** Releases device parameters. */
    ~CudaClassificationTrainer();
    CudaClassificationTrainer(const CudaClassificationTrainer&) = delete;
    CudaClassificationTrainer& operator=(const CudaClassificationTrainer&) = delete;
    /** Trains one independent sample for graphs without a recurrent layer. */
    ClassificationResult step(const std::vector<float>& input, int class_id);
    /** One mse / sigmoid-BCE / pixel-softmax-CE step; requires a constructor vector_loss and a
     *  cfg without a final [softmax] (the GPU counterpart of VectorTrainer::step). */
    VectorResult step_vector(const std::vector<float>& input, const std::vector<float>& targets);
    /** Trains a consecutive sequence chunk with truncated BPTT. */
    std::vector<ClassificationResult> train_sequence(
        const std::vector<std::vector<float>>& inputs, const std::vector<int>& class_ids);
    /** Copies learned CUDA parameters into Network for saving or CPU inference. */
    void sync_to_network();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace detector
