#pragma once

#include "detector/trainer.hpp"
#include <memory>

namespace detector {

#ifdef DETECTOR_HAS_CUDA
// GPU-resident trainer for the same detector graph and semantics as Trainer (CPU): shortcut,
// batch-norm, route, upsample and YOLO heads, with minibatch gradient accumulation and the same
// LearningRateSchedule. Convolution/batch-norm/shortcut/route/upsample forward and backward, and
// the SGD update, run as CUDA kernels; only the YOLO heads' small output/gradient tensors are
// copied to/from the host each step, where yolo_loss_and_gradient (shared with Trainer) does the
// anchor assignment and loss so both backends use identical math. Call sync_to_network() before
// Network::save_weights() to copy the learned weights back to host.
class CudaTrainer {
public:
    explicit CudaTrainer(Network& network, LearningRateSchedule schedule, int batch_size = 1,
                          float weight_decay = 0.0005f, float bn_momentum = 0.99f);
    ~CudaTrainer();
    CudaTrainer(const CudaTrainer&) = delete;
    CudaTrainer& operator=(const CudaTrainer&) = delete;
    TrainResult step(const std::vector<float>& input, const std::vector<Target>& targets);
    float current_learning_rate() const;
    void sync_to_network();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace detector
