#pragma once

#include "detector/trainer.hpp"
#include <memory>
#include <vector>

namespace detector {

#ifdef DETECTOR_HAS_CUDA
// GPU-resident trainer for the same detector graph and semantics as Trainer (CPU): shortcut,
// batch-norm, route, upsample and YOLO heads, with true minibatch batch-norm statistics (mean/
// variance computed across every sample passed to step(), not per sample) and the same
// LearningRateSchedule. Convolution/batch-norm/shortcut/route/upsample forward and backward, and
// the SGD update (with the same classic-momentum formula as Trainer: per-parameter velocity
// buffers, zero-initialized, persisting across step() calls), run as CUDA kernels; only the YOLO
// heads' small output/gradient tensors are copied to/from the host each step, where
// yolo_loss_and_gradient (shared with Trainer) does the anchor assignment and loss so both
// backends use identical math. Device buffers are sized for `batch_size` samples; step() may be
// called with between 1 and `batch_size` samples (a smaller final partial batch is fine). Call
// sync_to_network() before Network::save_weights() to copy the learned weights back to host.
class CudaTrainer {
public:
    // `initial_iteration` seeds the step counter the LearningRateSchedule is keyed on - see
    // Trainer's matching parameter for the resume rationale.
    explicit CudaTrainer(Network& network, LearningRateSchedule schedule, int batch_size = 1,
                           float weight_decay = 0.0005f, float momentum = 0.9f, float bn_momentum = 0.99f,
                           int initial_iteration = 0, int device_id = -1);
    ~CudaTrainer();
    CudaTrainer(const CudaTrainer&) = delete;
    CudaTrainer& operator=(const CudaTrainer&) = delete;
    TrainResult step(const std::vector<std::vector<float>>& inputs,
                      const std::vector<std::vector<Target>>& targets);
    float current_learning_rate() const;
    // Current step count, matching Trainer::iteration() (see there for the resume rationale).
    int iteration() const;
    void sync_to_network();
    // Darknet-style multiscale resize: changes the network's (and this trainer's device buffers')
    // input width/height between step() calls, preserving every learned parameter and all
    // momentum/batch-norm running-statistics state. See cuda_training.cu's Impl::resize.
    void resize(int width, int height);
    // Average learned parameters, momentum and running BN state after independent local steps.
    // Weights are sample counts, so uneven final minibatches contribute proportionally.
    static void average_state(const std::vector<CudaTrainer*>& trainers,
                              const std::vector<std::size_t>& sample_counts);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int device_id_ = -1;
};

// Replicated data-parallel detector trainer. Each selected CUDA device owns a model copy and
// processes a disjoint shard concurrently; every global step averages model/optimizer state.
class MultiCudaTrainer {
public:
    MultiCudaTrainer(Network& network, LearningRateSchedule schedule, int batch_size,
                     std::vector<int> device_ids, float weight_decay = 0.0005f,
                     float momentum = 0.9f, float bn_momentum = 0.99f, int initial_iteration = 0);
    TrainResult step(const std::vector<std::vector<float>>& inputs,
                     const std::vector<std::vector<Target>>& targets);
    float current_learning_rate() const;
    int iteration() const;
    void sync_to_network();
    void resize(int width, int height);

private:
    Network& network_;
    std::vector<Network> replicas_;
    std::vector<std::unique_ptr<CudaTrainer>> trainers_;
    int batch_size_;
};
#endif

}  // namespace detector
