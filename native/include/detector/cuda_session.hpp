#pragma once

#include "detector/network.hpp"
#include <memory>

namespace detector {

#ifdef DETECTOR_HAS_CUDA
// Owns device tensors and convolution parameters across repeated inferences.
// Load weights before construction; do not mutate the Network while it exists.
class CudaSession {
public:
    explicit CudaSession(const Network& network);
    ~CudaSession();
    CudaSession(const CudaSession&) = delete;
    CudaSession& operator=(const CudaSession&) = delete;
    std::vector<Head> forward(const std::vector<float>& input);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace detector
