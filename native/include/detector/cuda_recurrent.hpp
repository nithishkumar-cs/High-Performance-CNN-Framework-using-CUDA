#pragma once

#include "detector/recurrent.hpp"
#include <vector>

namespace detector {

/** Results of one GPU sequence forward pass and full backpropagation through time. */
struct CudaRecurrentResult {
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<float>> input_gradients;
    std::vector<float> parameter_gradients;
};

/** Computes RNN, LSTM, or GRU recurrence and BPTT on CUDA from zero initial state.
 *  Each input and injected hidden gradient is one timestep; the result's parameter
 *  gradient order matches RecurrentCell::parameter_values(). The cell is unchanged.
 *  Throws std::runtime_error for invalid sequence shapes or CUDA failures.
 */
CudaRecurrentResult recurrent_forward_backward_cuda(
    const RecurrentCell& cell, const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& hidden_gradients);

}  // namespace detector
