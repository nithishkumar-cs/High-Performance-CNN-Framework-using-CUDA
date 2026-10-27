#pragma once

#include "detector/recurrent.hpp"
#include <vector>

namespace detector {

/** Results of one GPU sequence forward pass and full backpropagation through time. */
struct CudaRecurrentResult {
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<float>> input_gradients;
    std::vector<float> parameter_gradients;
    std::vector<float> final_hidden;
    std::vector<float> final_cell;
};

/** Computes RNN, LSTM, or GRU recurrence and BPTT on CUDA.
 *  Each input and injected hidden gradient is one timestep; the result's parameter
 *  gradient order matches RecurrentCell::parameter_values(). The cell is unchanged.
 *  Empty initial states mean zero; nonempty states must match hidden_size().
 *  Throws std::runtime_error for invalid sequence shapes or CUDA failures.
 */
CudaRecurrentResult recurrent_forward_backward_cuda(
    const RecurrentCell& cell, const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& hidden_gradients,
    const std::vector<float>& initial_hidden = {}, const std::vector<float>& initial_cell = {});

/** Runs recurrence without BPTT, preserving the supplied initial state. */
CudaRecurrentResult recurrent_forward_cuda(
    const RecurrentCell& cell, const std::vector<std::vector<float>>& inputs,
    const std::vector<float>& initial_hidden = {}, const std::vector<float>& initial_cell = {});

}  // namespace detector
