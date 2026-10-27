#include "detector/cuda_recurrent.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace detector {
namespace {

constexpr int kThreads = 128;

/** Reports CUDA allocation, transfer, or launch failures with the operation name. */
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

/** Owns a contiguous device float array, including cleanup on exceptional paths. */
class DeviceFloats {
public:
    /** Allocates count device floats. */
    explicit DeviceFloats(std::size_t count) : count_(count) {
        check(cudaMalloc(&data_, count * sizeof(float)), "cudaMalloc recurrent buffer");
    }
    /** Releases the device allocation. */
    ~DeviceFloats() { if (data_) cudaFree(data_); }
    DeviceFloats(const DeviceFloats&) = delete;
    DeviceFloats& operator=(const DeviceFloats&) = delete;
    /** Returns the owned device pointer. */
    float* get() const { return static_cast<float*>(data_); }
    /** Exchanges buffer ownership without a device copy. */
    void swap(DeviceFloats& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(count_, other.count_);
    }
    /** Clears the entire device buffer. */
    void clear() { check(cudaMemset(data_, 0, count_ * sizeof(float)), "cudaMemset recurrent buffer"); }
    /** Copies exactly count_ floats from the host. */
    void upload(const float* source) {
        check(cudaMemcpy(data_, source, count_ * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy recurrent H2D");
    }
    /** Copies exactly count_ floats to the host. */
    void download(float* destination) const {
        check(cudaMemcpy(destination, data_, count_ * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy recurrent D2H");
    }
private:
    void* data_ = nullptr;
    std::size_t count_ = 0;
};

/** Stable logistic activation for CUDA recurrent gates. */
__device__ float sigmoid(float value) {
    return value >= 0 ? 1.0f / (1.0f + expf(-value)) : expf(value) / (1.0f + expf(value));
}

/** Computes independent gate activations; GRU candidate waits for the reset gate. */
__global__ void forward_gates(const float* params, const float* x, const float* hprev,
                              float* gates, int input_size, int hidden_size, int gate_count, int kind) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= gate_count * hidden_size) return;
    int gate = pos / hidden_size, row = pos % hidden_size;
    if (kind == 2 && gate == 2) return;
    int stride = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
    const float* wx = params + gate * stride + row * input_size;
    const float* wh = params + gate * stride + hidden_size * input_size + row * hidden_size;
    float sum = params[gate * stride + hidden_size * input_size + hidden_size * hidden_size + row];
    for (int i = 0; i < input_size; ++i) sum += wx[i] * x[i];
    for (int i = 0; i < hidden_size; ++i) sum += wh[i] * hprev[i];
    gates[pos] = kind == 0 || (kind == 1 && gate == 3) ? tanhf(sum) : sigmoid(sum);
}

/** Updates plain RNN or LSTM hidden and cell state from activated gates. */
__global__ void forward_state(const float* gates, const float* cprev, float* h, float* c,
                              int hidden_size, int kind) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= hidden_size) return;
    if (kind == 0) { h[k] = gates[k]; return; }
    float cell = gates[hidden_size + k] * cprev[k] + gates[k] * gates[3 * hidden_size + k];
    c[k] = cell;
    h[k] = gates[2 * hidden_size + k] * tanhf(cell);
}

/** Computes the GRU candidate after the reset gate, then updates hidden state. */
__global__ void forward_gru_candidate(const float* params, const float* x, const float* hprev,
                                      float* gates, float* h, int input_size, int hidden_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= hidden_size) return;
    int stride = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
    const float* wx = params + 2 * stride + row * input_size;
    const float* wh = params + 2 * stride + hidden_size * input_size + row * hidden_size;
    float sum = params[2 * stride + hidden_size * input_size + hidden_size * hidden_size + row];
    for (int i = 0; i < input_size; ++i) sum += wx[i] * x[i];
    for (int i = 0; i < hidden_size; ++i) sum += wh[i] * (gates[hidden_size + i] * hprev[i]);
    float candidate = tanhf(sum), update = gates[row];
    gates[2 * hidden_size + row] = candidate;
    h[row] = (1.0f - update) * candidate + update * hprev[row];
}

/** Differentiates each output gate and propagates LSTM cell state one step. */
__global__ void backward_gates(const float* direct, const float* hcarry, const float* ccarry,
                               const float* hprev, const float* cprev, const float* cell,
                               const float* gates, float* deltas, float* cprev_gradient,
                               int hidden_size, int kind) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= hidden_size) return;
    float dh = direct[k] + hcarry[k];
    if (kind == 0) {
        float output = gates[k];
        deltas[k] = dh * (1.0f - output * output);
    } else if (kind == 1) {
        float input = gates[k], forget = gates[hidden_size + k];
        float output = gates[2 * hidden_size + k], candidate = gates[3 * hidden_size + k];
        float tanh_cell = tanhf(cell[k]);
        float dc = ccarry[k] + dh * output * (1.0f - tanh_cell * tanh_cell);
        deltas[k] = dc * candidate * input * (1.0f - input);
        deltas[hidden_size + k] = dc * cprev[k] * forget * (1.0f - forget);
        deltas[2 * hidden_size + k] = dh * tanh_cell * output * (1.0f - output);
        deltas[3 * hidden_size + k] = dc * input * (1.0f - candidate * candidate);
        cprev_gradient[k] = dc * forget;
    } else {
        float update = gates[k], candidate = gates[2 * hidden_size + k];
        deltas[k] = dh * (hprev[k] - candidate) * update * (1.0f - update);
        deltas[2 * hidden_size + k] = dh * (1.0f - update) * (1.0f - candidate * candidate);
    }
}

/** Differentiates the GRU reset gate through its candidate recurrent matrix. */
__global__ void backward_gru_reset(const float* params, float* deltas, const float* gates,
                                   const float* hprev, float* reset_hidden_gradient,
                                   int input_size, int hidden_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= hidden_size) return;
    int stride = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
    const float* wh = params + 2 * stride + hidden_size * input_size;
    float sum = 0.0f;
    for (int row = 0; row < hidden_size; ++row)
        sum += wh[row * hidden_size + k] * deltas[2 * hidden_size + row];
    reset_hidden_gradient[k] = sum;
    float reset = gates[hidden_size + k];
    deltas[hidden_size + k] = sum * hprev[k] * reset * (1.0f - reset);
}

/** Accumulates per-timestep parameter gradients with one owner per parameter. */
__global__ void backward_parameters(const float* x, const float* hprev, const float* gates,
                                    const float* deltas, float* gradients,
                                    int input_size, int hidden_size, int gate_count, int kind) {
    int stride = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= gate_count * stride) return;
    int gate = pos / stride, item = pos % stride;
    int row, col;
    float value;
    if (item < hidden_size * input_size) {
        row = item / input_size; col = item % input_size;
        value = x[col];
    } else if (item < hidden_size * input_size + hidden_size * hidden_size) {
        item -= hidden_size * input_size;
        row = item / hidden_size; col = item % hidden_size;
        value = hprev[col] * (kind == 2 && gate == 2 ? gates[hidden_size + col] : 1.0f);
    } else {
        row = item - hidden_size * input_size - hidden_size * hidden_size;
        value = 1.0f;
    }
    gradients[pos] += deltas[gate * hidden_size + row] * value;
}

/** Propagates gradients to the timestep input and preceding hidden state. */
__global__ void backward_inputs(const float* params, const float* deltas, const float* direct,
                                const float* hcarry, const float* gates, const float* reset_hidden_gradient,
                                float* dx, float* hprev_gradient, int input_size, int hidden_size,
                                int gate_count, int kind) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= (input_size > hidden_size ? input_size : hidden_size)) return;
    int stride = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
    if (pos < input_size) {
        float sum = 0.0f;
        for (int gate = 0; gate < gate_count; ++gate)
            for (int row = 0; row < hidden_size; ++row)
                sum += params[gate * stride + row * input_size + pos] * deltas[gate * hidden_size + row];
        dx[pos] = sum;
    }
    if (pos < hidden_size) {
        float sum = 0.0f;
        for (int gate = 0; gate < gate_count; ++gate) {
            if (kind == 2 && gate == 2) continue;
            int base = gate * stride + hidden_size * input_size;
            for (int row = 0; row < hidden_size; ++row)
                sum += params[base + row * hidden_size + pos] * deltas[gate * hidden_size + row];
        }
        if (kind == 2) sum += (direct[pos] + hcarry[pos]) * gates[pos]
                                 + reset_hidden_gradient[pos] * gates[hidden_size + pos];
        hprev_gradient[pos] = sum;
    }
}

}  // namespace

CudaRecurrentResult recurrent_forward_backward_cuda(
    const RecurrentCell& cell, const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& hidden_gradients) {
    int timesteps = static_cast<int>(inputs.size());
    int input_size = cell.input_size(), hidden_size = cell.hidden_size();
    if (timesteps < 1 || hidden_gradients.size() != inputs.size() || input_size < 1 || hidden_size < 1)
        throw std::runtime_error("invalid CUDA recurrent sequence shape");
    for (int t = 0; t < timesteps; ++t)
        if (inputs[t].size() != static_cast<std::size_t>(input_size) ||
            hidden_gradients[t].size() != static_cast<std::size_t>(hidden_size))
            throw std::runtime_error("CUDA recurrent timestep size mismatch");
    int kind = static_cast<int>(cell.kind());
    int gate_count = kind == 0 ? 1 : kind == 1 ? 4 : 3;
    int stride = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
    std::vector<float> flat_inputs, flat_direct;
    flat_inputs.reserve(static_cast<std::size_t>(timesteps) * input_size);
    flat_direct.reserve(static_cast<std::size_t>(timesteps) * hidden_size);
    for (int t = 0; t < timesteps; ++t) {
        flat_inputs.insert(flat_inputs.end(), inputs[t].begin(), inputs[t].end());
        flat_direct.insert(flat_direct.end(), hidden_gradients[t].begin(), hidden_gradients[t].end());
    }
    auto values = cell.parameter_values();
    DeviceFloats params(values.size()), x(flat_inputs.size()), direct(flat_direct.size());
    DeviceFloats h(static_cast<std::size_t>(timesteps + 1) * hidden_size);
    DeviceFloats c(static_cast<std::size_t>(timesteps + 1) * hidden_size);
    DeviceFloats gates(static_cast<std::size_t>(timesteps) * gate_count * hidden_size);
    DeviceFloats dx(flat_inputs.size()), dparams(values.size());
    DeviceFloats delta(static_cast<std::size_t>(gate_count) * hidden_size);
    DeviceFloats hcarry(hidden_size), hnext(hidden_size), ccarry(hidden_size), cnext(hidden_size);
    DeviceFloats reset_gradient(hidden_size);
    params.upload(values.data()); x.upload(flat_inputs.data()); direct.upload(flat_direct.data());
    h.clear(); c.clear(); dparams.clear(); hcarry.clear(); ccarry.clear();
    int gate_blocks = (gate_count * hidden_size + kThreads - 1) / kThreads;
    int hidden_blocks = (hidden_size + kThreads - 1) / kThreads;
    for (int t = 0; t < timesteps; ++t) {
        float* g = gates.get() + static_cast<std::size_t>(t) * gate_count * hidden_size;
        const float* prev_h = h.get() + static_cast<std::size_t>(t) * hidden_size;
        forward_gates<<<gate_blocks, kThreads>>>(params.get(), x.get() + t * input_size, prev_h,
            g, input_size, hidden_size, gate_count, kind);
        check(cudaGetLastError(), "CUDA recurrent forward gates");
        if (kind == 2) {
            forward_gru_candidate<<<hidden_blocks, kThreads>>>(params.get(), x.get() + t * input_size,
                prev_h, g, h.get() + static_cast<std::size_t>(t + 1) * hidden_size, input_size, hidden_size);
        } else {
            forward_state<<<hidden_blocks, kThreads>>>(g, c.get() + static_cast<std::size_t>(t) * hidden_size,
                h.get() + static_cast<std::size_t>(t + 1) * hidden_size,
                c.get() + static_cast<std::size_t>(t + 1) * hidden_size, hidden_size, kind);
        }
        check(cudaGetLastError(), "CUDA recurrent forward state");
    }
    int parameter_blocks = (gate_count * stride + kThreads - 1) / kThreads;
    int input_blocks = (std::max(input_size, hidden_size) + kThreads - 1) / kThreads;
    for (int t = timesteps - 1; t >= 0; --t) {
        const float* prev_h = h.get() + static_cast<std::size_t>(t) * hidden_size;
        const float* g = gates.get() + static_cast<std::size_t>(t) * gate_count * hidden_size;
        const float* dh = direct.get() + static_cast<std::size_t>(t) * hidden_size;
        backward_gates<<<hidden_blocks, kThreads>>>(dh, hcarry.get(), ccarry.get(), prev_h,
            c.get() + static_cast<std::size_t>(t) * hidden_size,
            c.get() + static_cast<std::size_t>(t + 1) * hidden_size, g, delta.get(), cnext.get(),
            hidden_size, kind);
        check(cudaGetLastError(), "CUDA recurrent backward gates");
        if (kind == 2) {
            backward_gru_reset<<<hidden_blocks, kThreads>>>(params.get(), delta.get(), g, prev_h,
                reset_gradient.get(), input_size, hidden_size);
            check(cudaGetLastError(), "CUDA recurrent reset gradient");
        }
        backward_parameters<<<parameter_blocks, kThreads>>>(x.get() + t * input_size, prev_h, g,
            delta.get(), dparams.get(), input_size, hidden_size, gate_count, kind);
        check(cudaGetLastError(), "CUDA recurrent parameter gradient");
        backward_inputs<<<input_blocks, kThreads>>>(params.get(), delta.get(), dh, hcarry.get(), g,
            reset_gradient.get(), dx.get() + t * input_size, hnext.get(), input_size, hidden_size,
            gate_count, kind);
        check(cudaGetLastError(), "CUDA recurrent input gradient");
        hcarry.swap(hnext);
        if (kind == 1) ccarry.swap(cnext);
    }
    std::vector<float> flat_h(static_cast<std::size_t>(timesteps + 1) * hidden_size);
    std::vector<float> flat_dx(flat_inputs.size()), flat_dparams(values.size());
    h.download(flat_h.data()); dx.download(flat_dx.data()); dparams.download(flat_dparams.data());
    CudaRecurrentResult result;
    result.parameter_gradients = std::move(flat_dparams);
    for (int t = 0; t < timesteps; ++t) {
        result.outputs.emplace_back(flat_h.begin() + static_cast<std::size_t>(t + 1) * hidden_size,
                                    flat_h.begin() + static_cast<std::size_t>(t + 2) * hidden_size);
        result.input_gradients.emplace_back(flat_dx.begin() + static_cast<std::size_t>(t) * input_size,
                                            flat_dx.begin() + static_cast<std::size_t>(t + 1) * input_size);
    }
    return result;
}

}  // namespace detector
