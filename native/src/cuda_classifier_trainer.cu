#include "detector/cuda_classifier_trainer.hpp"
#include "detector/cuda_recurrent.hpp"
#ifdef DETECTOR_HAS_CUDA
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace detector {
namespace {

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

struct DeviceMemory {
    std::vector<void*> allocations;
    ~DeviceMemory() { for (void* pointer : allocations) cudaFree(pointer); }
    float* allocate(std::size_t count) {
        float* pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(float)), "cudaMalloc");
        allocations.push_back(pointer);
        return pointer;
    }
    int* allocate_int(std::size_t count) {
        int* pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(int)), "cudaMalloc indices");
        allocations.push_back(pointer);
        return pointer;
    }
    float* upload(const std::vector<float>& host) {
        float* pointer = allocate(host.size());
        check(cudaMemcpy(pointer, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy weights H2D");
        return pointer;
    }
};

__global__ void dense_forward(const float* input, const float* weights, const float* bias,
                              float* preactivation, float* output, int inputs, int outputs, bool leaky) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= outputs) return;
    float value = bias[oc];
    for (int ic = 0; ic < inputs; ++ic)
        value += weights[static_cast<std::size_t>(oc) * inputs + ic] * input[ic];
    preactivation[oc] = value;
    output[oc] = leaky && value < 0 ? value * 0.1f : value;
}

__global__ void softmax_vector(const float* input, float* output, int count) {
    if (blockIdx.x || threadIdx.x) return;
    float maximum = input[0];
    for (int i = 1; i < count; ++i) maximum = fmaxf(maximum, input[i]);
    float total = 0.0f;
    for (int i = 0; i < count; ++i) {
        output[i] = expf(input[i] - maximum);
        total += output[i];
    }
    for (int i = 0; i < count; ++i) output[i] /= total;
}

__global__ void activation_backward(float* gradient, const float* preactivation,
                                    int count, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    if (leaky && preactivation[pos] < 0) gradient[pos] *= 0.1f;
}

__global__ void dense_backward_input(const float* gradient, const float* weights,
                                     float* previous_gradient, int inputs, int outputs) {
    int ic = blockIdx.x * blockDim.x + threadIdx.x;
    if (ic >= inputs) return;
    float value = 0.0f;
    for (int oc = 0; oc < outputs; ++oc)
        value += gradient[oc] * weights[static_cast<std::size_t>(oc) * inputs + ic];
    previous_gradient[ic] = value;
}

__global__ void dense_sgd(float* weights, float* bias, const float* gradient,
                          const float* previous_output, int inputs, int outputs,
                          float learning_rate, float weight_decay) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= inputs * outputs) return;
    int oc = pos / inputs;
    int ic = pos % inputs;
    float weight = weights[pos];
    weights[pos] = weight - learning_rate *
        (gradient[oc] * previous_output[ic] + weight_decay * weight);
    if (ic == 0) bias[oc] -= learning_rate * gradient[oc];
}

__global__ void convolution_forward(const float* input, const float* weights, const float* bias,
                                    float* preactivation, float* output, int channels, int ih, int iw,
                                    int filters, int oh, int ow, int kernel, int stride, int pad,
                                    bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= filters * oh * ow) return;
    int ox = pos % ow;
    int oy = pos / ow % oh;
    int oc = pos / (ow * oh);
    float value = bias[oc];
    for (int ic = 0; ic < channels; ++ic)
        for (int ky = 0; ky < kernel; ++ky)
            for (int kx = 0; kx < kernel; ++kx) {
                int iy = oy * stride + ky - pad;
                int ix = ox * stride + kx - pad;
                if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
                std::size_t wi = ((static_cast<std::size_t>(oc) * channels + ic) * kernel + ky) *
                    kernel + kx;
                value += weights[wi] * input[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix];
            }
    preactivation[pos] = value;
    output[pos] = leaky && value < 0 ? value * 0.1f : value;
}

__global__ void maxpool_forward(const float* input, float* output, int* index,
                                int channels, int ih, int iw, int oh, int ow,
                                int kernel, int stride) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= channels * oh * ow) return;
    int ox = pos % ow;
    int oy = pos / ow % oh;
    int c = pos / (ow * oh);
    int pad = (kernel - 1) / 2;
    float best = -3.402823466e+38f;
    int best_pos = 0;
    for (int ky = 0; ky < kernel; ++ky)
        for (int kx = 0; kx < kernel; ++kx) {
            int iy = oy * stride + ky - pad;
            int ix = ox * stride + kx - pad;
            if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
            int source = (c * ih + iy) * iw + ix;
            if (input[source] > best) { best = input[source]; best_pos = source; }
        }
    output[pos] = best;
    index[pos] = best_pos;
}

__global__ void maxpool_backward(const float* gradient, const int* index,
                                 float* previous_gradient, int count) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < count) atomicAdd(previous_gradient + index[pos], gradient[pos]);
}

__global__ void convolution_backward_input(const float* gradient, const float* weights,
                                           float* previous_gradient, int channels, int ih, int iw,
                                           int filters, int oh, int ow, int kernel, int stride, int pad) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= channels * ih * iw) return;
    int ix = pos % iw;
    int iy = pos / iw % ih;
    int ic = pos / (iw * ih);
    float value = 0.0f;
    for (int oc = 0; oc < filters; ++oc)
        for (int ky = 0; ky < kernel; ++ky)
            for (int kx = 0; kx < kernel; ++kx) {
                int row = iy + pad - ky;
                int col = ix + pad - kx;
                if (row < 0 || col < 0 || row % stride || col % stride) continue;
                int oy = row / stride;
                int ox = col / stride;
                if (oy >= oh || ox >= ow) continue;
                std::size_t wi = ((static_cast<std::size_t>(oc) * channels + ic) * kernel + ky) *
                    kernel + kx;
                value += gradient[(static_cast<std::size_t>(oc) * oh + oy) * ow + ox] * weights[wi];
            }
    previous_gradient[pos] = value;
}

__global__ void convolution_sgd_weights(float* weights, const float* gradient,
                                        const float* previous_output, int channels, int ih, int iw,
                                        int filters, int oh, int ow, int kernel, int stride, int pad,
                                        float learning_rate, float weight_decay) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int count = filters * channels * kernel * kernel;
    if (pos >= count) return;
    int kx = pos % kernel;
    int ky = pos / kernel % kernel;
    int ic = pos / (kernel * kernel) % channels;
    int oc = pos / (kernel * kernel * channels);
    float sum = 0.0f;
    for (int oy = 0; oy < oh; ++oy)
        for (int ox = 0; ox < ow; ++ox) {
            int iy = oy * stride + ky - pad;
            int ix = ox * stride + kx - pad;
            if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
            sum += gradient[(static_cast<std::size_t>(oc) * oh + oy) * ow + ox] *
                previous_output[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix];
        }
    float weight = weights[pos];
    weights[pos] = weight - learning_rate * (sum + weight_decay * weight);
}

__global__ void convolution_sgd_bias(float* bias, const float* gradient,
                                     int filters, int oh, int ow, float learning_rate) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= filters) return;
    float sum = 0.0f;
    for (int pos = 0; pos < oh * ow; ++pos)
        sum += gradient[static_cast<std::size_t>(oc) * oh * ow + pos];
    bias[oc] -= learning_rate * sum;
}

/** Applies one averaged dense update after every timestep has backpropagated. */
__global__ void dense_sequence_sgd(float* weights, float* bias, const float* gradient,
                                   const float* previous, int inputs, int outputs, int steps,
                                   float learning_rate, float weight_decay) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= inputs * outputs) return;
    int oc = pos / inputs, ic = pos % inputs;
    float sum = 0.0f, bias_sum = 0.0f;
    for (int t = 0; t < steps; ++t) {
        float g = gradient[static_cast<std::size_t>(t) * outputs + oc];
        sum += g * previous[static_cast<std::size_t>(t) * inputs + ic];
        bias_sum += g;
    }
    float old = weights[pos];
    weights[pos] = old - learning_rate * (sum / steps + weight_decay * old);
    if (ic == 0) bias[oc] -= learning_rate * bias_sum / steps;
}

/** Applies one averaged convolution weight update across sequence timesteps. */
__global__ void convolution_sequence_sgd_weights(float* weights, const float* gradient,
        const float* previous, int channels, int ih, int iw, int filters, int oh, int ow,
        int kernel, int stride, int pad, int steps, float learning_rate, float weight_decay) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= filters * channels * kernel * kernel) return;
    int kx = pos % kernel, ky = pos / kernel % kernel;
    int ic = pos / (kernel * kernel) % channels;
    int oc = pos / (kernel * kernel * channels);
    float sum = 0.0f;
    for (int t = 0; t < steps; ++t)
        for (int oy = 0; oy < oh; ++oy)
            for (int ox = 0; ox < ow; ++ox) {
                int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
                sum += gradient[(static_cast<std::size_t>(t) * filters + oc) * oh * ow + oy * ow + ox] *
                       previous[(static_cast<std::size_t>(t) * channels + ic) * ih * iw + iy * iw + ix];
            }
    float old = weights[pos];
    weights[pos] = old - learning_rate * (sum / steps + weight_decay * old);
}

/** Applies one averaged convolution bias update across sequence timesteps. */
__global__ void convolution_sequence_sgd_bias(float* bias, const float* gradient,
        int filters, int oh, int ow, int steps, float learning_rate) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= filters) return;
    float sum = 0.0f;
    for (int t = 0; t < steps; ++t)
        for (int pos = 0; pos < oh * ow; ++pos)
            sum += gradient[(static_cast<std::size_t>(t) * filters + oc) * oh * ow + pos];
    bias[oc] -= learning_rate * sum / steps;
}

/** Reports a failed CUDA classifier kernel launch at its graph stage. */
void launch_check(const char* operation = "CUDA classifier kernel launch") {
    check(cudaGetLastError(), operation);
}

}  // namespace

struct CudaClassificationTrainer::Impl {
    Network& network;
    float learning_rate;
    float weight_decay;
    DeviceMemory memory;
    float* input = nullptr;
    std::vector<float*> output;
    std::vector<float*> preactivation;
    std::vector<float*> gradient;
    std::vector<float*> weights;
    std::vector<float*> biases;
    std::vector<int*> max_index;
    int recurrent_index = -1;

    /** Allocates graph buffers and uploads learnable nonrecurrent parameters. */
    Impl(Network& net, float lr, float decay)
        : network(net), learning_rate(lr), weight_decay(decay) {
        if (!std::isfinite(lr) || lr <= 0 || !std::isfinite(decay) || decay < 0)
            throw std::runtime_error("invalid CUDA SGD parameters");
        const auto& layers = net.layers_;
        if (layers.size() < 2 || layers.back().kind != Kind::softmax)
            throw std::runtime_error("CUDA classifier requires dense layers and final [softmax]");
        for (std::size_t i = 0; i + 1 < layers.size(); ++i) {
            if (layers[i].kind == Kind::rnn || layers[i].kind == Kind::lstm || layers[i].kind == Kind::gru) {
                if (recurrent_index >= 0) throw std::runtime_error("CUDA classifier supports at most one recurrent layer");
                recurrent_index = static_cast<int>(i);
                if (!net.cells_[i]) throw std::runtime_error("initialize or load recurrent weights before CUDA training");
                continue;
            }
            if (layers[i].kind != Kind::connected && layers[i].kind != Kind::convolutional &&
                layers[i].kind != Kind::maxpool)
                throw std::runtime_error("CUDA classifier trainer supports connected, convolutional, maxpool, one recurrent layer, and softmax");
        }
        int devices = 0;
        check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices < 1) throw std::runtime_error("no CUDA device available");
        input = memory.allocate(net.input_.elements());
        output.resize(layers.size());
        preactivation.resize(layers.size(), nullptr);
        gradient.resize(layers.size(), nullptr);
        weights.resize(layers.size(), nullptr);
        biases.resize(layers.size(), nullptr);
        max_index.resize(layers.size(), nullptr);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            output[i] = memory.allocate(layer.output.elements());
            if (layer.kind == Kind::softmax) continue;
            gradient[i] = memory.allocate(layer.output.elements());
            if (layer.kind == Kind::maxpool) {
                max_index[i] = memory.allocate_int(layer.output.elements());
                continue;
            }
            if (static_cast<int>(i) == recurrent_index) continue;
            const std::size_t inputs = layer.input.elements();
            std::size_t fan_in = layer.kind == Kind::connected ? inputs :
                static_cast<std::size_t>(layer.input.c) * layer.size * layer.size;
            if (layer.batch_normalize)
                throw std::runtime_error("CUDA classifier training does not support batch normalization yet");
            if (inputs > 10000000 || layer.weights.size() != fan_in * layer.output.c ||
                layer.biases.size() != static_cast<std::size_t>(layer.output.c))
                throw std::runtime_error("initialize or load classifier weights before CUDA training");
            preactivation[i] = memory.allocate(layer.output.elements());
            weights[i] = memory.upload(layer.weights);
            biases[i] = memory.upload(layer.biases);
        }
    }

    /** Runs one SGD step for a graph with no recurrent layer. */
    ClassificationResult step(const std::vector<float>& host_input, int class_id) {
        const auto& layers = network.layers_;
        if (recurrent_index >= 0)
            throw std::runtime_error("cfg has a recurrent layer; use train_sequence() instead of step()");
        if (host_input.size() != network.input_.elements())
            throw std::runtime_error("CUDA classifier input shape mismatch");
        int classes = layers.back().output.c;
        if (class_id < 0 || class_id >= classes)
            throw std::runtime_error("CUDA classifier target outside class range");
        check(cudaMemcpy(input, host_input.data(), host_input.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "cudaMemcpy classifier input");
        constexpr int threads = 256;
        for (std::size_t i = 0; i + 1 < layers.size(); ++i) {
            const auto& layer = layers[i];
            const float* previous = i ? output[i - 1] : input;
            if (layer.kind == Kind::connected)
                dense_forward<<<(layer.output.c + threads - 1) / threads, threads>>>(
                    previous, weights[i], biases[i], preactivation[i], output[i],
                    static_cast<int>(layer.input.elements()), layer.output.c, layer.leaky);
            else if (layer.kind == Kind::convolutional)
                convolution_forward<<<(static_cast<int>(layer.output.elements()) + threads - 1) / threads, threads>>>(
                    previous, weights[i], biases[i], preactivation[i], output[i],
                    layer.input.c, layer.input.h, layer.input.w,
                    layer.output.c, layer.output.h, layer.output.w,
                    layer.size, layer.stride, layer.padding, layer.leaky);
            else
                maxpool_forward<<<(static_cast<int>(layer.output.elements()) + threads - 1) / threads, threads>>>(
                    previous, output[i], max_index[i], layer.input.c, layer.input.h, layer.input.w,
                    layer.output.h, layer.output.w, layer.size, layer.stride);
            launch_check();
        }
        softmax_vector<<<1, 1>>>(output[layers.size() - 2], output.back(), classes);
        launch_check();
        std::vector<float> probabilities(static_cast<std::size_t>(classes));
        check(cudaMemcpy(probabilities.data(), output.back(), probabilities.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "cudaMemcpy classifier probabilities");
        float loss = -std::log(std::max(probabilities[static_cast<std::size_t>(class_id)],
                                         std::numeric_limits<float>::min()));
        bool correct = static_cast<int>(std::max_element(probabilities.begin(), probabilities.end()) -
                                        probabilities.begin()) == class_id;
        probabilities[static_cast<std::size_t>(class_id)] -= 1.0f;
        check(cudaMemcpy(gradient[layers.size() - 2], probabilities.data(),
                         probabilities.size() * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy classifier gradient");

        for (std::size_t reverse = layers.size() - 1; reverse-- > 0;) {
            const auto& layer = layers[reverse];
            int outputs = layer.output.c;
            int inputs = static_cast<int>(layer.input.elements());
            if (layer.kind == Kind::maxpool) {
                if (reverse) {
                    check(cudaMemset(gradient[reverse - 1], 0, inputs * sizeof(float)),
                          "cudaMemset maxpool gradient");
                    int count = static_cast<int>(layer.output.elements());
                    maxpool_backward<<<(count + threads - 1) / threads, threads>>>(
                        gradient[reverse], max_index[reverse], gradient[reverse - 1], count);
                    launch_check();
                }
                continue;
            }
            int count = static_cast<int>(layer.output.elements());
            activation_backward<<<(count + threads - 1) / threads, threads>>>(
                gradient[reverse], preactivation[reverse], count, layer.leaky);
            launch_check();
            if (layer.kind == Kind::connected) {
                if (reverse) {
                    dense_backward_input<<<(inputs + threads - 1) / threads, threads>>>(
                        gradient[reverse], weights[reverse], gradient[reverse - 1], inputs, outputs);
                    launch_check();
                }
                dense_sgd<<<(inputs * outputs + threads - 1) / threads, threads>>>(
                    weights[reverse], biases[reverse], gradient[reverse],
                    reverse ? output[reverse - 1] : input, inputs, outputs,
                    learning_rate, weight_decay);
                launch_check();
            } else {
                if (reverse) {
                    convolution_backward_input<<<(inputs + threads - 1) / threads, threads>>>(
                        gradient[reverse], weights[reverse], gradient[reverse - 1],
                        layer.input.c, layer.input.h, layer.input.w,
                        layer.output.c, layer.output.h, layer.output.w,
                        layer.size, layer.stride, layer.padding);
                    launch_check();
                }
                int weight_count = static_cast<int>(layer.weights.size());
                convolution_sgd_weights<<<(weight_count + threads - 1) / threads, threads>>>(
                    weights[reverse], gradient[reverse], reverse ? output[reverse - 1] : input,
                    layer.input.c, layer.input.h, layer.input.w,
                    layer.output.c, layer.output.h, layer.output.w,
                    layer.size, layer.stride, layer.padding, learning_rate, weight_decay);
                launch_check();
                convolution_sgd_bias<<<(outputs + threads - 1) / threads, threads>>>(
                    biases[reverse], gradient[reverse], outputs,
                    layer.output.h, layer.output.w, learning_rate);
                launch_check();
            }
        }
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize classifier SGD");
        if (!std::isfinite(loss)) throw std::runtime_error("non-finite CUDA classification loss");
        return {loss, correct};
    }

    /** Trains one time chunk; nonrecurrent layers remain on CUDA and the CUDA recurrent
     *  runner performs the entire chunk's recurrence and BPTT before one SGD update. */
    std::vector<ClassificationResult> train_sequence(
            const std::vector<std::vector<float>>& host_inputs, const std::vector<int>& class_ids) {
        const auto& layers = network.layers_;
        if (recurrent_index < 0)
            throw std::runtime_error("train_sequence requires one recurrent layer; use step() otherwise");
        if (host_inputs.empty() || host_inputs.size() != class_ids.size())
            throw std::runtime_error("train_sequence requires matching, non-empty inputs/class_ids");
        const int steps = static_cast<int>(host_inputs.size());
        const int classes = layers.back().output.c;
        for (std::size_t t = 0; t < host_inputs.size(); ++t) {
            if (host_inputs[t].size() != network.input_.elements())
                throw std::runtime_error("CUDA classification input shape mismatch");
            if (class_ids[t] < 0 || class_ids[t] >= classes)
                throw std::runtime_error("CUDA classification target outside class range");
        }
        DeviceMemory sequence_memory;
        const std::size_t input_size = network.input_.elements();
        float* sequence_input = sequence_memory.allocate(static_cast<std::size_t>(steps) * input_size);
        for (int t = 0; t < steps; ++t)
            check(cudaMemcpy(sequence_input + static_cast<std::size_t>(t) * input_size,
                             host_inputs[t].data(), input_size * sizeof(float), cudaMemcpyHostToDevice),
                  "cudaMemcpy classifier sequence input");
        std::vector<float*> values(layers.size()), derivatives(layers.size(), nullptr);
        std::vector<float*> pre(layers.size(), nullptr);
        std::vector<int*> indices(layers.size(), nullptr);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            std::size_t count = static_cast<std::size_t>(steps) * layers[i].output.elements();
            values[i] = sequence_memory.allocate(count);
            if (layers[i].kind == Kind::softmax) continue;
            derivatives[i] = sequence_memory.allocate(count);
            check(cudaMemset(derivatives[i], 0, count * sizeof(float)), "cudaMemset sequence gradient");
            if (layers[i].kind == Kind::connected || layers[i].kind == Kind::convolutional)
                pre[i] = sequence_memory.allocate(count);
            if (layers[i].kind == Kind::maxpool)
                indices[i] = sequence_memory.allocate_int(count);
        }
        constexpr int threads = 256;
        std::vector<std::vector<float>> recurrent_inputs;
        std::vector<float> initial_hidden, initial_cell;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            const float* previous = i ? values[i - 1] : sequence_input;
            const int in_count = static_cast<int>(layer.input.elements());
            const int out_count = static_cast<int>(layer.output.elements());
            if (static_cast<int>(i) == recurrent_index) {
                auto& cell = *network.cells_[i];
                initial_hidden = cell.hidden_state(); initial_cell = cell.cell_state();
                recurrent_inputs.resize(steps, std::vector<float>(static_cast<std::size_t>(in_count)));
                for (int t = 0; t < steps; ++t)
                    check(cudaMemcpy(recurrent_inputs[t].data(), previous + static_cast<std::size_t>(t) * in_count,
                                     in_count * sizeof(float), cudaMemcpyDeviceToHost),
                          "cudaMemcpy recurrent sequence input");
                auto recurrence = recurrent_forward_cuda(cell, recurrent_inputs, initial_hidden, initial_cell);
                for (int t = 0; t < steps; ++t)
                    check(cudaMemcpy(values[i] + static_cast<std::size_t>(t) * out_count,
                                     recurrence.outputs[t].data(), out_count * sizeof(float), cudaMemcpyHostToDevice),
                          "cudaMemcpy recurrent sequence output");
                cell.set_state(recurrence.final_hidden, recurrence.final_cell);
                continue;
            }
            for (int t = 0; t < steps; ++t) {
                const float* source = previous + static_cast<std::size_t>(t) * in_count;
                float* destination = values[i] + static_cast<std::size_t>(t) * out_count;
                if (layer.kind == Kind::connected)
                    dense_forward<<<(out_count + threads - 1) / threads, threads>>>(source, weights[i], biases[i],
                        pre[i] + static_cast<std::size_t>(t) * out_count, destination,
                        in_count, out_count, layer.leaky);
                else if (layer.kind == Kind::convolutional)
                    convolution_forward<<<(out_count + threads - 1) / threads, threads>>>(source, weights[i], biases[i],
                        pre[i] + static_cast<std::size_t>(t) * out_count, destination,
                        layer.input.c, layer.input.h, layer.input.w, layer.output.c, layer.output.h,
                        layer.output.w, layer.size, layer.stride, layer.padding, layer.leaky);
                else if (layer.kind == Kind::maxpool)
                    maxpool_forward<<<(out_count + threads - 1) / threads, threads>>>(source, destination,
                        indices[i] + static_cast<std::size_t>(t) * out_count,
                        layer.input.c, layer.input.h, layer.input.w, layer.output.h, layer.output.w,
                        layer.size, layer.stride);
                else
                    softmax_vector<<<1, 1>>>(source, destination, out_count);
                launch_check("CUDA classifier sequence forward");
            }
        }
        std::vector<ClassificationResult> results(steps);
        std::vector<float> probabilities(static_cast<std::size_t>(classes));
        for (int t = 0; t < steps; ++t) {
            check(cudaMemcpy(probabilities.data(), values.back() + static_cast<std::size_t>(t) * classes,
                             classes * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy sequence probabilities");
            float loss = -std::log(std::max(probabilities[static_cast<std::size_t>(class_ids[t])],
                                             std::numeric_limits<float>::min()));
            if (!std::isfinite(loss)) throw std::runtime_error("non-finite CUDA classification loss");
            bool correct = static_cast<int>(std::max_element(probabilities.begin(), probabilities.end()) -
                                            probabilities.begin()) == class_ids[t];
            results[t] = {loss, correct};
            probabilities[static_cast<std::size_t>(class_ids[t])] -= 1.0f;
            check(cudaMemcpy(derivatives[layers.size() - 2] + static_cast<std::size_t>(t) * classes,
                             probabilities.data(), classes * sizeof(float), cudaMemcpyHostToDevice),
                  "cudaMemcpy sequence loss gradient");
        }
        for (std::size_t reverse = layers.size() - 1; reverse-- > 0;) {
            const auto& layer = layers[reverse];
            const int in_count = static_cast<int>(layer.input.elements());
            const int out_count = static_cast<int>(layer.output.elements());
            const float* previous = reverse ? values[reverse - 1] : sequence_input;
            if (static_cast<int>(reverse) == recurrent_index) {
                std::vector<std::vector<float>> direct(steps, std::vector<float>(out_count));
                for (int t = 0; t < steps; ++t)
                    check(cudaMemcpy(direct[t].data(), derivatives[reverse] + static_cast<std::size_t>(t) * out_count,
                                     out_count * sizeof(float), cudaMemcpyDeviceToHost),
                          "cudaMemcpy recurrent sequence gradient");
                auto& cell = *network.cells_[reverse];
                auto recurrence = recurrent_forward_backward_cuda(
                    cell, recurrent_inputs, direct, initial_hidden, initial_cell);
                if (reverse)
                    for (int t = 0; t < steps; ++t)
                        check(cudaMemcpy(derivatives[reverse - 1] + static_cast<std::size_t>(t) * in_count,
                                         recurrence.input_gradients[t].data(), in_count * sizeof(float),
                                         cudaMemcpyHostToDevice), "cudaMemcpy recurrent input gradient");
                auto parameters = cell.parameters();
                for (std::size_t p = 0; p < parameters.size(); ++p)
                    *parameters[p] -= learning_rate * recurrence.parameter_gradients[p];
                continue;
            }
            for (int t = 0; t < steps; ++t) {
                float* grad = derivatives[reverse] + static_cast<std::size_t>(t) * out_count;
                if (layer.kind == Kind::maxpool) {
                    if (reverse)
                        maxpool_backward<<<(out_count + threads - 1) / threads, threads>>>(
                            grad, indices[reverse] + static_cast<std::size_t>(t) * out_count,
                            derivatives[reverse - 1] + static_cast<std::size_t>(t) * in_count, out_count);
                    launch_check("CUDA classifier sequence maxpool backward");
                    continue;
                }
                activation_backward<<<(out_count + threads - 1) / threads, threads>>>(
                    grad, pre[reverse] + static_cast<std::size_t>(t) * out_count, out_count, layer.leaky);
                launch_check("CUDA classifier sequence activation backward");
                if (!reverse) continue;
                float* preceding = derivatives[reverse - 1] + static_cast<std::size_t>(t) * in_count;
                if (layer.kind == Kind::connected)
                    dense_backward_input<<<(in_count + threads - 1) / threads, threads>>>(
                        grad, weights[reverse], preceding, in_count, out_count);
                else
                    convolution_backward_input<<<(in_count + threads - 1) / threads, threads>>>(
                        grad, weights[reverse], preceding, layer.input.c, layer.input.h, layer.input.w,
                        layer.output.c, layer.output.h, layer.output.w, layer.size, layer.stride, layer.padding);
                launch_check("CUDA classifier sequence input backward");
            }
            if (layer.kind == Kind::maxpool) continue;
            if (layer.kind == Kind::connected)
                dense_sequence_sgd<<<(in_count * out_count + threads - 1) / threads, threads>>>(
                    weights[reverse], biases[reverse], derivatives[reverse], previous,
                    in_count, out_count, steps, learning_rate, weight_decay);
            else {
                int weight_count = static_cast<int>(layer.weights.size());
                convolution_sequence_sgd_weights<<<(weight_count + threads - 1) / threads, threads>>>(
                    weights[reverse], derivatives[reverse], previous, layer.input.c, layer.input.h,
                    layer.input.w, layer.output.c, layer.output.h, layer.output.w,
                    layer.size, layer.stride, layer.padding, steps, learning_rate, weight_decay);
                launch_check("CUDA classifier sequence convolution weight SGD");
                convolution_sequence_sgd_bias<<<(layer.output.c + threads - 1) / threads, threads>>>(
                    biases[reverse], derivatives[reverse], layer.output.c,
                    layer.output.h, layer.output.w, steps, learning_rate);
                launch_check("CUDA classifier sequence convolution bias SGD");
            }
            if (layer.kind == Kind::connected) launch_check("CUDA classifier sequence dense SGD");
        }
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize classifier sequence SGD");
        return results;
    }

    /** Downloads learned nonrecurrent parameters for checkpointing or CPU inference. */
    void sync_to_network() {
        auto& layers = network.layers_;
        for (std::size_t i = 0; i + 1 < layers.size(); ++i) {
            auto& layer = layers[i];
            if (layer.kind == Kind::maxpool || static_cast<int>(i) == recurrent_index) continue;
            check(cudaMemcpy(layer.weights.data(), weights[i],
                             layer.weights.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy learned weights");
            check(cudaMemcpy(layer.biases.data(), biases[i],
                             layer.biases.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy learned biases");
        }
    }
};

CudaClassificationTrainer::CudaClassificationTrainer(Network& network, float learning_rate, float weight_decay)
    : impl_(std::make_unique<Impl>(network, learning_rate, weight_decay)) {}
CudaClassificationTrainer::~CudaClassificationTrainer() = default;
ClassificationResult CudaClassificationTrainer::step(const std::vector<float>& input, int class_id) {
    return impl_->step(input, class_id);
}
std::vector<ClassificationResult> CudaClassificationTrainer::train_sequence(
        const std::vector<std::vector<float>>& inputs, const std::vector<int>& class_ids) {
    return impl_->train_sequence(inputs, class_ids);
}
void CudaClassificationTrainer::sync_to_network() { impl_->sync_to_network(); }

}  // namespace detector
#endif
