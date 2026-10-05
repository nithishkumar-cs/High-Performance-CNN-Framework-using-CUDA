#include "detector/cuda_trainer.hpp"
#ifdef DETECTOR_HAS_CUDA
#include "detector/yolo_loss.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace detector {
namespace {

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

void launch_check() { check(cudaGetLastError(), "CUDA training kernel launch"); }

struct DeviceMemory {
    std::vector<void*> allocations;
    ~DeviceMemory() { for (void* pointer : allocations) cudaFree(pointer); }
    float* floats(std::size_t count) {
        float* pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(float)), "cudaMalloc");
        check(cudaMemset(pointer, 0, count * sizeof(float)), "cudaMemset");
        allocations.push_back(pointer);
        return pointer;
    }
    int* ints(std::size_t count) {
        int* pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(int)), "cudaMalloc indices");
        allocations.push_back(pointer);
        return pointer;
    }
    float* upload(const std::vector<float>& host) {
        float* pointer = floats(host.size());
        check(cudaMemcpy(pointer, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        return pointer;
    }
};

constexpr float kBnEps = 1e-5f;
constexpr int kThreads = 256;
inline int blocks_for(int count) { return count > 0 ? (count + kThreads - 1) / kThreads : 1; }

__global__ void convolution_forward_plain(const float* input, const float* weights, const float* bias,
                                          float* preactivation, float* output, int channels, int ih, int iw,
                                          int filters, int oh, int ow, int kernel, int stride, int pad, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= filters * oh * ow) return;
    int ox = pos % ow, oy = pos / ow % oh, oc = pos / (ow * oh);
    float value = bias[oc];
    for (int ic = 0; ic < channels; ++ic)
        for (int ky = 0; ky < kernel; ++ky)
            for (int kx = 0; kx < kernel; ++kx) {
                int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
                std::size_t wi = ((static_cast<std::size_t>(oc) * channels + ic) * kernel + ky) * kernel + kx;
                value += weights[wi] * input[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix];
            }
    preactivation[pos] = value;
    output[pos] = leaky && value < 0 ? value * 0.1f : value;
}

// Raw (pre-batch-norm) convolution sum only; mean/variance/normalization are separate kernels
// so each parallelizes over its own natural dimension (output elements, then channels).
__global__ void convolution_forward_raw(const float* input, const float* weights, float* raw,
                                        int channels, int ih, int iw, int filters, int oh, int ow,
                                        int kernel, int stride, int pad) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= filters * oh * ow) return;
    int ox = pos % ow, oy = pos / ow % oh, oc = pos / (ow * oh);
    float value = 0;
    for (int ic = 0; ic < channels; ++ic)
        for (int ky = 0; ky < kernel; ++ky)
            for (int kx = 0; kx < kernel; ++kx) {
                int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
                std::size_t wi = ((static_cast<std::size_t>(oc) * channels + ic) * kernel + ky) * kernel + kx;
                value += weights[wi] * input[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix];
            }
    raw[pos] = value;
}

__global__ void batch_norm_statistics(const float* raw, float* mean, float* variance, int filters, int plane) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= filters) return;
    const float* channel = raw + static_cast<std::size_t>(oc) * plane;
    float sum = 0;
    for (int p = 0; p < plane; ++p) sum += channel[p];
    float m = sum / plane;
    float var_sum = 0;
    for (int p = 0; p < plane; ++p) { float d = channel[p] - m; var_sum += d * d; }
    mean[oc] = m;
    variance[oc] = var_sum / plane;
}

__global__ void batch_norm_apply(const float* raw, const float* mean, const float* variance,
                                 const float* gamma, const float* beta, float* preactivation,
                                 float* output, int count, int plane, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    int oc = pos / plane;
    float inv_std = 1.0f / sqrtf(variance[oc] + kBnEps);
    float xhat = (raw[pos] - mean[oc]) * inv_std;
    float value = gamma[oc] * xhat + beta[oc];
    preactivation[pos] = value;
    output[pos] = leaky && value < 0 ? value * 0.1f : value;
}

__global__ void update_running_stats(float* running_mean, float* running_variance, const float* mean,
                                     const float* variance, int filters, float momentum) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= filters) return;
    running_mean[oc] = momentum * running_mean[oc] + (1 - momentum) * mean[oc];
    running_variance[oc] = momentum * running_variance[oc] + (1 - momentum) * variance[oc];
}

__global__ void max_pool_forward(const float* input, float* output, int* index, int channels,
                                 int ih, int iw, int oh, int ow, int kernel, int stride) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= channels * oh * ow) return;
    int ox = pos % ow, oy = pos / ow % oh, c = pos / (ow * oh);
    int pad = (kernel - 1) / 2;
    float best = -3.402823466e+38f;
    int best_pos = 0;
    for (int ky = 0; ky < kernel; ++ky)
        for (int kx = 0; kx < kernel; ++kx) {
            int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
            if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
            int source = (c * ih + iy) * iw + ix;
            if (input[source] > best) { best = input[source]; best_pos = source; }
        }
    output[pos] = best;
    index[pos] = best_pos;
}

__global__ void upsample_forward(const float* input, float* output, int channels, int ih, int iw,
                                 int oh, int ow, int stride) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= channels * oh * ow) return;
    int x = pos % ow, y = pos / ow % oh, c = pos / (ow * oh);
    output[pos] = input[(static_cast<std::size_t>(c) * ih + y / stride) * iw + x / stride];
}

__global__ void shortcut_forward(const float* previous, const float* branch, float* preactivation,
                                 float* output, int count, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    float value = previous[pos] + branch[pos];
    preactivation[pos] = value;
    output[pos] = leaky && value < 0 ? value * 0.1f : value;
}

__global__ void activation_backward(float* gradient, const float* preactivation, int count, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < count && leaky && preactivation[pos] < 0) gradient[pos] *= 0.1f;
}

// First batch-norm backward pass: per channel (one thread each), reduces g=dL/dvalue (already in
// `g`, post activation_backward) and xhat into the sums BN backward needs, and accumulates this
// sample's gamma/beta gradient directly (no atomics: one thread per channel, no cross-thread write).
__global__ void batch_norm_backward_reduce(const float* g, const float* raw, const float* mean,
                                           const float* variance, const float* gamma,
                                           float* sum_dxhat, float* sum_dxhat_xhat,
                                           float* scale_grad_accum, float* bias_grad_accum,
                                           int filters, int plane) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= filters) return;
    float inv_std = 1.0f / sqrtf(variance[oc] + kBnEps);
    const float* gc = g + static_cast<std::size_t>(oc) * plane;
    const float* rc = raw + static_cast<std::size_t>(oc) * plane;
    float sd = 0, sdx = 0, dgamma = 0, dbeta = 0;
    for (int p = 0; p < plane; ++p) {
        float xhat = (rc[p] - mean[oc]) * inv_std;
        float dxhat = gc[p] * gamma[oc];
        sd += dxhat;
        sdx += dxhat * xhat;
        dgamma += gc[p] * xhat;
        dbeta += gc[p];
    }
    sum_dxhat[oc] = sd;
    sum_dxhat_xhat[oc] = sdx;
    scale_grad_accum[oc] += dgamma;
    bias_grad_accum[oc] += dbeta;
}

// Second pass: per output element, turns g into dz (gradient w.r.t. the pre-norm conv sum),
// overwriting `g` in place so the same buffer feeds conv_backward_input/weight below.
__global__ void batch_norm_backward_dz(float* g, const float* raw, const float* mean,
                                       const float* variance, const float* gamma,
                                       const float* sum_dxhat, const float* sum_dxhat_xhat, int count, int plane) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    int oc = pos / plane;
    float inv_std = 1.0f / sqrtf(variance[oc] + kBnEps);
    float xhat = (raw[pos] - mean[oc]) * inv_std;
    float dxhat = g[pos] * gamma[oc];
    g[pos] = inv_std / plane * (plane * dxhat - sum_dxhat[oc] - xhat * sum_dxhat_xhat[oc]);
}

__global__ void bias_grad_reduce(const float* dz, float* bias_grad_accum, int filters, int plane) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= filters) return;
    const float* dc = dz + static_cast<std::size_t>(oc) * plane;
    float sum = 0;
    for (int p = 0; p < plane; ++p) sum += dc[p];
    bias_grad_accum[oc] += sum;
}

// Gather-based: one thread per input element, no atomics needed (each thread's single write
// accumulates onto whatever this destination already holds from an earlier-processed layer).
__global__ void convolution_backward_input(const float* dz, const float* weights, float* previous_gradient,
                                           int channels, int ih, int iw, int filters, int oh, int ow,
                                           int kernel, int stride, int pad) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= channels * ih * iw) return;
    int ix = pos % iw, iy = pos / iw % ih, ic = pos / (iw * ih);
    float value = 0;
    for (int oc = 0; oc < filters; ++oc)
        for (int ky = 0; ky < kernel; ++ky)
            for (int kx = 0; kx < kernel; ++kx) {
                int row = iy + pad - ky, col = ix + pad - kx;
                if (row < 0 || col < 0 || row % stride || col % stride) continue;
                int oy = row / stride, ox = col / stride;
                if (oy >= oh || ox >= ow) continue;
                std::size_t wi = ((static_cast<std::size_t>(oc) * channels + ic) * kernel + ky) * kernel + kx;
                value += dz[(static_cast<std::size_t>(oc) * oh + oy) * ow + ox] * weights[wi];
            }
    previous_gradient[pos] += value;
}

__global__ void convolution_backward_weight(const float* dz, const float* previous_output, float* weight_grad_accum,
                                            int channels, int ih, int iw, int filters, int oh, int ow,
                                            int kernel, int stride, int pad) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int count = filters * channels * kernel * kernel;
    if (pos >= count) return;
    int kx = pos % kernel, ky = pos / kernel % kernel, ic = pos / (kernel * kernel) % channels, oc = pos / (kernel * kernel * channels);
    float sum = 0;
    for (int oy = 0; oy < oh; ++oy)
        for (int ox = 0; ox < ow; ++ox) {
            int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
            if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
            sum += dz[(static_cast<std::size_t>(oc) * oh + oy) * ow + ox] *
                previous_output[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix];
        }
    weight_grad_accum[pos] += sum;
}

__global__ void max_pool_backward(const float* gradient, const int* index, float* previous_gradient, int count) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < count) atomicAdd(previous_gradient + index[pos], gradient[pos]);
}

// Multiple output positions map to the same input pixel (upsampling duplicates it), so distinct
// threads in this launch can target the same destination: atomicAdd is required here.
__global__ void upsample_backward(const float* gradient, float* previous_gradient, int channels,
                                  int ih, int iw, int oh, int ow, int stride) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= channels * oh * ow) return;
    int x = pos % ow, y = pos / ow % oh, c = pos / (ow * oh);
    atomicAdd(previous_gradient + (static_cast<std::size_t>(c) * ih + y / stride) * iw + x / stride, gradient[pos]);
}

// One thread per element writes to two disjoint buffers at its own index: no atomics needed.
__global__ void shortcut_backward(const float* gradient, float* previous_gradient, float* branch_gradient, int count) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    previous_gradient[pos] += gradient[pos];
    branch_gradient[pos] += gradient[pos];
}

// Plain elementwise accumulate: dst[p] += src[p]. Used for the yolo pass-through and the
// route split, where (unlike shortcut) only one destination buffer is touched per element.
__global__ void add_into(float* dst, const float* src, int count) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < count) dst[pos] += src[pos];
}

__global__ void sgd_apply(float* param, const float* grad_accum, int count, float lr, float inv_batch) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < count) param[pos] -= lr * grad_accum[pos] * inv_batch;
}

__global__ void sgd_apply_with_decay(float* param, const float* grad_accum, int count, float lr,
                                     float inv_batch, float decay) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < count) param[pos] -= lr * (grad_accum[pos] * inv_batch + decay * param[pos]);
}

__global__ void zero(float* buffer, int count) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < count) buffer[pos] = 0.0f;
}

}  // namespace

struct CudaTrainer::Impl {
    Network& network;
    LearningRateSchedule schedule;
    int batch_size;
    float weight_decay;
    float bn_momentum;
    int iteration = 0;
    int accumulated = 0;
    DeviceMemory memory;
    float* input = nullptr;
    std::vector<float*> output, preactivation, gradient;
    std::vector<float*> raw, batch_mean, batch_var, sum_dxhat, sum_dxhat_xhat, running_mean, running_variance;
    std::vector<float*> weights, biases, scales;
    std::vector<float*> weight_grad_accum, bias_grad_accum, scale_grad_accum;
    std::vector<int*> max_index;

    Impl(Network& net, LearningRateSchedule sched, int batch, float decay, float momentum)
        : network(net), schedule(std::move(sched)), batch_size(batch), weight_decay(decay), bn_momentum(momentum) {
        if (!std::isfinite(schedule.base_lr) || schedule.base_lr <= 0 || schedule.burn_in < 0 ||
            schedule.steps.size() != schedule.scales.size())
            throw std::runtime_error("invalid learning rate schedule");
        if (!std::isfinite(decay) || decay < 0) throw std::runtime_error("invalid weight decay");
        if (batch < 1) throw std::runtime_error("invalid batch size");
        if (!std::isfinite(momentum) || momentum <= 0 || momentum >= 1) throw std::runtime_error("invalid batch-norm momentum");
        int devices = 0;
        check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices < 1) throw std::runtime_error("no CUDA device available");
        const auto& layers = net.layers();
        for (const auto& layer : layers)
            if (layer.kind != Kind::convolutional && layer.kind != Kind::maxpool && layer.kind != Kind::route &&
                layer.kind != Kind::upsample && layer.kind != Kind::shortcut && layer.kind != Kind::yolo)
                throw std::runtime_error("CUDA detector training supports convolutional, maxpool, route, "
                                          "upsample, shortcut and yolo layers only");
        input = memory.floats(net.input_shape().elements());
        std::size_t n = layers.size();
        output.resize(n); preactivation.resize(n, nullptr); gradient.resize(n);
        raw.resize(n, nullptr); batch_mean.resize(n, nullptr); batch_var.resize(n, nullptr);
        sum_dxhat.resize(n, nullptr); sum_dxhat_xhat.resize(n, nullptr);
        running_mean.resize(n, nullptr); running_variance.resize(n, nullptr);
        weights.resize(n, nullptr); biases.resize(n, nullptr); scales.resize(n, nullptr);
        weight_grad_accum.resize(n, nullptr); bias_grad_accum.resize(n, nullptr); scale_grad_accum.resize(n, nullptr);
        max_index.resize(n, nullptr);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& layer = layers[i];
            output[i] = memory.floats(layer.output.elements());
            gradient[i] = memory.floats(layer.output.elements());
            if (layer.kind == Kind::maxpool) { max_index[i] = memory.ints(layer.output.elements()); continue; }
            if (layer.kind == Kind::shortcut) { preactivation[i] = memory.floats(layer.output.elements()); continue; }
            if (layer.kind != Kind::convolutional) continue;
            preactivation[i] = memory.floats(layer.output.elements());
            std::size_t fan_in = static_cast<std::size_t>(layer.input.c) * layer.size * layer.size;
            if (layer.weights.size() != fan_in * layer.output.c || layer.biases.size() != static_cast<std::size_t>(layer.output.c))
                throw std::runtime_error("initialize or load detector weights before CUDA training");
            weights[i] = memory.upload(layer.weights);
            biases[i] = memory.upload(layer.biases);
            weight_grad_accum[i] = memory.floats(layer.weights.size());
            bias_grad_accum[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
            if (layer.batch_normalize) {
                if (layer.scales.size() != static_cast<std::size_t>(layer.output.c) ||
                    layer.means.size() != static_cast<std::size_t>(layer.output.c) ||
                    layer.variances.size() != static_cast<std::size_t>(layer.output.c))
                    throw std::runtime_error("initialize or load batch-norm parameters before CUDA training");
                raw[i] = memory.floats(layer.output.elements());
                batch_mean[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
                batch_var[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
                sum_dxhat[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
                sum_dxhat_xhat[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
                scales[i] = memory.upload(layer.scales);
                scale_grad_accum[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
                running_mean[i] = memory.upload(layer.means);
                running_variance[i] = memory.upload(layer.variances);
            }
        }
    }

    TrainResult step(const std::vector<float>& host_input, const std::vector<Target>& targets) {
        const auto& layers = network.layers();
        if (host_input.size() != network.input_shape().elements()) throw std::runtime_error("training input shape mismatch");
        check(cudaMemcpy(input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy training input");

        // Forward.
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            const float* previous = i ? output[i - 1] : input;
            int elements = static_cast<int>(layer.output.elements());
            if (layer.kind == Kind::convolutional) {
                int plane = layer.output.h * layer.output.w;
                if (layer.batch_normalize) {
                    convolution_forward_raw<<<blocks_for(elements), kThreads>>>(previous, weights[i], raw[i],
                        layer.input.c, layer.input.h, layer.input.w, layer.output.c, layer.output.h, layer.output.w,
                        layer.size, layer.stride, layer.padding);
                    launch_check();
                    batch_norm_statistics<<<blocks_for(layer.output.c), kThreads>>>(raw[i], batch_mean[i], batch_var[i],
                        layer.output.c, plane);
                    launch_check();
                    batch_norm_apply<<<blocks_for(elements), kThreads>>>(raw[i], batch_mean[i], batch_var[i], scales[i],
                        biases[i], preactivation[i], output[i], elements, plane, layer.leaky);
                    launch_check();
                    update_running_stats<<<blocks_for(layer.output.c), kThreads>>>(running_mean[i], running_variance[i],
                        batch_mean[i], batch_var[i], layer.output.c, bn_momentum);
                    launch_check();
                } else {
                    convolution_forward_plain<<<blocks_for(elements), kThreads>>>(previous, weights[i], biases[i],
                        preactivation[i], output[i], layer.input.c, layer.input.h, layer.input.w,
                        layer.output.c, layer.output.h, layer.output.w, layer.size, layer.stride, layer.padding, layer.leaky);
                    launch_check();
                }
            } else if (layer.kind == Kind::maxpool) {
                max_pool_forward<<<blocks_for(elements), kThreads>>>(previous, output[i], max_index[i],
                    layer.input.c, layer.input.h, layer.input.w, layer.output.h, layer.output.w, layer.size, layer.stride);
                launch_check();
            } else if (layer.kind == Kind::route) {
                std::size_t offset = 0;
                for (int source : layer.sources) {
                    std::size_t count = layers[static_cast<std::size_t>(source)].output.elements();
                    check(cudaMemcpy(output[i] + offset, output[static_cast<std::size_t>(source)],
                                     count * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy route");
                    offset += count;
                }
            } else if (layer.kind == Kind::upsample) {
                upsample_forward<<<blocks_for(elements), kThreads>>>(previous, output[i], layer.input.c,
                    layer.input.h, layer.input.w, layer.output.h, layer.output.w, layer.stride);
                launch_check();
            } else if (layer.kind == Kind::shortcut) {
                shortcut_forward<<<blocks_for(elements), kThreads>>>(previous,
                    output[static_cast<std::size_t>(layer.sources.front())], preactivation[i], output[i], elements, layer.leaky);
                launch_check();
            } else {  // yolo: pass-through, matching the CPU trainer's forward for terminal heads.
                check(cudaMemcpy(output[i], previous, static_cast<std::size_t>(elements) * sizeof(float),
                                 cudaMemcpyDeviceToDevice), "cudaMemcpy yolo forward");
            }
        }

        // YOLO loss: shared host-side assignment/gradient (identical math to the CPU trainer),
        // computed only on the small head tensors downloaded here.
        std::vector<std::vector<float>> host_output(layers.size()), host_gradient(layers.size());
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].kind != Kind::yolo) continue;
            host_output[i].resize(layers[i].output.elements());
            check(cudaMemcpy(host_output[i].data(), output[i], host_output[i].size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy yolo output");
            host_gradient[i].assign(host_output[i].size(), 0.0f);
        }
        auto loss_result = yolo_loss_and_gradient(network, targets, host_output, host_gradient);

        // Backward: zero every gradient buffer, upload the yolo heads' host-computed gradient as
        // the seed, then traverse layers in reverse; each kernel accumulates into its
        // predecessor(s), since sequential launches on one stream see each other's prior writes.
        for (std::size_t i = 0; i < layers.size(); ++i) {
            int elements = static_cast<int>(layers[i].output.elements());
            zero<<<blocks_for(elements), kThreads>>>(gradient[i], elements);
            launch_check();
        }
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].kind != Kind::yolo) continue;
            check(cudaMemcpy(gradient[i], host_gradient[i].data(), host_gradient[i].size() * sizeof(float),
                             cudaMemcpyHostToDevice), "cudaMemcpy yolo gradient");
        }
        for (std::size_t reverse = layers.size(); reverse-- > 0;) {
            const auto& layer = layers[reverse];
            const float* previous = reverse ? output[reverse - 1] : input;
            float* grad = gradient[reverse];
            int elements = static_cast<int>(layer.output.elements());
            if (layer.kind == Kind::yolo) {
                if (reverse) { add_into<<<blocks_for(elements), kThreads>>>(gradient[reverse - 1], grad, elements); launch_check(); }
            } else if (layer.kind == Kind::route) {
                std::size_t offset = 0;
                for (int source : layer.sources) {
                    int count = static_cast<int>(layers[static_cast<std::size_t>(source)].output.elements());
                    add_into<<<blocks_for(count), kThreads>>>(gradient[static_cast<std::size_t>(source)], grad + offset, count);
                    launch_check();
                    offset += static_cast<std::size_t>(count);
                }
            } else if (layer.kind == Kind::upsample) {
                if (reverse) {
                    upsample_backward<<<blocks_for(elements), kThreads>>>(grad, gradient[reverse - 1], layer.input.c,
                        layer.input.h, layer.input.w, layer.output.h, layer.output.w, layer.stride);
                    launch_check();
                }
            } else if (layer.kind == Kind::shortcut) {
                if (!reverse) throw std::runtime_error("shortcut cannot be the first layer");
                activation_backward<<<blocks_for(elements), kThreads>>>(grad, preactivation[reverse], elements, layer.leaky);
                launch_check();
                shortcut_backward<<<blocks_for(elements), kThreads>>>(grad, gradient[reverse - 1],
                    gradient[static_cast<std::size_t>(layer.sources.front())], elements);
                launch_check();
            } else if (layer.kind == Kind::maxpool) {
                if (reverse) {
                    max_pool_backward<<<blocks_for(elements), kThreads>>>(grad, max_index[reverse], gradient[reverse - 1], elements);
                    launch_check();
                }
            } else {  // convolutional
                activation_backward<<<blocks_for(elements), kThreads>>>(grad, preactivation[reverse], elements, layer.leaky);
                launch_check();
                int plane = layer.output.h * layer.output.w;
                if (layer.batch_normalize) {
                    batch_norm_backward_reduce<<<blocks_for(layer.output.c), kThreads>>>(grad, raw[reverse], batch_mean[reverse],
                        batch_var[reverse], scales[reverse], sum_dxhat[reverse], sum_dxhat_xhat[reverse],
                        scale_grad_accum[reverse], bias_grad_accum[reverse], layer.output.c, plane);
                    launch_check();
                    batch_norm_backward_dz<<<blocks_for(elements), kThreads>>>(grad, raw[reverse], batch_mean[reverse],
                        batch_var[reverse], scales[reverse], sum_dxhat[reverse], sum_dxhat_xhat[reverse], elements, plane);
                    launch_check();
                } else {
                    bias_grad_reduce<<<blocks_for(layer.output.c), kThreads>>>(grad, bias_grad_accum[reverse], layer.output.c, plane);
                    launch_check();
                }
                convolution_backward_weight<<<blocks_for(static_cast<int>(layer.weights.size())), kThreads>>>(
                    grad, previous, weight_grad_accum[reverse], layer.input.c, layer.input.h, layer.input.w,
                    layer.output.c, layer.output.h, layer.output.w, layer.size, layer.stride, layer.padding);
                launch_check();
                if (reverse) {
                    convolution_backward_input<<<blocks_for(static_cast<int>(layer.input.elements())), kThreads>>>(
                        grad, weights[reverse], gradient[reverse - 1], layer.input.c, layer.input.h, layer.input.w,
                        layer.output.c, layer.output.h, layer.output.w, layer.size, layer.stride, layer.padding);
                    launch_check();
                }
            }
        }

        bool updated = false;
        if (++accumulated >= batch_size) {
            float lr = schedule.at(iteration);
            float inv_batch = 1.0f / static_cast<float>(batch_size);
            for (std::size_t i = 0; i < layers.size(); ++i) {
                const auto& layer = layers[i];
                if (layer.kind != Kind::convolutional) continue;
                int wcount = static_cast<int>(layer.weights.size());
                sgd_apply_with_decay<<<blocks_for(wcount), kThreads>>>(weights[i], weight_grad_accum[i], wcount, lr, inv_batch, weight_decay);
                launch_check();
                sgd_apply<<<blocks_for(layer.output.c), kThreads>>>(biases[i], bias_grad_accum[i], layer.output.c, lr, inv_batch);
                launch_check();
                zero<<<blocks_for(wcount), kThreads>>>(weight_grad_accum[i], wcount);
                launch_check();
                zero<<<blocks_for(layer.output.c), kThreads>>>(bias_grad_accum[i], layer.output.c);
                launch_check();
                if (layer.batch_normalize) {
                    sgd_apply<<<blocks_for(layer.output.c), kThreads>>>(scales[i], scale_grad_accum[i], layer.output.c, lr, inv_batch);
                    launch_check();
                    zero<<<blocks_for(layer.output.c), kThreads>>>(scale_grad_accum[i], layer.output.c);
                    launch_check();
                }
            }
            accumulated = 0;
            ++iteration;
            updated = true;
        }
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize training");
        if (!std::isfinite(loss_result.loss)) throw std::runtime_error("non-finite CUDA training loss");
        return {loss_result.loss, targets.size(), updated};
    }

    float current_learning_rate() const { return schedule.at(iteration); }

    void sync_to_network() {
        const auto& layers = network.layers();
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].kind != Kind::convolutional) continue;
            auto& layer = network.layer(i);
            check(cudaMemcpy(layer.weights.data(), weights[i], layer.weights.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy learned weights");
            check(cudaMemcpy(layer.biases.data(), biases[i], layer.biases.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy learned biases");
            if (layer.batch_normalize) {
                check(cudaMemcpy(layer.scales.data(), scales[i], layer.scales.size() * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy learned scales");
                check(cudaMemcpy(layer.means.data(), running_mean[i], layer.means.size() * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy running mean");
                check(cudaMemcpy(layer.variances.data(), running_variance[i], layer.variances.size() * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy running variance");
            }
        }
    }
};

CudaTrainer::CudaTrainer(Network& network, LearningRateSchedule schedule, int batch_size, float weight_decay, float bn_momentum)
    : impl_(std::make_unique<Impl>(network, std::move(schedule), batch_size, weight_decay, bn_momentum)) {}
CudaTrainer::~CudaTrainer() = default;
TrainResult CudaTrainer::step(const std::vector<float>& input, const std::vector<Target>& targets) {
    return impl_->step(input, targets);
}
float CudaTrainer::current_learning_rate() const { return impl_->current_learning_rate(); }
void CudaTrainer::sync_to_network() { impl_->sync_to_network(); }

}  // namespace detector
#endif
