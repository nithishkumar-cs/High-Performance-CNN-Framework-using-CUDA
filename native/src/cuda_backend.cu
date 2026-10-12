#include "detector/cuda_session.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace detector {
namespace {

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

struct DeviceMemory {
    std::vector<void*> allocations;
    ~DeviceMemory() { for (void* item : allocations) cudaFree(item); }
    float* floats(std::size_t count) {
        float* pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(float)), "cudaMalloc");
        allocations.push_back(pointer);
        return pointer;
    }
    float* upload(const std::vector<float>& host) {
        float* pointer = floats(host.size());
        check(cudaMemcpy(pointer, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        return pointer;
    }
};

__global__ void convolution(const float* input, const float* weights, const float* multiplier,
                            const float* offset, float* output, int icount, int ih, int iw,
                            int ocount, int oh, int ow, int kernel, int stride, int pad, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ocount * oh * ow;
    if (pos >= total) return;
    int x = pos % ow;
    int y = pos / ow % oh;
    int oc = pos / (ow * oh);
    float sum = 0;
    for (int ic = 0; ic < icount; ++ic)
        for (int ky = 0; ky < kernel; ++ky)
            for (int kx = 0; kx < kernel; ++kx) {
                int iy = y * stride + ky - pad;
                int ix = x * stride + kx - pad;
                if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
                std::size_t wi = ((static_cast<std::size_t>(oc) * icount + ic) * kernel + ky) * kernel + kx;
                sum += input[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix] * weights[wi];
            }
    float value = sum * multiplier[oc] + offset[oc];
    output[pos] = leaky && value < 0 ? value * 0.1f : value;
}

__global__ void max_pool(const float* input, float* output, int channels, int ih, int iw,
                         int oh, int ow, int kernel, int stride) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * oh * ow;
    if (pos >= total) return;
    int x = pos % ow;
    int y = pos / ow % oh;
    int c = pos / (ow * oh);
    int pad = (kernel - 1) / 2;
    float best = -3.402823466e+38f;
    for (int ky = 0; ky < kernel; ++ky)
        for (int kx = 0; kx < kernel; ++kx) {
            int iy = y * stride + ky - pad;
            int ix = x * stride + kx - pad;
            if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
            best = fmaxf(best, input[(static_cast<std::size_t>(c) * ih + iy) * iw + ix]);
        }
    output[pos] = best;
}

__global__ void upsample(const float* input, float* output, int channels, int ih, int iw,
                         int oh, int ow, int stride) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= channels * oh * ow) return;
    int x = pos % ow;
    int y = pos / ow % oh;
    int c = pos / (ow * oh);
    output[pos] = input[(static_cast<std::size_t>(c) * ih + y / stride) * iw + x / stride];
}

__global__ void shortcut_add(const float* input, const float* branch, float* output, int count, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    float value = input[pos] + branch[pos];
    output[pos] = leaky && value < 0 ? value * 0.1f : value;
}

// Space-to-depth rearrangement (see network.cpp's reorg_forward for the shared convention this
// mirrors); one thread per OUTPUT element, each reading its one corresponding input position.
__global__ void reorg(const float* input, float* output, int channels, int ih, int iw, int stride) {
    int out_h = ih / stride, out_w = iw / stride, out_c = channels * stride * stride;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= out_c * out_h * out_w) return;
    int ox = pos % out_w, oy = pos / out_w % out_h, oc = pos / (out_w * out_h);
    int ic = oc / (stride * stride), offset = oc % (stride * stride);
    int ix = ox * stride + offset % stride, iy = oy * stride + offset / stride;
    output[pos] = input[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix];
}

__global__ void connected(const float* input, const float* weights, const float* bias,
                          float* output, int inputs, int outputs, bool leaky) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= outputs) return;
    float value = bias[oc];
    for (int ic = 0; ic < inputs; ++ic)
        value += input[ic] * weights[static_cast<std::size_t>(oc) * inputs + ic];
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

void launch_check() { check(cudaGetLastError(), "CUDA kernel launch"); }

}  // namespace

struct CudaSession::Impl {
    struct ConvolutionParameters { float* weights = nullptr; float* scales = nullptr; float* offsets = nullptr; };
    const Network& network;
    DeviceMemory memory;
    float* input = nullptr;
    std::vector<float*> output;
    std::vector<ConvolutionParameters> parameters;

    explicit Impl(const Network& net) : network(net) {
        int devices = 0;
        check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices < 1) throw std::runtime_error("no CUDA device available");
        input = memory.floats(network.input_shape().elements());
        const auto& layers = network.layers();
        output.resize(layers.size());
        parameters.resize(layers.size());
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            output[i] = memory.floats(layer.output.elements());
            if (layer.kind != Kind::convolutional && layer.kind != Kind::connected) continue;
            if (layer.biases.size() != static_cast<std::size_t>(layer.filters) || layer.weights.empty())
                throw std::runtime_error("load weights before constructing CudaSession");
            std::vector<float> multiplier(static_cast<std::size_t>(layer.filters), 1.0f);
            std::vector<float> offsets = layer.biases;
            if (layer.kind == Kind::convolutional && layer.batch_normalize)
                for (int c = 0; c < layer.filters; ++c) {
                    multiplier[c] = layer.scales[c] / std::sqrt(layer.variances[c] + 1e-5f);
                    offsets[c] -= layer.means[c] * multiplier[c];
                }
            parameters[i] = {memory.upload(layer.weights), memory.upload(multiplier), memory.upload(offsets)};
        }
    }

    std::vector<Head> forward(const std::vector<float>& host_input) {
        if (host_input.size() != network.input_shape().elements()) throw std::runtime_error("input does not match cfg shape");
        check(cudaMemcpy(input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy input");
        const auto& layers = network.layers();
        std::vector<Head> heads;
        constexpr int threads = 256;
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const auto& layer = layers[index];
            const float* previous = index ? output[index - 1] : input;
            float* result = output[index];
            int elements = static_cast<int>(layer.output.elements());
            int blocks = (elements + threads - 1) / threads;
            if (layer.kind == Kind::convolutional) {
                const auto& param = parameters[index];
                convolution<<<blocks, threads>>>(previous, param.weights, param.scales, param.offsets, result,
                    layer.input.c, layer.input.h, layer.input.w, layer.output.c, layer.output.h,
                    layer.output.w, layer.size, layer.stride, layer.padding, layer.leaky);
                launch_check();
            } else if (layer.kind == Kind::maxpool) {
                max_pool<<<blocks, threads>>>(previous, result, layer.input.c, layer.input.h,
                    layer.input.w, layer.output.h, layer.output.w, layer.size, layer.stride);
                launch_check();
            } else if (layer.kind == Kind::route) {
                std::size_t offset = 0;
                for (int branch : layer.sources) {
                    std::size_t count = layers[static_cast<std::size_t>(branch)].output.elements();
                    check(cudaMemcpy(result + offset, output[static_cast<std::size_t>(branch)],
                                     count * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy route");
                    offset += count;
                }
            } else if (layer.kind == Kind::upsample) {
                upsample<<<blocks, threads>>>(previous, result, layer.input.c, layer.input.h,
                    layer.input.w, layer.output.h, layer.output.w, layer.stride);
                launch_check();
            } else if (layer.kind == Kind::shortcut) {
                shortcut_add<<<blocks, threads>>>(previous, output[static_cast<std::size_t>(layer.sources.front())],
                    result, elements, layer.leaky);
                launch_check();
            } else if (layer.kind == Kind::connected) {
                const auto& param = parameters[index];
                connected<<<blocks, threads>>>(previous, param.weights, param.offsets, result,
                    static_cast<int>(layer.input.elements()), layer.filters, layer.leaky);
                launch_check();
            } else if (layer.kind == Kind::dropout) {
                check(cudaMemcpy(result, previous, layer.output.elements() * sizeof(float), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy dropout");  // no-op at inference, matching Darknet
            } else if (layer.kind == Kind::reorg) {
                reorg<<<blocks, threads>>>(previous, result, layer.input.c, layer.input.h, layer.input.w, layer.stride);
                launch_check();
            } else if (layer.kind == Kind::softmax) {
                softmax_vector<<<1, 1>>>(previous, result, elements);
                launch_check();
                Head head;
                head.layer = index;
                head.values.resize(layer.output.elements());
                check(cudaMemcpy(head.values.data(), result, head.values.size() * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy softmax output");
                heads.push_back(std::move(head));
            } else {
                check(cudaMemcpy(result, previous, layer.output.elements() * sizeof(float), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy YOLO head");
                Head head;
                head.layer = index;
                head.values.resize(layer.output.elements());
                check(cudaMemcpy(head.values.data(), result, head.values.size() * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy YOLO output");
                heads.push_back(std::move(head));
            }
        }
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        return heads;
    }
};

CudaSession::CudaSession(const Network& network) : impl_(std::make_unique<Impl>(network)) {}
CudaSession::~CudaSession() = default;
std::vector<Head> CudaSession::forward(const std::vector<float>& input) { return impl_->forward(input); }

std::vector<Head> forward_cuda(const Network& network, const std::vector<float>& input) {
    CudaSession session(network);
    return session.forward(input);
}

}  // namespace detector
