#include "detector/cuda_trainer.hpp"
#ifdef DETECTOR_HAS_CUDA
#include "detector/yolo_loss.hpp"
#include "l2norm_cuda.cuh"
#include "avgpool_cuda.cuh"
#ifdef DETECTOR_HAS_CUDNN
#include "detector/cudnn_backend.hpp"
#endif
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace detector {
namespace {

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

#ifndef DETECTOR_HAS_CUDNN
/** Raises an exception when a cuBLAS operation fails. */
void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}
#endif

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
#ifndef DETECTOR_HAS_CUDNN
    /** Allocates tracked double precision scratch storage for parallel reductions. */
    double* doubles(std::size_t count) {
        double* pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(double)), "cudaMalloc doubles");
        allocations.push_back(pointer);
        return pointer;
    }
#endif
    float* upload(const std::vector<float>& host) {
        float* pointer = floats(host.size());
        check(cudaMemcpy(pointer, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        return pointer;
    }
    // Frees `existing` (if non-null) and replaces it with a fresh zeroed buffer of `count`
    // elements, for multiscale resize: only the spatial-dimension-sized activation/gradient
    // buffers need this; per-channel buffers (weights, BN stats, velocities) are untouched by a
    // resize since channel counts never change.
    void replace(float*& existing, std::size_t count) {
        if (existing) {
            auto it = std::find(allocations.begin(), allocations.end(), static_cast<void*>(existing));
            if (it != allocations.end()) allocations.erase(it);
            cudaFree(existing);
            existing = nullptr;
        }
        existing = floats(count);
    }
    void replace(int*& existing, std::size_t count) {
        if (existing) {
            auto it = std::find(allocations.begin(), allocations.end(), static_cast<void*>(existing));
            if (it != allocations.end()) allocations.erase(it);
            cudaFree(existing);
            existing = nullptr;
        }
        existing = ints(count);
    }
#ifndef DETECTOR_HAS_CUDNN
    /** Replaces a tracked scratch allocation after the network changes spatial size. */
    void replace(double*& existing, std::size_t count) {
        if (existing) {
            auto it = std::find(allocations.begin(), allocations.end(), static_cast<void*>(existing));
            if (it != allocations.end()) allocations.erase(it);
            cudaFree(existing);
            existing = nullptr;
        }
        existing = doubles(count);
    }
#endif
};

#ifndef DETECTOR_HAS_CUDNN
constexpr float kBnEps = 1e-5f;
#endif
constexpr int kThreads = 256;
#ifndef DETECTOR_HAS_CUDNN
constexpr int kReductionTile = 4096;
#endif
inline int blocks_for(int count) { return count > 0 ? (count + kThreads - 1) / kThreads : 1; }

#ifdef DETECTOR_HAS_CUDNN
/** Applies bias and leaky activation to cuDNN's raw convolution output. */
__global__ void convolution_bias_activation(const float* raw, const float* bias,
                                             float* preactivation, float* output,
                                             int count, int plane, int channels, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    float value = raw[pos] + bias[(pos / plane) % channels];
    preactivation[pos] = value;
    output[pos] = leaky && value < 0 ? value * 0.1f : value;
}

/** Applies the activation after cuDNN's batch-normalization affine transform. */
__global__ void batch_norm_activation(const float* preactivation, float* output, int count, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    float value = preactivation[pos];
    output[pos] = leaky && value < 0 ? value * 0.1f : value;
}

/** Recovers biased batch variance from cuDNN's saved inverse standard deviation. */
__global__ void inverse_variance_to_variance(const float* inverse, float* variance, int count) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    variance[pos] = fmaxf(0.0f, 1.0f / (inverse[pos] * inverse[pos]) - 1e-5f);
}
#endif

#ifndef DETECTOR_HAS_CUDNN
/** Expands one CHW image into a position-major matrix for cuBLAS convolutions. */
__global__ void convolution_im2col(const float* input, float* columns, int channels, int ih, int iw,
                                   int oh, int ow, int kernel, int stride, int pad) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = oh * ow;
    int count = channels * kernel * kernel * plane;
    if (pos >= count) return;
    int pixel = pos % plane;
    int key = pos / plane;
    int kx = key % kernel, ky = key / kernel % kernel, ic = key / (kernel * kernel);
    int iy = pixel / ow * stride + ky - pad, ix = pixel % ow * stride + kx - pad;
    columns[pos] = iy >= 0 && ix >= 0 && iy < ih && ix < iw
        ? input[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix] : 0.0f;
}

/** Gathers the column gradient into each input pixel without contended atomics. */
__global__ void convolution_col2im_add(const float* columns, float* previous_gradient,
                                       int channels, int ih, int iw, int oh, int ow,
                                       int kernel, int stride, int pad) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= channels * ih * iw) return;
    int ix = pos % iw, iy = pos / iw % ih, ic = pos / (iw * ih);
    int plane = oh * ow;
    float sum = 0.0f;
    for (int ky = 0; ky < kernel; ++ky)
        for (int kx = 0; kx < kernel; ++kx) {
            int row = iy + pad - ky, col = ix + pad - kx;
            if (row < 0 || col < 0 || row % stride || col % stride) continue;
            int oy = row / stride, ox = col / stride;
            if (oy >= oh || ox >= ow) continue;
            int key = (ic * kernel + ky) * kernel + kx;
            sum += columns[static_cast<std::size_t>(key) * plane + oy * ow + ox];
        }
    previous_gradient[pos] += sum;
}

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

/** Reduces one sample/channel tile in parallel for batch mean and variance. */
__global__ void batch_norm_statistics_partial(const float* raw, double* partial, int filters,
                                              int plane, int sample_stride, int chunks) {
    int chunk = blockIdx.x, oc = blockIdx.y, sample = blockIdx.z;
    int start = chunk * kReductionTile, end = min(start + kReductionTile, plane);
    const float* channel = raw + static_cast<std::size_t>(sample) * sample_stride + static_cast<std::size_t>(oc) * plane;
    double sum = 0.0, square = 0.0;
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        double value = channel[p];
        sum += value;
        square += value * value;
    }
    __shared__ double sums[kThreads], squares[kThreads];
    sums[threadIdx.x] = sum;
    squares[threadIdx.x] = square;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset /= 2) {
        if (threadIdx.x < offset) {
            sums[threadIdx.x] += sums[threadIdx.x + offset];
            squares[threadIdx.x] += squares[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        std::size_t index = ((static_cast<std::size_t>(sample) * filters + oc) * chunks + chunk) * 2;
        partial[index] = sums[0];
        partial[index + 1] = squares[0];
    }
}

/** Combines tile sums using double precision before storing float BN statistics. */
__global__ void batch_norm_statistics_finish(const double* partial, float* mean, float* variance,
                                             int filters, int plane, int n, int chunks) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= filters) return;
    double sum = 0.0, square = 0.0;
    for (int s = 0; s < n; ++s)
        for (int chunk = 0; chunk < chunks; ++chunk) {
            std::size_t index = ((static_cast<std::size_t>(s) * filters + oc) * chunks + chunk) * 2;
            sum += partial[index];
            square += partial[index + 1];
        }
    double count = static_cast<double>(n) * plane;
    double m = sum / count;
    mean[oc] = static_cast<float>(m);
    variance[oc] = static_cast<float>(fmax(0.0, square / count - m * m));
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
#endif

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

// Space-to-depth rearrangement (see network.cpp's reorg_forward for the shared convention this
// mirrors, independently re-derived here since this translation unit cannot call into it); a
// bijection with no learnable parameters, so forward/backward each need only one thread per
// element with no atomics (every thread targets a distinct destination).
__global__ void reorg_forward_kernel(const float* input, float* output, int channels, int ih, int iw, int stride) {
    int out_h = ih / stride, out_w = iw / stride, out_c = channels * stride * stride;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= out_c * out_h * out_w) return;
    int ox = pos % out_w, oy = pos / out_w % out_h, oc = pos / (out_w * out_h);
    int ic = oc / (stride * stride), offset = oc % (stride * stride);
    int ix = ox * stride + offset % stride, iy = oy * stride + offset / stride;
    output[pos] = input[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix];
}

__global__ void reorg_backward_kernel(const float* gradient, float* previous_gradient, int channels,
                                      int ih, int iw, int stride) {
    int out_h = ih / stride, out_w = iw / stride, out_c = channels * stride * stride;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= out_c * out_h * out_w) return;
    int ox = pos % out_w, oy = pos / out_w % out_h, oc = pos / (out_w * out_h);
    int ic = oc / (stride * stride), offset = oc % (stride * stride);
    int ix = ox * stride + offset % stride, iy = oy * stride + offset / stride;
    previous_gradient[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix] += gradient[pos];
}

// Non-cryptographic xorshift hash -> uniform float in [0,1); only needs to look random enough
// for dropout's keep/drop decision, not withstand adversarial analysis. `seed` is derived from
// (iteration, layer, sample, position) so every element gets an independent-looking draw and
// every step() call redraws a fresh mask - this is a real (not --seed-reproducible) source of
// randomness, matching Darknet's own rand()-based dropout and this project's CPU Trainer (see
// trainer.hpp's dropout_rng_), not CPU/GPU bit-identical since the two use unrelated algorithms.
__device__ __forceinline__ float hash_uniform(unsigned int seed) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return (seed & 0x00FFFFFFu) / 16777216.0f;
}

// Inverted dropout: kept elements scaled by 1/(1-p). `scale_out` stashes the per-element 0 or
// keep_scale decision so dropout_backward reuses the exact mask this forward pass drew.
__global__ void dropout_forward(const float* input, float* output, float* scale_out, int count,
                                float probability, float keep_scale, unsigned int seed) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    float scale = hash_uniform(seed + static_cast<unsigned int>(pos) * 2654435761u) < probability ? 0.0f : keep_scale;
    scale_out[pos] = scale;
    output[pos] = input[pos] * scale;
}

__global__ void dropout_backward(const float* gradient, const float* scale, float* previous_gradient, int count) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    previous_gradient[pos] += gradient[pos] * scale[pos];
}

__global__ void activation_backward(float* gradient, const float* preactivation, int count, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < count && leaky && preactivation[pos] < 0) gradient[pos] *= 0.1f;
}

#ifndef DETECTOR_HAS_CUDNN
__global__ void batch_norm_backward_partial(const float* g, const float* raw, const float* mean,
                                            const float* variance, const float* gamma, double* partial,
                                            int filters, int plane, int sample_stride, int chunks) {
    int chunk = blockIdx.x, oc = blockIdx.y, sample = blockIdx.z;
    int start = chunk * kReductionTile, end = min(start + kReductionTile, plane);
    const float* gc = g + static_cast<std::size_t>(sample) * sample_stride + static_cast<std::size_t>(oc) * plane;
    const float* rc = raw + static_cast<std::size_t>(sample) * sample_stride + static_cast<std::size_t>(oc) * plane;
    float inv_std = 1.0f / sqrtf(variance[oc] + kBnEps);
    double sd = 0.0, sdx = 0.0, dgamma = 0.0, dbeta = 0.0;
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        float xhat = (rc[p] - mean[oc]) * inv_std;
        float dxhat = gc[p] * gamma[oc];
        sd += dxhat;
        sdx += dxhat * xhat;
        dgamma += gc[p] * xhat;
        dbeta += gc[p];
    }
    __shared__ double sums[4][kThreads];
    sums[0][threadIdx.x] = sd;
    sums[1][threadIdx.x] = sdx;
    sums[2][threadIdx.x] = dgamma;
    sums[3][threadIdx.x] = dbeta;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset /= 2) {
        if (threadIdx.x < offset)
            for (int term = 0; term < 4; ++term)
                sums[term][threadIdx.x] += sums[term][threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        std::size_t index = ((static_cast<std::size_t>(sample) * filters + oc) * chunks + chunk) * 4;
        for (int term = 0; term < 4; ++term) partial[index + term] = sums[term][0];
    }
}

/** Combines BN gradient tiles and accumulates the scale and bias gradients. */
__global__ void batch_norm_backward_finish(const double* partial, float* sum_dxhat,
                                           float* sum_dxhat_xhat, float* scale_grad_accum,
                                           float* bias_grad_accum, int filters, int n, int chunks) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= filters) return;
    double sums[4] = {0.0, 0.0, 0.0, 0.0};
    for (int s = 0; s < n; ++s)
        for (int chunk = 0; chunk < chunks; ++chunk) {
            std::size_t index = ((static_cast<std::size_t>(s) * filters + oc) * chunks + chunk) * 4;
            for (int term = 0; term < 4; ++term) sums[term] += partial[index + term];
        }
    sum_dxhat[oc] = static_cast<float>(sums[0]);
    sum_dxhat_xhat[oc] = static_cast<float>(sums[1]);
    scale_grad_accum[oc] += static_cast<float>(sums[2]);
    bias_grad_accum[oc] += static_cast<float>(sums[3]);
}

// Second pass: per output element of one sample, turns g into dz (gradient w.r.t. the pre-norm
// conv sum), overwriting `g` in place so the same buffer feeds conv_backward_input/weight below.
// `plane` is this sample's per-channel spatial size (for the oc lookup); `total_n` is the true
// batch element count per channel (n * plane) the mean/variance/sums were computed over, which
// is the correct normalizer here once n > 1 (equal to `plane` when n == 1, so behavior for a
// single-sample call is unchanged).
__global__ void batch_norm_backward_dz(float* g, const float* raw, const float* mean,
                                       const float* variance, const float* gamma,
                                       const float* sum_dxhat, const float* sum_dxhat_xhat,
                                       int count, int plane, int total_n) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    int oc = pos / plane;
    float inv_std = 1.0f / sqrtf(variance[oc] + kBnEps);
    float xhat = (raw[pos] - mean[oc]) * inv_std;
    float dxhat = g[pos] * gamma[oc];
    g[pos] = inv_std / total_n * (total_n * dxhat - sum_dxhat[oc] - xhat * sum_dxhat_xhat[oc]);
}
#endif

__global__ void bias_grad_reduce(const float* dz, float* bias_grad_accum, int filters, int plane) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= filters) return;
    const float* dc = dz + static_cast<std::size_t>(oc) * plane;
    float sum = 0;
    for (int p = 0; p < plane; ++p) sum += dc[p];
    bias_grad_accum[oc] += sum;
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

// Classic SGD momentum: velocity = momentum*velocity + this step's averaged gradient (plus
// weight decay for weights), then param -= lr*velocity. `velocity` persists across calls and
// starts zeroed, so the first call is identical to plain SGD regardless of `momentum`.
__global__ void sgd_apply_momentum(float* param, const float* grad_accum, float* velocity, int count,
                                   float lr, float inv_batch, float momentum) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    float v = momentum * velocity[pos] + grad_accum[pos] * inv_batch;
    velocity[pos] = v;
    param[pos] -= lr * v;
}

__global__ void sgd_apply_with_decay_momentum(float* param, const float* grad_accum, float* velocity, int count,
                                              float lr, float inv_batch, float decay, float momentum) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    float v = momentum * velocity[pos] + grad_accum[pos] * inv_batch + decay * param[pos];
    velocity[pos] = v;
    param[pos] -= lr * v;
}

__global__ void zero(float* buffer, int count) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < count) buffer[pos] = 0.0f;
}

}  // namespace

/** Owns CUDA training state and a cuBLAS workspace that is reused across layers. */
struct CudaTrainer::Impl {
    Network& network;
    LearningRateSchedule schedule;
    int batch_size;
    float weight_decay;
    float momentum;
    float bn_momentum;
    int iteration;
    DeviceMemory memory;
#ifndef DETECTOR_HAS_CUDNN
    cublasHandle_t blas = nullptr;
    std::size_t column_capacity = 1;
    float* columns = nullptr;
    float* column_gradient = nullptr;
    std::size_t bn_partial_capacity = 1;
    double* bn_partial = nullptr;
#endif
    float* input = nullptr;
    std::vector<float*> output, preactivation, gradient;
    std::vector<float*> raw, batch_mean, batch_var, sum_dxhat, running_mean, running_variance;
#ifndef DETECTOR_HAS_CUDNN
    std::vector<float*> sum_dxhat_xhat;
#endif
    std::vector<float*> weights, biases, scales;
    std::vector<float*> weight_grad_accum, bias_grad_accum, scale_grad_accum;
    std::vector<float*> weight_velocity, bias_velocity, scale_velocity;
    std::vector<int*> max_index;
#ifdef DETECTOR_HAS_CUDNN
    std::vector<std::unique_ptr<CudnnConvolution>> cudnn_convolutions;
    std::vector<std::unique_ptr<CudnnBatchNorm>> cudnn_batch_norms;
#endif

    /** Initializes parameters, spatial buffers, and a full-precision matrix multiplication handle. */
    Impl(Network& net, LearningRateSchedule sched, int batch, float decay, float sgd_momentum, float bn_mom,
         int initial_iteration)
        : network(net), schedule(std::move(sched)), batch_size(batch), weight_decay(decay), momentum(sgd_momentum),
          bn_momentum(bn_mom), iteration(initial_iteration) {
        if (!std::isfinite(schedule.base_lr) || schedule.base_lr <= 0 || schedule.burn_in < 0 ||
            schedule.steps.size() != schedule.scales.size())
            throw std::runtime_error("invalid learning rate schedule");
        if (!std::isfinite(decay) || decay < 0) throw std::runtime_error("invalid weight decay");
        if (batch < 1) throw std::runtime_error("invalid batch size");
        if (initial_iteration < 0) throw std::runtime_error("invalid initial iteration");
        if (!std::isfinite(sgd_momentum) || sgd_momentum < 0 || sgd_momentum >= 1) throw std::runtime_error("invalid momentum");
        if (!std::isfinite(bn_mom) || bn_mom <= 0 || bn_mom >= 1) throw std::runtime_error("invalid batch-norm momentum");
        int devices = 0;
        check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices < 1) throw std::runtime_error("no CUDA device available");
        const auto& layers = net.layers();
        for (const auto& layer : layers) {
            if (layer.kind == Kind::rnn || layer.kind == Kind::lstm || layer.kind == Kind::gru)
                throw std::runtime_error("CUDA detector Trainer does not support rnn/lstm/gru layers; "
                                         "use CudaClassificationTrainer::train_sequence");
            if (layer.kind != Kind::convolutional && layer.kind != Kind::maxpool && layer.kind != Kind::avgpool && layer.kind != Kind::route &&
                layer.kind != Kind::upsample && layer.kind != Kind::shortcut && layer.kind != Kind::yolo &&
                layer.kind != Kind::dropout && layer.kind != Kind::reorg && layer.kind != Kind::region &&
                layer.kind != Kind::l2norm)
                throw std::runtime_error("CUDA detector training supports convolutional, maxpool, avgpool, route, "
                                           "upsample, shortcut, yolo, region, dropout, reorg and l2norm layers only");
        }
        // input/output/preactivation/gradient/raw/max_index are sized for `batch` sample blocks
        // laid out contiguously (sample-major) so a true minibatch's cross-sample batch-norm
        // statistics (and every other per-sample forward/backward op) can run within one step().
        const auto batch_elems = static_cast<std::size_t>(batch);
        input = memory.floats(batch_elems * net.input_shape().elements());
        std::size_t n = layers.size();
        output.resize(n); preactivation.resize(n, nullptr); gradient.resize(n);
        raw.resize(n, nullptr); batch_mean.resize(n, nullptr); batch_var.resize(n, nullptr);
        sum_dxhat.resize(n, nullptr);
#ifndef DETECTOR_HAS_CUDNN
        sum_dxhat_xhat.resize(n, nullptr);
#endif
        running_mean.resize(n, nullptr); running_variance.resize(n, nullptr);
        weights.resize(n, nullptr); biases.resize(n, nullptr); scales.resize(n, nullptr);
        weight_grad_accum.resize(n, nullptr); bias_grad_accum.resize(n, nullptr); scale_grad_accum.resize(n, nullptr);
        weight_velocity.resize(n, nullptr); bias_velocity.resize(n, nullptr); scale_velocity.resize(n, nullptr);
        max_index.resize(n, nullptr);
#ifdef DETECTOR_HAS_CUDNN
        cudnn_convolutions.resize(n);
        cudnn_batch_norms.resize(n);
#endif
        for (std::size_t i = 0; i < n; ++i) {
            const auto& layer = layers[i];
            output[i] = memory.floats(batch_elems * layer.output.elements());
            gradient[i] = memory.floats(batch_elems * layer.output.elements());
            if (layer.kind == Kind::maxpool) { max_index[i] = memory.ints(batch_elems * layer.output.elements()); continue; }
            // Dropout reuses the shortcut-style `preactivation` slot to stash each element's
            // per-step 0/keep_scale draw, so backward replays the exact mask forward drew.
            if (layer.kind == Kind::shortcut || layer.kind == Kind::dropout) {
                preactivation[i] = memory.floats(batch_elems * layer.output.elements());
                continue;
            }
            if (layer.kind != Kind::convolutional) continue;
            preactivation[i] = memory.floats(batch_elems * layer.output.elements());
            std::size_t fan_in = static_cast<std::size_t>(layer.input.c) * layer.size * layer.size;
#ifndef DETECTOR_HAS_CUDNN
            column_capacity = std::max(column_capacity, fan_in * static_cast<std::size_t>(layer.output.h) * layer.output.w);
#else
            cudnn_convolutions[i] = std::make_unique<CudnnConvolution>(layer.input.c, layer.input.h,
                layer.input.w, layer.output.c, layer.size, layer.stride, layer.padding);
#endif
            if (layer.weights.size() != fan_in * layer.output.c || layer.biases.size() != static_cast<std::size_t>(layer.output.c))
                throw std::runtime_error("initialize or load detector weights before CUDA training");
            weights[i] = memory.upload(layer.weights);
            biases[i] = memory.upload(layer.biases);
            weight_grad_accum[i] = memory.floats(layer.weights.size());
            bias_grad_accum[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
            weight_velocity[i] = memory.floats(layer.weights.size());
            bias_velocity[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
            if (layer.batch_normalize) {
#ifndef DETECTOR_HAS_CUDNN
                int chunks = (layer.output.h * layer.output.w + kReductionTile - 1) / kReductionTile;
                bn_partial_capacity = std::max(bn_partial_capacity, batch_elems * layer.output.c * chunks * 4);
#else
                cudnn_batch_norms[i] = std::make_unique<CudnnBatchNorm>(layer.output.c,
                    layer.output.h, layer.output.w);
#endif
                if (layer.scales.size() != static_cast<std::size_t>(layer.output.c) ||
                    layer.means.size() != static_cast<std::size_t>(layer.output.c) ||
                    layer.variances.size() != static_cast<std::size_t>(layer.output.c))
                    throw std::runtime_error("initialize or load batch-norm parameters before CUDA training");
                raw[i] = memory.floats(batch_elems * layer.output.elements());
                batch_mean[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
                batch_var[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
                sum_dxhat[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
#ifndef DETECTOR_HAS_CUDNN
                sum_dxhat_xhat[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
#endif
                scales[i] = memory.upload(layer.scales);
                scale_grad_accum[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
                scale_velocity[i] = memory.floats(static_cast<std::size_t>(layer.output.c));
                running_mean[i] = memory.upload(layer.means);
                running_variance[i] = memory.upload(layer.variances);
            }
        }
#ifndef DETECTOR_HAS_CUDNN
        columns = memory.floats(column_capacity);
        column_gradient = memory.floats(column_capacity);
        bn_partial = memory.doubles(bn_partial_capacity);
        check(cublasCreate(&blas), "cublasCreate");
        check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
#endif
    }

#ifndef DETECTOR_HAS_CUDNN
    /** Releases the matrix multiplication handle before tracked CUDA allocations. */
    ~Impl() { if (blas) cublasDestroy(blas); }
#endif

    // Darknet-style multiscale resize: recomputes every layer's shape on `network` for a new
    // input width/height, then reallocates only the spatial-dimension-sized buffers (input,
    // per-layer output/gradient/preactivation/raw, maxpool's max_index) at the new size. Every
    // per-channel buffer (weights, biases, scales, BN running stats, SGD velocities, gradient
    // accumulators) is untouched - resizing never changes a layer's channel count, only its
    // spatial extent - so learned parameters and momentum/BN state survive the resize exactly
    // like Darknet's own resize_network, unlike a destroy-and-reconstruct approach would.
    /** Resizes spatial buffers while preserving learned parameters and optimizer state. */
    void resize(int width, int height) {
        network.resize(width, height);
        const auto& layers = network.layers();
        memory.replace(input, static_cast<std::size_t>(batch_size) * network.input_shape().elements());
#ifndef DETECTOR_HAS_CUDNN
        column_capacity = 1;
        bn_partial_capacity = 1;
#endif
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            std::size_t elems = static_cast<std::size_t>(batch_size) * layer.output.elements();
            memory.replace(output[i], elems);
            memory.replace(gradient[i], elems);
            if (layer.kind == Kind::maxpool) { memory.replace(max_index[i], elems); continue; }
            if (layer.kind == Kind::shortcut || layer.kind == Kind::dropout) { memory.replace(preactivation[i], elems); continue; }
            if (layer.kind != Kind::convolutional) continue;
#ifdef DETECTOR_HAS_CUDNN
            cudnn_convolutions[i] = std::make_unique<CudnnConvolution>(layer.input.c, layer.input.h,
                layer.input.w, layer.output.c, layer.size, layer.stride, layer.padding);
            if (layer.batch_normalize)
                cudnn_batch_norms[i] = std::make_unique<CudnnBatchNorm>(layer.output.c,
                    layer.output.h, layer.output.w);
#else
            column_capacity = std::max(column_capacity, static_cast<std::size_t>(layer.input.c) * layer.size * layer.size *
                                                       layer.output.h * layer.output.w);
#endif
            memory.replace(preactivation[i], elems);
            if (layer.batch_normalize) {
                memory.replace(raw[i], elems);
#ifndef DETECTOR_HAS_CUDNN
                int chunks = (layer.output.h * layer.output.w + kReductionTile - 1) / kReductionTile;
                bn_partial_capacity = std::max(bn_partial_capacity,
                                               static_cast<std::size_t>(batch_size) * layer.output.c * chunks * 4);
#endif
            }
        }
#ifndef DETECTOR_HAS_CUDNN
        memory.replace(columns, column_capacity);
        memory.replace(column_gradient, column_capacity);
        memory.replace(bn_partial, bn_partial_capacity);
#endif
    }

    // `inputs.size()` (== n) must be between 1 and `batch_size`; device buffers are sample-major
    // blocks of `batch_size` capacity, so this uses only their first n blocks.
    /** Trains one minibatch and updates device-resident parameters once. */
    TrainResult step(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<Target>>& targets) {
        const std::size_t n = inputs.size();
        if (n == 0 || n > static_cast<std::size_t>(batch_size))
            throw std::runtime_error("step() requires between 1 and batch_size samples");
        if (targets.size() != n) throw std::runtime_error("inputs/targets length mismatch");
        const auto& layers = network.layers();
        std::size_t input_elems = network.input_shape().elements();
#ifndef DETECTOR_HAS_CUDNN
        const float one = 1.0f, zero_beta = 0.0f;
#endif
        for (std::size_t s = 0; s < n; ++s) {
            if (inputs[s].size() != input_elems) throw std::runtime_error("training input shape mismatch");
            check(cudaMemcpy(input + s * input_elems, inputs[s].data(), input_elems * sizeof(float), cudaMemcpyHostToDevice),
                  "cudaMemcpy training input");
        }

        // Forward: layer-major so every sample reaches a batch_normalize layer's raw pre-BN conv
        // output before that layer's true batch statistics (computed over all n samples) are
        // finalized and applied.
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            std::size_t elems = layer.output.elements();
            int elements = static_cast<int>(elems);
            std::size_t prev_elems = i ? layers[i - 1].output.elements() : input_elems;
            auto previous_of = [&](std::size_t s) -> const float* {
                return (i ? output[i - 1] : input) + s * prev_elems;
            };
            if (layer.kind == Kind::convolutional) {
                int plane = layer.output.h * layer.output.w;
                if (layer.batch_normalize) {
#ifdef DETECTOR_HAS_CUDNN
                    cudnn_convolutions[i]->forward(previous_of(0), weights[i], raw[i], static_cast<int>(n));
                    cudnn_batch_norms[i]->forward(raw[i], scales[i], biases[i], preactivation[i],
                        running_mean[i], running_variance[i], batch_mean[i], sum_dxhat[i], static_cast<int>(n));
                    inverse_variance_to_variance<<<blocks_for(layer.output.c), kThreads>>>(
                        sum_dxhat[i], batch_var[i], layer.output.c);
                    launch_check();
                    batch_norm_activation<<<blocks_for(static_cast<int>(n * elems)), kThreads>>>(
                        preactivation[i], output[i], static_cast<int>(n * elems), layer.leaky);
                    launch_check();
#else
                    int key_count = layer.input.c * layer.size * layer.size;
                    for (std::size_t s = 0; s < n; ++s) {
                        convolution_im2col<<<blocks_for(key_count * plane), kThreads>>>(previous_of(s), columns,
                            layer.input.c, layer.input.h, layer.input.w, layer.output.h, layer.output.w,
                            layer.size, layer.stride, layer.padding);
                        launch_check();
                        check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, plane, layer.output.c, key_count,
                                          &one, columns, plane, weights[i], key_count, &zero_beta, raw[i] + s * elems, plane),
                              "cublasSgemm forward");
                    }
                    int chunks = (plane + kReductionTile - 1) / kReductionTile;
                    dim3 grid(chunks, layer.output.c, static_cast<unsigned int>(n));
                    batch_norm_statistics_partial<<<grid, kThreads>>>(raw[i], bn_partial,
                        layer.output.c, plane, elements, chunks);
                    launch_check();
                    batch_norm_statistics_finish<<<blocks_for(layer.output.c), kThreads>>>(bn_partial,
                        batch_mean[i], batch_var[i], layer.output.c, plane, static_cast<int>(n), chunks);
                    launch_check();
                    for (std::size_t s = 0; s < n; ++s) {
                        batch_norm_apply<<<blocks_for(elements), kThreads>>>(raw[i] + s * elems, batch_mean[i], batch_var[i],
                            scales[i], biases[i], preactivation[i] + s * elems, output[i] + s * elems, elements, plane, layer.leaky);
                        launch_check();
                    }
#endif
                    update_running_stats<<<blocks_for(layer.output.c), kThreads>>>(running_mean[i], running_variance[i],
                        batch_mean[i], batch_var[i], layer.output.c, bn_momentum);
                    launch_check();
                } else {
#ifdef DETECTOR_HAS_CUDNN
                    cudnn_convolutions[i]->forward(previous_of(0), weights[i], preactivation[i], static_cast<int>(n));
                    convolution_bias_activation<<<blocks_for(static_cast<int>(n * elems)), kThreads>>>(
                        preactivation[i], biases[i], preactivation[i], output[i],
                        static_cast<int>(n * elems), plane, layer.output.c, layer.leaky);
                    launch_check();
#else
                    for (std::size_t s = 0; s < n; ++s) {
                        convolution_forward_plain<<<blocks_for(elements), kThreads>>>(previous_of(s), weights[i], biases[i],
                            preactivation[i] + s * elems, output[i] + s * elems, layer.input.c, layer.input.h, layer.input.w,
                            layer.output.c, layer.output.h, layer.output.w, layer.size, layer.stride, layer.padding, layer.leaky);
                        launch_check();
                    }
#endif
                }
            } else if (layer.kind == Kind::maxpool) {
                for (std::size_t s = 0; s < n; ++s) {
                    max_pool_forward<<<blocks_for(elements), kThreads>>>(previous_of(s), output[i] + s * elems, max_index[i] + s * elems,
                        layer.input.c, layer.input.h, layer.input.w, layer.output.h, layer.output.w, layer.size, layer.stride);
                    launch_check();
                }
            } else if (layer.kind == Kind::avgpool) {
                avgpool_forward_kernel<<<static_cast<int>(n) * layer.input.c, kThreads>>>(
                    previous_of(0), output[i], layer.input.c, layer.input.h * layer.input.w);
                launch_check();
            } else if (layer.kind == Kind::route) {
                for (std::size_t s = 0; s < n; ++s) {
                    std::size_t offset = 0;
                    for (int source : layer.sources) {
                        std::size_t count = layers[static_cast<std::size_t>(source)].output.elements();
                        check(cudaMemcpy(output[i] + s * elems + offset, output[static_cast<std::size_t>(source)] + s * count,
                                         count * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy route");
                        offset += count;
                    }
                }
            } else if (layer.kind == Kind::upsample) {
                for (std::size_t s = 0; s < n; ++s) {
                    upsample_forward<<<blocks_for(elements), kThreads>>>(previous_of(s), output[i] + s * elems, layer.input.c,
                        layer.input.h, layer.input.w, layer.output.h, layer.output.w, layer.stride);
                    launch_check();
                }
            } else if (layer.kind == Kind::shortcut) {
                std::size_t source = static_cast<std::size_t>(layer.sources.front());
                for (std::size_t s = 0; s < n; ++s) {
                    shortcut_forward<<<blocks_for(elements), kThreads>>>(previous_of(s), output[source] + s * elems,
                        preactivation[i] + s * elems, output[i] + s * elems, elements, layer.leaky);
                    launch_check();
                }
            } else if (layer.kind == Kind::dropout) {
                float keep_scale = 1.0f / (1.0f - layer.dropout_probability);
                for (std::size_t s = 0; s < n; ++s) {
                    // Seed mixes iteration/layer/sample so every call, layer and sample draws an
                    // independent-looking mask (a fresh one each step(), like Darknet's rand()).
                    unsigned int seed = static_cast<unsigned int>(iteration) * 2654435761u +
                                         static_cast<unsigned int>(i) * 40503u + static_cast<unsigned int>(s) * 97u + 12345u;
                    dropout_forward<<<blocks_for(elements), kThreads>>>(previous_of(s), output[i] + s * elems,
                        preactivation[i] + s * elems, elements, layer.dropout_probability, keep_scale, seed);
                    launch_check();
                }
            } else if (layer.kind == Kind::l2norm) {
                l2norm_forward_kernel<<<static_cast<int>(n), kThreads>>>(previous_of(0), output[i], elements);
                launch_check();
            } else if (layer.kind == Kind::reorg) {
                for (std::size_t s = 0; s < n; ++s) {
                    reorg_forward_kernel<<<blocks_for(elements), kThreads>>>(previous_of(s), output[i] + s * elems,
                        layer.input.c, layer.input.h, layer.input.w, layer.stride);
                    launch_check();
                }
            } else {  // yolo/region: pass-through, matching the CPU trainer's forward for terminal heads.
                for (std::size_t s = 0; s < n; ++s)
                    check(cudaMemcpy(output[i] + s * elems, previous_of(s), elems * sizeof(float),
                                     cudaMemcpyDeviceToDevice), "cudaMemcpy yolo forward");
            }
        }

        // Zero every used gradient slot (the first n sample blocks of each layer) before seeding
        // the YOLO heads' host-computed gradient, so backward's add_into kernels start from zero.
        for (std::size_t i = 0; i < layers.size(); ++i) {
            int count = static_cast<int>(n * layers[i].output.elements());
            zero<<<blocks_for(count), kThreads>>>(gradient[i], count);
            launch_check();
        }

        // YOLO loss: shared host-side assignment/gradient (identical math to the CPU trainer),
        // computed per sample on only the small head tensors downloaded here.
        float total_loss = 0;
        std::size_t total_positives = 0;
        std::vector<std::vector<float>> host_output(layers.size()), host_gradient(layers.size());
        for (std::size_t s = 0; s < n; ++s) {
            for (std::size_t i = 0; i < layers.size(); ++i) {
                if (layers[i].kind != Kind::yolo && layers[i].kind != Kind::region) continue;
                std::size_t elems = layers[i].output.elements();
                host_output[i].resize(elems);
                check(cudaMemcpy(host_output[i].data(), output[i] + s * elems, elems * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy yolo output");
                host_gradient[i].assign(elems, 0.0f);
            }
            auto loss_result = yolo_loss_and_gradient(network, targets[s], host_output, host_gradient);
            total_loss += loss_result.loss;
            total_positives += loss_result.positives;
            for (std::size_t i = 0; i < layers.size(); ++i) {
                if (layers[i].kind != Kind::yolo && layers[i].kind != Kind::region) continue;
                std::size_t elems = layers[i].output.elements();
                check(cudaMemcpy(gradient[i] + s * elems, host_gradient[i].data(), elems * sizeof(float),
                                 cudaMemcpyHostToDevice), "cudaMemcpy yolo gradient");
            }
        }

        // Backward: traverse layers in reverse; each kernel accumulates into its predecessor(s),
        // since sequential launches on one stream see each other's prior writes.
        for (std::size_t reverse = layers.size(); reverse-- > 0;) {
            const auto& layer = layers[reverse];
            std::size_t elems = layer.output.elements();
            int elements = static_cast<int>(elems);
            std::size_t prev_elems = reverse ? layers[reverse - 1].output.elements() : input_elems;
            auto previous_of = [&](std::size_t s) -> const float* {
                return (reverse ? output[reverse - 1] : input) + s * prev_elems;
            };
            auto prev_grad_of = [&](std::size_t s) -> float* { return gradient[reverse - 1] + s * prev_elems; };
            if (layer.kind == Kind::yolo || layer.kind == Kind::region) {
                if (reverse)
                    for (std::size_t s = 0; s < n; ++s) {
                        add_into<<<blocks_for(elements), kThreads>>>(prev_grad_of(s), gradient[reverse] + s * elems, elements);
                        launch_check();
                    }
            } else if (layer.kind == Kind::route) {
                for (std::size_t s = 0; s < n; ++s) {
                    std::size_t offset = 0;
                    for (int source : layer.sources) {
                        int count = static_cast<int>(layers[static_cast<std::size_t>(source)].output.elements());
                        add_into<<<blocks_for(count), kThreads>>>(gradient[static_cast<std::size_t>(source)] + s * static_cast<std::size_t>(count),
                            gradient[reverse] + s * elems + offset, count);
                        launch_check();
                        offset += static_cast<std::size_t>(count);
                    }
                }
            } else if (layer.kind == Kind::upsample) {
                if (reverse)
                    for (std::size_t s = 0; s < n; ++s) {
                        upsample_backward<<<blocks_for(elements), kThreads>>>(gradient[reverse] + s * elems, prev_grad_of(s),
                            layer.input.c, layer.input.h, layer.input.w, layer.output.h, layer.output.w, layer.stride);
                        launch_check();
                    }
            } else if (layer.kind == Kind::shortcut) {
                if (!reverse) throw std::runtime_error("shortcut cannot be the first layer");
                std::size_t source = static_cast<std::size_t>(layer.sources.front());
                for (std::size_t s = 0; s < n; ++s) {
                    float* grad = gradient[reverse] + s * elems;
                    activation_backward<<<blocks_for(elements), kThreads>>>(grad, preactivation[reverse] + s * elems, elements, layer.leaky);
                    launch_check();
                    shortcut_backward<<<blocks_for(elements), kThreads>>>(grad, prev_grad_of(s), gradient[source] + s * elems, elements);
                    launch_check();
                }
            } else if (layer.kind == Kind::maxpool) {
                if (reverse)
                    for (std::size_t s = 0; s < n; ++s) {
                        max_pool_backward<<<blocks_for(elements), kThreads>>>(gradient[reverse] + s * elems, max_index[reverse] + s * elems,
                            prev_grad_of(s), elements);
                        launch_check();
                    }
            } else if (layer.kind == Kind::avgpool) {
                if (reverse) {
                    const int spatial = layer.input.h * layer.input.w;
                    const int count = static_cast<int>(n * layer.input.elements());
                    avgpool_backward_kernel<<<blocks_for(count), kThreads>>>(
                        gradient[reverse], prev_grad_of(0), spatial, count);
                    launch_check();
                }
            } else if (layer.kind == Kind::dropout) {
                if (reverse)
                    for (std::size_t s = 0; s < n; ++s) {
                        dropout_backward<<<blocks_for(elements), kThreads>>>(gradient[reverse] + s * elems,
                            preactivation[reverse] + s * elems, prev_grad_of(s), elements);
                        launch_check();
                    }
            } else if (layer.kind == Kind::l2norm) {
                if (reverse) {
                    l2norm_backward_kernel<<<static_cast<int>(n), kThreads>>>(previous_of(0),
                        output[reverse], gradient[reverse], prev_grad_of(0), elements);
                    launch_check();
                }
            } else if (layer.kind == Kind::reorg) {
                if (reverse)
                    for (std::size_t s = 0; s < n; ++s) {
                        reorg_backward_kernel<<<blocks_for(elements), kThreads>>>(gradient[reverse] + s * elems, prev_grad_of(s),
                            layer.input.c, layer.input.h, layer.input.w, layer.stride);
                        launch_check();
                    }
            } else {  // convolutional
                int plane = layer.output.h * layer.output.w;
                for (std::size_t s = 0; s < n; ++s) {
                    activation_backward<<<blocks_for(elements), kThreads>>>(gradient[reverse] + s * elems,
                        preactivation[reverse] + s * elems, elements, layer.leaky);
                    launch_check();
                }
                if (layer.batch_normalize) {
#ifdef DETECTOR_HAS_CUDNN
                    cudnn_batch_norms[reverse]->backward(raw[reverse], gradient[reverse], scales[reverse],
                        batch_mean[reverse], sum_dxhat[reverse], preactivation[reverse],
                        scale_grad_accum[reverse], bias_grad_accum[reverse], static_cast<int>(n));
                    check(cudaMemcpy(gradient[reverse], preactivation[reverse], n * elems * sizeof(float),
                                     cudaMemcpyDeviceToDevice), "cudaMemcpy cuDNN BN gradient");
#else
                    int chunks = (plane + kReductionTile - 1) / kReductionTile;
                    dim3 grid(chunks, layer.output.c, static_cast<unsigned int>(n));
                    batch_norm_backward_partial<<<grid, kThreads>>>(gradient[reverse], raw[reverse],
                        batch_mean[reverse], batch_var[reverse], scales[reverse], bn_partial,
                        layer.output.c, plane, elements, chunks);
                    launch_check();
                    batch_norm_backward_finish<<<blocks_for(layer.output.c), kThreads>>>(bn_partial,
                        sum_dxhat[reverse], sum_dxhat_xhat[reverse], scale_grad_accum[reverse],
                        bias_grad_accum[reverse], layer.output.c, static_cast<int>(n), chunks);
                    launch_check();
                    int total_n = static_cast<int>(n) * plane;
                    for (std::size_t s = 0; s < n; ++s) {
                        batch_norm_backward_dz<<<blocks_for(elements), kThreads>>>(gradient[reverse] + s * elems, raw[reverse] + s * elems,
                            batch_mean[reverse], batch_var[reverse], scales[reverse], sum_dxhat[reverse], sum_dxhat_xhat[reverse],
                            elements, plane, total_n);
                        launch_check();
                    }
#endif
                } else {
                    for (std::size_t s = 0; s < n; ++s) {
                        bias_grad_reduce<<<blocks_for(layer.output.c), kThreads>>>(gradient[reverse] + s * elems, bias_grad_accum[reverse],
                            layer.output.c, plane);
                        launch_check();
                    }
                }
#ifdef DETECTOR_HAS_CUDNN
                cudnn_convolutions[reverse]->backward_filter(previous_of(0), gradient[reverse],
                    weight_grad_accum[reverse], static_cast<int>(n));
                if (reverse)
                    cudnn_convolutions[reverse]->backward_data(weights[reverse], gradient[reverse],
                        prev_grad_of(0), static_cast<int>(n));
#else
                for (std::size_t s = 0; s < n; ++s) {
                    int key_count = layer.input.c * layer.size * layer.size;
                    convolution_im2col<<<blocks_for(key_count * plane), kThreads>>>(previous_of(s), columns,
                        layer.input.c, layer.input.h, layer.input.w, layer.output.h, layer.output.w,
                        layer.size, layer.stride, layer.padding);
                    launch_check();
                    check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, key_count, layer.output.c, plane,
                                      &one, columns, plane, gradient[reverse] + s * elems, plane,
                                      &one, weight_grad_accum[reverse], key_count), "cublasSgemm weight gradient");
                    if (reverse) {
                        check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, plane, key_count, layer.output.c,
                                          &one, gradient[reverse] + s * elems, plane, weights[reverse], key_count,
                                          &zero_beta, column_gradient, plane), "cublasSgemm input gradient");
                        convolution_col2im_add<<<blocks_for(static_cast<int>(layer.input.elements())), kThreads>>>(
                            column_gradient, prev_grad_of(s), layer.input.c, layer.input.h, layer.input.w,
                            layer.output.h, layer.output.w, layer.size, layer.stride, layer.padding);
                        launch_check();
                    }
                }
#endif
            }
        }

        // One SGD-with-momentum update, averaged over this call's n samples (every call is a
        // full minibatch); velocity buffers persist across calls (see sgd_apply_momentum).
        float lr = schedule.at(iteration);
        float inv_n = 1.0f / static_cast<float>(n);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            if (layer.kind != Kind::convolutional) continue;
            int wcount = static_cast<int>(layer.weights.size());
            sgd_apply_with_decay_momentum<<<blocks_for(wcount), kThreads>>>(
                weights[i], weight_grad_accum[i], weight_velocity[i], wcount, lr, inv_n, weight_decay, momentum);
            launch_check();
            sgd_apply_momentum<<<blocks_for(layer.output.c), kThreads>>>(
                biases[i], bias_grad_accum[i], bias_velocity[i], layer.output.c, lr, inv_n, momentum);
            launch_check();
            zero<<<blocks_for(wcount), kThreads>>>(weight_grad_accum[i], wcount);
            launch_check();
            zero<<<blocks_for(layer.output.c), kThreads>>>(bias_grad_accum[i], layer.output.c);
            launch_check();
            if (layer.batch_normalize) {
                sgd_apply_momentum<<<blocks_for(layer.output.c), kThreads>>>(
                    scales[i], scale_grad_accum[i], scale_velocity[i], layer.output.c, lr, inv_n, momentum);
                launch_check();
                zero<<<blocks_for(layer.output.c), kThreads>>>(scale_grad_accum[i], layer.output.c);
                launch_check();
            }
        }
        ++iteration;
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize training");
        if (!std::isfinite(total_loss)) throw std::runtime_error("non-finite CUDA training loss");
        return {total_loss, total_positives};
    }

    float current_learning_rate() const { return schedule.at(iteration); }

    /** Lists persistent state in a stable order for cross-device averaging. */
    std::vector<std::pair<float*, std::size_t>> state_buffers() const {
        std::vector<std::pair<float*, std::size_t>> result;
        const auto& layers = network.layers();
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            if (layer.kind != Kind::convolutional) continue;
            result.emplace_back(weights[i], layer.weights.size());
            result.emplace_back(biases[i], layer.biases.size());
            result.emplace_back(weight_velocity[i], layer.weights.size());
            result.emplace_back(bias_velocity[i], layer.biases.size());
            if (layer.batch_normalize) {
                result.emplace_back(scales[i], layer.scales.size());
                result.emplace_back(scale_velocity[i], layer.scales.size());
                result.emplace_back(running_mean[i], layer.means.size());
                result.emplace_back(running_variance[i], layer.variances.size());
            }
        }
        return result;
    }

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

CudaTrainer::CudaTrainer(Network& network, LearningRateSchedule schedule, int batch_size, float weight_decay,
                           float momentum, float bn_momentum, int initial_iteration, int device_id) {
    if (device_id >= 0) check(cudaSetDevice(device_id), "cudaSetDevice training");
    // Must run before the first CUDA call creates the context. Default scheduling (cudaDeviceScheduleAuto)
    // picks a host-thread busy-wait spin loop on this machine (few host threads vs. many logical CPUs),
    // which pegs a full CPU core at ~100% for the whole multi-day training run for no throughput benefit -
    // pure wasted heat that, in a laptop chassis sharing cooling between CPU and GPU, measurably worsens the
    // GPU's own SW thermal-slowdown throttling over sustained runs (observed per-epoch time degrading from
    // ~46min to ~12.4h over the YOLOv2 full-budget retrain). Blocking sync parks the host thread instead.
    check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync), "cudaSetDeviceFlags");
    check(cudaGetDevice(&device_id_), "cudaGetDevice training");
    impl_ = std::make_unique<Impl>(network, std::move(schedule), batch_size, weight_decay, momentum, bn_momentum,
                                     initial_iteration);
}
CudaTrainer::~CudaTrainer() {
    if (impl_) {
        cudaSetDevice(device_id_);
        impl_.reset();
    }
}
TrainResult CudaTrainer::step(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<Target>>& targets) {
    check(cudaSetDevice(device_id_), "cudaSetDevice training step");
    return impl_->step(inputs, targets);
}
float CudaTrainer::current_learning_rate() const { return impl_->current_learning_rate(); }
int CudaTrainer::iteration() const { return impl_->iteration; }
void CudaTrainer::sync_to_network() {
    check(cudaSetDevice(device_id_), "cudaSetDevice training checkpoint");
    impl_->sync_to_network();
}
void CudaTrainer::resize(int width, int height) {
    check(cudaSetDevice(device_id_), "cudaSetDevice training resize");
    impl_->resize(width, height);
}

/** Reduces each persistent tensor on the host and broadcasts the weighted result to every GPU. */
void CudaTrainer::average_state(const std::vector<CudaTrainer*>& trainers,
                                 const std::vector<std::size_t>& sample_counts) {
    if (trainers.empty() || trainers.size() != sample_counts.size())
        throw std::runtime_error("invalid CUDA trainer state average");
    std::size_t total = 0;
    int next_iteration = 0;
    std::vector<std::vector<std::pair<float*, std::size_t>>> buffers;
    buffers.reserve(trainers.size());
    for (std::size_t i = 0; i < trainers.size(); ++i) {
        if (!trainers[i]) throw std::runtime_error("null CUDA trainer in state average");
        total += sample_counts[i];
        next_iteration = std::max(next_iteration, trainers[i]->impl_->iteration);
        buffers.push_back(trainers[i]->impl_->state_buffers());
    }
    if (total == 0) throw std::runtime_error("empty CUDA trainer state average");
    for (std::size_t i = 0; i < trainers.size(); ++i) {
        int expected = next_iteration - (sample_counts[i] == 0 ? 1 : 0);
        if (trainers[i]->impl_->iteration != expected || buffers[i].size() != buffers[0].size())
            throw std::runtime_error("CUDA trainers have incompatible state or step counts");
    }
    for (std::size_t tensor = 0; tensor < buffers[0].size(); ++tensor) {
        const std::size_t count = buffers[0][tensor].second;
        std::vector<double> sum(count, 0.0);
        std::vector<float> values(count);
        for (std::size_t i = 0; i < trainers.size(); ++i) {
            if (buffers[i][tensor].second != count)
                throw std::runtime_error("CUDA trainers have different tensor sizes");
            if (sample_counts[i] == 0) continue;
            check(cudaSetDevice(trainers[i]->device_id_), "cudaSetDevice state gather");
            check(cudaMemcpy(values.data(), buffers[i][tensor].first, count * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy state gather");
            const double weight = static_cast<double>(sample_counts[i]) / static_cast<double>(total);
            for (std::size_t j = 0; j < count; ++j) sum[j] += weight * values[j];
        }
        for (std::size_t j = 0; j < count; ++j) values[j] = static_cast<float>(sum[j]);
        for (std::size_t i = 0; i < trainers.size(); ++i) {
            check(cudaSetDevice(trainers[i]->device_id_), "cudaSetDevice state broadcast");
            check(cudaMemcpy(buffers[i][tensor].first, values.data(), count * sizeof(float), cudaMemcpyHostToDevice),
                  "cudaMemcpy state broadcast");
        }
    }
    for (auto* trainer : trainers) trainer->impl_->iteration = next_iteration;
}

/** Builds one independently allocated model replica and trainer per distinct selected GPU. */
MultiCudaTrainer::MultiCudaTrainer(Network& network, LearningRateSchedule schedule, int batch_size,
                                   std::vector<int> device_ids, float weight_decay, float momentum,
                                   float bn_momentum, int initial_iteration)
    : network_(network), batch_size_(batch_size) {
    if (device_ids.size() < 2 || batch_size < static_cast<int>(device_ids.size()))
        throw std::runtime_error("multi-GPU training requires at least two devices and batch >= GPU count");
    int available = 0;
    check(cudaGetDeviceCount(&available), "cudaGetDeviceCount multi-GPU");
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        if (device_ids[i] < 0 || device_ids[i] >= available ||
            std::find(device_ids.begin(), device_ids.begin() + i, device_ids[i]) != device_ids.begin() + i)
            throw std::runtime_error("--gpus must list distinct available CUDA device indices");
    }
    replicas_.reserve(device_ids.size() - 1);
    for (std::size_t i = 1; i < device_ids.size(); ++i) replicas_.push_back(network);
    trainers_.reserve(device_ids.size());
    for (std::size_t i = 0; i < device_ids.size(); ++i)
        trainers_.push_back(std::make_unique<CudaTrainer>(i == 0 ? network_ : replicas_[i - 1], schedule,
            (batch_size + static_cast<int>(device_ids.size()) - 1) / static_cast<int>(device_ids.size()),
            weight_decay, momentum, bn_momentum, initial_iteration, device_ids[i]));
}

/** Runs balanced shards concurrently, then averages by the actual sample count of each shard. */
TrainResult MultiCudaTrainer::step(const std::vector<std::vector<float>>& inputs,
                                   const std::vector<std::vector<Target>>& targets) {
    if (inputs.empty() || inputs.size() > static_cast<std::size_t>(batch_size_) || inputs.size() != targets.size())
        throw std::runtime_error("invalid multi-GPU detector minibatch");
    const std::size_t workers = trainers_.size();
    std::vector<std::size_t> counts(workers, 0);
    std::vector<std::future<TrainResult>> jobs(workers);
    for (std::size_t i = 0; i < workers; ++i) {
        const std::size_t begin = inputs.size() * i / workers;
        const std::size_t end = inputs.size() * (i + 1) / workers;
        counts[i] = end - begin;
        if (!counts[i]) continue;
        jobs[i] = std::async(std::launch::async, [&, i, begin, end] {
            std::vector<std::vector<float>> local_inputs(inputs.begin() + begin, inputs.begin() + end);
            std::vector<std::vector<Target>> local_targets(targets.begin() + begin, targets.begin() + end);
            return trainers_[i]->step(local_inputs, local_targets);
        });
    }
    TrainResult result{};
    for (std::size_t i = 0; i < workers; ++i) {
        if (!counts[i]) continue;
        auto local = jobs[i].get();
        result.loss += local.loss;
        result.positives += local.positives;
    }
    std::vector<CudaTrainer*> raw;
    for (auto& trainer : trainers_) raw.push_back(trainer.get());
    CudaTrainer::average_state(raw, counts);
    return result;
}

float MultiCudaTrainer::current_learning_rate() const { return trainers_[0]->current_learning_rate(); }
int MultiCudaTrainer::iteration() const { return trainers_[0]->iteration(); }
void MultiCudaTrainer::sync_to_network() { trainers_[0]->sync_to_network(); }
void MultiCudaTrainer::resize(int width, int height) {
    for (auto& trainer : trainers_) trainer->resize(width, height);
}

}  // namespace detector
#endif
