#include "yolo_cuda/postprocess.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <cuda_runtime.h>

namespace yolo_cuda {
namespace {

struct Candidate {
    float x1, y1, x2, y2, confidence;
    std::int32_t class_id;
};

__device__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__global__ void decode_kernel(
    const float* output,
    Candidate* candidates,
    std::size_t row_count,
    std::int32_t class_count,
    float confidence_threshold,
    float image_width,
    float image_height) {
    const auto row = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= row_count) return;

    const std::size_t stride = static_cast<std::size_t>(5 + class_count);
    const float* values = output + row * stride;
    const float objectness = sigmoid(values[4]);

    float best_score = 0.0f;
    std::int32_t best_class = -1;
    for (std::int32_t c = 0; c < class_count; ++c) {
        const float score = objectness * sigmoid(values[5 + c]);
        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }

    Candidate candidate{};
    if (best_score >= confidence_threshold) {
        const float cx = sigmoid(values[0]) * image_width;
        const float cy = sigmoid(values[1]) * image_height;
        const float width = expf(values[2]) * image_width;
        const float height = expf(values[3]) * image_height;
        candidate = {cx - width * 0.5f, cy - height * 0.5f,
                     cx + width * 0.5f, cy + height * 0.5f,
                     best_score, best_class};
    } else {
        candidate.class_id = -1;
    }
    candidates[row] = candidate;
}

float intersection_over_union(const Detection& a, const Detection& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    return intersection / std::max(area_a + area_b - intersection, 1.0e-12f);
}

}  // namespace

std::vector<Detection> decode_and_nms(
    const float* device_output,
    std::size_t row_count,
    const PostprocessConfig& config) {
    if (device_output == nullptr || row_count == 0 || config.class_count <= 0) {
        return {};
    }

    Candidate* device_candidates = nullptr;
    std::vector<Candidate> host_candidates(row_count);
    cudaError_t status = cudaMalloc(&device_candidates, row_count * sizeof(Candidate));
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));

    const int threads = 256;
    const int blocks = static_cast<int>((row_count + threads - 1) / threads);
    decode_kernel<<<blocks, threads>>>(device_output, device_candidates, row_count,
                                       config.class_count, config.confidence_threshold,
                                       static_cast<float>(config.image_width),
                                       static_cast<float>(config.image_height));
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_candidates.data(), device_candidates,
                            row_count * sizeof(Candidate), cudaMemcpyDeviceToHost);
    }
    cudaFree(device_candidates);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));

    std::vector<Detection> candidates;
    for (const Candidate& item : host_candidates) {
        if (item.class_id >= 0) {
            candidates.push_back({item.x1, item.y1, item.x2, item.y2,
                                  item.confidence, item.class_id});
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<Detection> kept;
    for (const Detection& candidate : candidates) {
        bool suppressed = false;
        for (const Detection& selected : kept) {
            if (candidate.class_id == selected.class_id &&
                intersection_over_union(candidate, selected) > config.nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(candidate);
    }
    return kept;
}

}  // namespace yolo_cuda
