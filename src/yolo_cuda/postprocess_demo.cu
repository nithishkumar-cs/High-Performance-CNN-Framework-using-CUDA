#include "yolo_cuda/postprocess.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t rows = 3;
    constexpr std::int32_t classes = 2;
    constexpr std::size_t stride = 5 + classes;
    const float host_output[rows * stride] = {
        0.0f, 0.0f, -0.2f, -0.2f, 5.0f, 5.0f, -4.0f,
        0.02f, 0.01f, -0.2f, -0.2f, 4.0f, 4.0f, -4.0f,
        -0.5f, -0.5f, 0.0f, 0.0f, -3.0f, 4.0f, -4.0f,
    };

    float* device_output = nullptr;
    cudaMalloc(&device_output, sizeof(host_output));
    cudaMemcpy(device_output, host_output, sizeof(host_output), cudaMemcpyHostToDevice);

    yolo_cuda::PostprocessConfig config;
    config.class_count = classes;
    config.image_width = 640;
    config.image_height = 640;
    const auto detections = yolo_cuda::decode_and_nms(device_output, rows, config);
    cudaFree(device_output);

    std::cout << "detections=" << detections.size() << '\n';
    for (const auto& detection : detections) {
        std::cout << detection.class_id << ' ' << detection.confidence << ' '
                  << detection.x1 << ' ' << detection.y1 << ' '
                  << detection.x2 << ' ' << detection.y2 << '\n';
    }
    return 0;
}
