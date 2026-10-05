#pragma once

#include "detector/network.hpp"
#include <memory>
#include <string>
#include <vector>

namespace detector {

// A Jetson-built TensorRT plan with the same two raw YOLO head outputs as the
// ONNX exporter. Preprocessing and decode/NMS remain in detector_core.
class TrtSession {
public:
    TrtSession(const Network& network, const std::string& engine_path);
    ~TrtSession();
    TrtSession(const TrtSession&) = delete;
    TrtSession& operator=(const TrtSession&) = delete;
    std::vector<Head> forward(const std::vector<float>& input);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace detector
