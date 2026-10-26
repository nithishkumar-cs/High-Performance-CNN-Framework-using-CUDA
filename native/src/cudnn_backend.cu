#include "detector/cudnn_backend.hpp"

#include <cudnn.h>

#include <stdexcept>
#include <string>

namespace detector {
namespace {

void check(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(status));
    }
}

}  // namespace

int cudnn_runtime_version() {
    cudnnHandle_t handle = nullptr;
    check(cudnnCreate(&handle), "cudnnCreate");
    int version = static_cast<int>(cudnnGetVersion());
    check(cudnnDestroy(handle), "cudnnDestroy");
    return version;
}

}  // namespace detector
