// Minimal link/runtime smoke test for the DETECTOR_CUDNN build: proves the
// pip-installed cuDNN wheel's headers/import-lib/DLLs actually link and load in
// this build (not yet a correctness test of any cuDNN-accelerated kernel - none
// exist yet, see cudnn_backend.hpp).
#include "detector/cudnn_backend.hpp"

#include <cassert>
#include <cstdio>

int main() {
    const int version = detector::cudnn_runtime_version();
    assert(version > 0);
    std::printf("cudnn_smoke: cuDNN runtime version %d\n", version);
    return 0;
}
