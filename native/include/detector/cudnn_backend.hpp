#pragma once

// Minimal cuDNN link/runtime smoke surface. Built only when DETECTOR_CUDNN=ON.
// The actual cuDNN-accelerated convolution/batch-norm kernels (replacing the
// cuBLAS im2col/col2im path in cuda_training.cu/cuda_backend.cu) are a separate,
// larger follow-on item - see DESIGN.yaml todo - this header only proves the
// pip-installed cuDNN wheel links and loads correctly in this build.

namespace detector {

// Returns cuDNN's runtime version (CUDNN_VERSION-style integer, e.g. 90400) by
// creating and immediately destroying a handle, or throws std::runtime_error if
// cuDNN failed to initialize (e.g. its DLLs are not on PATH at run time).
int cudnn_runtime_version();

}  // namespace detector
