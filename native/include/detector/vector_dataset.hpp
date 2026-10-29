#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace detector {

// This project's own fixed-record binary dataset format for VectorTrainer's regressor/tag
// example apps (there is no upstream Darknet-compatible format to match here: regressor.c/
// tag.c read a JPEG directory plus per-image text label files via OpenCV, which this build
// does not require). Each record is `input_size + target_size` little-endian float32 values,
// concatenated with no header - the input vector (already normalized, e.g. to [0,1] for
// pixels) then the target vector (regression values, or 0/1 multi-label tags).
class VectorDataset {
public:
    VectorDataset(const std::string& path, std::size_t input_size, std::size_t target_size);
    std::size_t size() const { return count_; }
    std::vector<float> input(std::size_t index) const;
    std::vector<float> target(std::size_t index) const;

private:
    std::size_t input_size_;
    std::size_t target_size_;
    std::size_t count_ = 0;
    std::vector<float> data_;  // count_ * (input_size_ + target_size_) values.
};

}  // namespace detector
