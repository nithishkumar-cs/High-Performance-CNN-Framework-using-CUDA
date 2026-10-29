#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace detector {

// CIFAR-10's own binary batch format (distinct from IDX): each of the dataset's fixed-size
// 32x32 RGB images is one 3073-byte record (1 label byte in [0,9], then 1024 R bytes, 1024 G
// bytes, 1024 B bytes, each channel plane row-major) with no file header, unlike IDX's
// big-endian magic/dimension header. Darknet's own cifar.c example trains against this same
// upstream format (cifar-10-binary.tar.gz's data_batch_N.bin/test_batch.bin). Concatenates
// every file given to the constructor (e.g. all 5 train batches) into one dataset.
class CifarDataset {
public:
    explicit CifarDataset(const std::vector<std::string>& batch_paths);
    std::size_t size() const { return labels_.size(); }
    static constexpr int channels() { return 3; }
    static constexpr int height() { return 32; }
    static constexpr int width() { return 32; }
    int label(std::size_t index) const;
    // CHW, channel-major (R plane, then G, then B), normalized to [0,1] - matches this
    // project's other image tensors (network.cpp's letterbox/CPU-inference convention).
    std::vector<float> image(std::size_t index) const;

private:
    std::vector<std::uint8_t> pixels_;  // size() * 3072 bytes, CHW per record, concatenated.
    std::vector<std::uint8_t> labels_;
};

}  // namespace detector
