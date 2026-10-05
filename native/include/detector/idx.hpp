#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace detector {

// One grayscale image per sample in standard IDX image/label files (e.g. MNIST).
class IdxDataset {
public:
    IdxDataset(const std::string& images_path, const std::string& labels_path);
    std::size_t size() const { return labels_.size(); }
    int height() const { return height_; }
    int width() const { return width_; }
    int label(std::size_t index) const;
    std::vector<float> image(std::size_t index) const;

private:
    int height_ = 0;
    int width_ = 0;
    std::vector<std::uint8_t> pixels_;
    std::vector<std::uint8_t> labels_;
};

}  // namespace detector
