#include "detector/vector_dataset.hpp"
#include <fstream>
#include <stdexcept>

namespace detector {

VectorDataset::VectorDataset(const std::string& path, std::size_t input_size, std::size_t target_size)
    : input_size_(input_size), target_size_(target_size) {
    if (!input_size || !target_size) throw std::runtime_error("vector dataset sizes must be positive");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open vector dataset file: " + path);
    auto byte_count = file.tellg();
    const std::size_t record_floats = input_size_ + target_size_;
    const std::size_t record_bytes = record_floats * sizeof(float);
    if (byte_count < 0 || static_cast<std::size_t>(byte_count) % record_bytes != 0)
        throw std::runtime_error("vector dataset file is not a whole number of records: " + path);
    count_ = static_cast<std::size_t>(byte_count) / record_bytes;
    if (!count_) throw std::runtime_error("vector dataset file contained no records: " + path);
    data_.resize(count_ * record_floats);
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(data_.data()),
                    static_cast<std::streamsize>(data_.size() * sizeof(float))))
        throw std::runtime_error("vector dataset read error: " + path);
}

std::vector<float> VectorDataset::input(std::size_t index) const {
    if (index >= count_) throw std::out_of_range("vector dataset sample index");
    const std::size_t start = index * (input_size_ + target_size_);
    return {data_.begin() + static_cast<std::ptrdiff_t>(start),
            data_.begin() + static_cast<std::ptrdiff_t>(start + input_size_)};
}

std::vector<float> VectorDataset::target(std::size_t index) const {
    if (index >= count_) throw std::out_of_range("vector dataset sample index");
    const std::size_t start = index * (input_size_ + target_size_) + input_size_;
    return {data_.begin() + static_cast<std::ptrdiff_t>(start),
            data_.begin() + static_cast<std::ptrdiff_t>(start + target_size_)};
}

}  // namespace detector
