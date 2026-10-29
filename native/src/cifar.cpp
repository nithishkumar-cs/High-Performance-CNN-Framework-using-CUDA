#include "detector/cifar.hpp"
#include <fstream>
#include <stdexcept>

namespace detector {
namespace {

constexpr std::size_t kRecordBytes = 1 + 3 * 32 * 32;  // label + R/G/B planes.

}  // namespace

CifarDataset::CifarDataset(const std::vector<std::string>& batch_paths) {
    if (batch_paths.empty()) throw std::runtime_error("CIFAR dataset needs at least one batch file");
    for (const auto& path : batch_paths) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("cannot open CIFAR batch file: " + path);
        auto byte_count = file.tellg();
        if (byte_count < 0 || static_cast<std::size_t>(byte_count) % kRecordBytes != 0)
            throw std::runtime_error("CIFAR batch file is not a whole number of 3073-byte records: " + path);
        const std::size_t records = static_cast<std::size_t>(byte_count) / kRecordBytes;
        file.seekg(0, std::ios::beg);
        std::vector<char> record(kRecordBytes);
        for (std::size_t i = 0; i < records; ++i) {
            if (!file.read(record.data(), static_cast<std::streamsize>(kRecordBytes)))
                throw std::runtime_error("CIFAR batch file read error: " + path);
            labels_.push_back(static_cast<std::uint8_t>(record[0]));
            pixels_.insert(pixels_.end(), record.begin() + 1, record.end());
        }
    }
    if (labels_.empty()) throw std::runtime_error("CIFAR batch files contained no records");
}

int CifarDataset::label(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("CIFAR sample index");
    return labels_[index];
}

std::vector<float> CifarDataset::image(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("CIFAR sample index");
    constexpr std::size_t area = 3 * 32 * 32;
    const std::size_t start = index * area;
    std::vector<float> result(area);
    for (std::size_t i = 0; i < area; ++i) result[i] = pixels_[start + i] / 255.0f;
    return result;
}

}  // namespace detector
