#include "detector/idx.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>

namespace detector {
namespace {

std::uint32_t read_be32(std::ifstream& file) {
    unsigned char bytes[4]{};
    if (!file.read(reinterpret_cast<char*>(bytes), 4)) throw std::runtime_error("truncated IDX header");
    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) | bytes[3];
}

std::vector<std::uint8_t> read_bytes(std::ifstream& file, std::size_t count) {
    std::vector<std::uint8_t> result(count);
    if (!file.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(count)))
        throw std::runtime_error("truncated IDX payload");
    char trailing;
    if (file.read(&trailing, 1)) throw std::runtime_error("IDX payload has trailing bytes");
    if (!file.eof()) throw std::runtime_error("IDX read error");
    return result;
}

}  // namespace

IdxDataset::IdxDataset(const std::string& images_path, const std::string& labels_path) {
    std::ifstream images(images_path, std::ios::binary);
    std::ifstream labels(labels_path, std::ios::binary);
    if (!images || !labels) throw std::runtime_error("cannot open IDX image/label files");
    if (read_be32(images) != 2051 || read_be32(labels) != 2049)
        throw std::runtime_error("invalid IDX image/label magic");
    std::uint32_t count = read_be32(images);
    std::uint32_t label_count = read_be32(labels);
    std::uint32_t height = read_be32(images);
    std::uint32_t width = read_be32(images);
    if (!count || count != label_count || !height || !width ||
        height > 8192 || width > 8192 ||
        static_cast<std::uint64_t>(count) * height * width > 200000000ULL)
        throw std::runtime_error("invalid or unsupported IDX dimensions");
    height_ = static_cast<int>(height);
    width_ = static_cast<int>(width);
    pixels_ = read_bytes(images, static_cast<std::size_t>(count) * height * width);
    labels_ = read_bytes(labels, label_count);
}

int IdxDataset::label(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("IDX sample index");
    return labels_[index];
}

std::vector<float> IdxDataset::image(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("IDX sample index");
    const std::size_t area = static_cast<std::size_t>(height_) * width_;
    const std::size_t start = index * area;
    std::vector<float> result(area);
    for (std::size_t i = 0; i < area; ++i) result[i] = pixels_[start + i] / 255.0f;
    return result;
}

}  // namespace detector
