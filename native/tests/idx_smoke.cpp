#include "detector/idx.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

void write_be32(std::ofstream& file, std::uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>(value >> 24),
        static_cast<unsigned char>(value >> 16),
        static_cast<unsigned char>(value >> 8),
        static_cast<unsigned char>(value)
    };
    file.write(reinterpret_cast<const char*>(bytes), 4);
}

int main() {
    fs::path directory = fs::temp_directory_path() /
        ("cuda-framework-idx-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directory(directory);
        auto image_path = directory / "images.idx3-ubyte";
        auto label_path = directory / "labels.idx1-ubyte";
        {
            std::ofstream image(image_path, std::ios::binary);
            write_be32(image, 2051); write_be32(image, 2);
            write_be32(image, 4); write_be32(image, 4);
            for (int i = 0; i < 16; ++i) image.put(0);
            for (int i = 0; i < 16; ++i) image.put(static_cast<char>(255));
        }
        {
            std::ofstream label(label_path, std::ios::binary);
            write_be32(label, 2049); write_be32(label, 2);
            label.put(0); label.put(1);
        }
        detector::IdxDataset data(image_path.string(), label_path.string());
        if (data.size() != 2 || data.height() != 4 || data.width() != 4 ||
            data.label(0) != 0 || data.label(1) != 1 ||
            data.image(0).size() != 16 || data.image(0)[0] != 0.0f ||
            data.image(1)[15] != 1.0f)
            throw std::runtime_error("IDX decode mismatch");
        {
            std::ofstream label(label_path, std::ios::binary | std::ios::app);
            label.put(2);
        }
        bool rejected = false;
        try { detector::IdxDataset bad(image_path.string(), label_path.string()); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("IDX trailing data accepted");
        fs::remove(image_path);
        fs::remove(label_path);
        fs::remove(directory);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (fs::exists(directory / "images.idx3-ubyte")) fs::remove(directory / "images.idx3-ubyte");
        if (fs::exists(directory / "labels.idx1-ubyte")) fs::remove(directory / "labels.idx1-ubyte");
        if (fs::exists(directory)) fs::remove(directory);
        return 1;
    }
}
