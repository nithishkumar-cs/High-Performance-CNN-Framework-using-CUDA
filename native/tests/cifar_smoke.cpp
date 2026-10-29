#include "detector/cifar.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Writes one CIFAR-10 binary record: 1 label byte then 1024 R, 1024 G, 1024 B bytes.
void write_record(std::ofstream& file, int label, unsigned char r, unsigned char g, unsigned char b) {
    file.put(static_cast<char>(label));
    for (int i = 0; i < 1024; ++i) file.put(static_cast<char>(r));
    for (int i = 0; i < 1024; ++i) file.put(static_cast<char>(g));
    for (int i = 0; i < 1024; ++i) file.put(static_cast<char>(b));
}

}  // namespace

int main() {
    fs::path directory = fs::temp_directory_path() /
        ("cuda-framework-cifar-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directory(directory);
        auto batch_a = directory / "batch_a.bin";
        auto batch_b = directory / "batch_b.bin";
        {
            std::ofstream file(batch_a, std::ios::binary);
            write_record(file, 0, 0, 0, 0);
            write_record(file, 3, 255, 128, 0);
        }
        {
            std::ofstream file(batch_b, std::ios::binary);
            write_record(file, 7, 0, 0, 255);
        }

        // Single file: 2 records, correct label/channel-plane decode.
        detector::CifarDataset single({batch_a.string()});
        if (single.size() != 2 || single.channels() != 3 || single.height() != 32 || single.width() != 32)
            throw std::runtime_error("CIFAR shape mismatch");
        if (single.label(0) != 0 || single.label(1) != 3)
            throw std::runtime_error("CIFAR label mismatch");
        auto image1 = single.image(1);
        if (image1.size() != 3072) throw std::runtime_error("CIFAR image size mismatch");
        // Record 1 is R=255,G=128,B=0: R plane [0,1024), G plane [1024,2048), B plane [2048,3072).
        if (image1[0] != 1.0f || image1[1024] != 128.0f / 255.0f || image1[2048] != 0.0f)
            throw std::runtime_error("CIFAR channel-plane decode mismatch");

        // Concatenating two files appends the second file's records after the first's.
        detector::CifarDataset concatenated({batch_a.string(), batch_b.string()});
        if (concatenated.size() != 3 || concatenated.label(2) != 7)
            throw std::runtime_error("CIFAR multi-file concatenation mismatch");

        // A file whose size is not a whole number of 3073-byte records is rejected.
        auto truncated = directory / "truncated.bin";
        {
            std::ofstream file(truncated, std::ios::binary);
            write_record(file, 0, 0, 0, 0);
            file.seekp(-1, std::ios::end);  // drop the last byte.
        }
        fs::resize_file(truncated, 3072);
        bool rejected = false;
        try { detector::CifarDataset bad({truncated.string()}); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("CIFAR truncated batch file accepted");

        fs::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        fs::remove_all(directory);
        return 1;
    }
}
