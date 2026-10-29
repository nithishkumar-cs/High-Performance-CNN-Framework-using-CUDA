#include "detector/vector_dataset.hpp"
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main() {
    fs::path path = fs::temp_directory_path() /
        ("cuda-framework-vecdata-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".vecbin");
    try {
        // Two records: input_size=2, target_size=3.
        const std::vector<float> record0 = {0.5f, -1.0f, 1.0f, 0.0f, 1.0f};
        const std::vector<float> record1 = {0.25f, 0.75f, 0.0f, 1.0f, 0.0f};
        {
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(record0.data()),
                       static_cast<std::streamsize>(record0.size() * sizeof(float)));
            file.write(reinterpret_cast<const char*>(record1.data()),
                       static_cast<std::streamsize>(record1.size() * sizeof(float)));
        }
        detector::VectorDataset data(path.string(), 2, 3);
        if (data.size() != 2) throw std::runtime_error("vector dataset size mismatch");
        auto input0 = data.input(0);
        auto target0 = data.target(0);
        auto input1 = data.input(1);
        auto target1 = data.target(1);
        if (input0 != std::vector<float>{0.5f, -1.0f} || target0 != std::vector<float>{1.0f, 0.0f, 1.0f} ||
            input1 != std::vector<float>{0.25f, 0.75f} || target1 != std::vector<float>{0.0f, 1.0f, 0.0f})
            throw std::runtime_error("vector dataset record decode mismatch");

        bool rejected = false;
        try { detector::VectorDataset bad(path.string(), 3, 3); }  // 5 floats isn't 2 records of 6.
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("vector dataset misaligned record size accepted");

        fs::remove(path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        fs::remove(path);
        return 1;
    }
}
