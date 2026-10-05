#include "detector/image.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#ifdef DETECTOR_HAS_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace detector {
namespace {

std::string ppm_token(std::istream& file) {
    std::string token;
    for (;;) {
        file >> std::ws;
        if (file.peek() != '#') break;
        std::string ignored;
        std::getline(file, ignored);
    }
    if (!(file >> token)) throw std::runtime_error("truncated PPM header");
    return token;
}

}  // namespace

Image read_image(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open image: " + path);
    char magic[2]{};
    file.read(magic, 2);
    if (file && magic[0] == 'P' && (magic[1] == '6' || magic[1] == '3')) {
        Image image;
        try {
            image.width = std::stoi(ppm_token(file));
            image.height = std::stoi(ppm_token(file));
            int maximum = std::stoi(ppm_token(file));
            if (image.width <= 0 || image.height <= 0 || maximum != 255)
                throw std::runtime_error("PPM requires positive dimensions and maxval 255");
            const std::size_t bytes = static_cast<std::size_t>(image.width) * image.height * 3;
            if (bytes > 500000000) throw std::runtime_error("PPM image too large");
            image.rgb.resize(bytes);
            if (magic[1] == '6') {
                char delimiter = 0;
                if (!file.get(delimiter) || !std::isspace(static_cast<unsigned char>(delimiter)))
                    throw std::runtime_error("invalid PPM pixel delimiter");
                if (delimiter == '\r' && file.peek() == '\n') file.get();
                if (!file.read(reinterpret_cast<char*>(image.rgb.data()), static_cast<std::streamsize>(bytes)))
                    throw std::runtime_error("truncated PPM pixels");
            } else {
                for (auto& channel : image.rgb) {
                    int value = std::stoi(ppm_token(file));
                    if (value < 0 || value > 255) throw std::runtime_error("invalid PPM channel");
                    channel = static_cast<std::uint8_t>(value);
                }
            }
            return image;
        } catch (const std::invalid_argument&) { throw std::runtime_error("invalid PPM header"); }
          catch (const std::out_of_range&) { throw std::runtime_error("invalid PPM dimension"); }
    }
#ifdef DETECTOR_HAS_OPENCV
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("cannot decode image: " + path);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return {rgb.cols, rgb.rows, std::vector<std::uint8_t>(rgb.data, rgb.data + rgb.total() * 3)};
#else
    throw std::runtime_error("only PPM images are supported; build with -DDETECTOR_OPENCV=ON for JPEG/PNG");
#endif
}

Letterbox letterbox(const Image& image, Shape target) {
    if (image.width <= 0 || image.height <= 0 || image.rgb.size() != static_cast<std::size_t>(image.width) * image.height * 3)
        throw std::runtime_error("invalid RGB image");
    if (target.c != 3 || target.h <= 0 || target.w <= 0) throw std::runtime_error("letterbox requires a three-channel network");
    float scale = std::min(static_cast<float>(target.w) / image.width, static_cast<float>(target.h) / image.height);
    Letterbox result;
    result.original_width = image.width;
    result.original_height = image.height;
    result.resized_width = std::max(1, std::min(target.w, static_cast<int>(std::round(image.width * scale))));
    result.resized_height = std::max(1, std::min(target.h, static_cast<int>(std::round(image.height * scale))));
    result.left = (target.w - result.resized_width) / 2;
    result.top = (target.h - result.resized_height) / 2;
    result.chw.assign(target.elements(), 0.5f);
    for (int y = 0; y < result.resized_height; ++y) {
        float source_y = (y + 0.5f) * image.height / result.resized_height - 0.5f;
        int y0 = std::clamp(static_cast<int>(std::floor(source_y)), 0, image.height - 1);
        int y1 = std::min(y0 + 1, image.height - 1);
        float fy = std::clamp(source_y - std::floor(source_y), 0.0f, 1.0f);
        for (int x = 0; x < result.resized_width; ++x) {
            float source_x = (x + 0.5f) * image.width / result.resized_width - 0.5f;
            int x0 = std::clamp(static_cast<int>(std::floor(source_x)), 0, image.width - 1);
            int x1 = std::min(x0 + 1, image.width - 1);
            float fx = std::clamp(source_x - std::floor(source_x), 0.0f, 1.0f);
            for (int c = 0; c < 3; ++c) {
                auto sample = [&](int yy, int xx) { return image.rgb[(static_cast<std::size_t>(yy) * image.width + xx) * 3 + c] / 255.0f; };
                float top = sample(y0, x0) * (1 - fx) + sample(y0, x1) * fx;
                float bottom = sample(y1, x0) * (1 - fx) + sample(y1, x1) * fx;
                result.chw[(static_cast<std::size_t>(c) * target.h + y + result.top) * target.w + x + result.left] =
                    top * (1 - fy) + bottom * fy;
            }
        }
    }
    return result;
}

void undo_letterbox(std::vector<Detection>& boxes, const Letterbox& transform) {
    if (transform.resized_width <= 0 || transform.resized_height <= 0) throw std::runtime_error("invalid letterbox transform");
    auto unmap_x = [&](float x) { return std::clamp((x - transform.left) * transform.original_width / transform.resized_width,
                                                     0.0f, static_cast<float>(transform.original_width)); };
    auto unmap_y = [&](float y) { return std::clamp((y - transform.top) * transform.original_height / transform.resized_height,
                                                     0.0f, static_cast<float>(transform.original_height)); };
    for (auto& box : boxes) {
        box.x1 = unmap_x(box.x1); box.x2 = unmap_x(box.x2);
        box.y1 = unmap_y(box.y1); box.y2 = unmap_y(box.y2);
    }
}

}  // namespace detector
