#include "detector/network.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace detector {
namespace {

struct Section { std::string name; std::map<std::string, std::string> values; int line = 0; };

std::string trim(const std::string& s) {
    auto first = s.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? std::string{} : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

std::vector<Section> sections_from(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open cfg: " + path);
    std::vector<Section> sections;
    std::string line;
    int number = 0;
    while (std::getline(file, line)) {
        ++number;
        auto comment = line.find_first_of("#;");
        if (comment != std::string::npos) line.resize(comment);
        line = trim(line);
        if (line.empty()) continue;
        if (line.front() == '[') {
            if (line.back() != ']') throw std::runtime_error("bad cfg section at line " + std::to_string(number));
            sections.push_back({trim(line.substr(1, line.size() - 2)), {}, number});
        } else {
            auto equal = line.find('=');
            if (sections.empty() || equal == std::string::npos) throw std::runtime_error("bad cfg option at line " + std::to_string(number));
            auto key = trim(line.substr(0, equal));
            auto value = trim(line.substr(equal + 1));
            if (key.empty() || value.empty()) throw std::runtime_error("empty cfg option at line " + std::to_string(number));
            sections.back().values[key] = value;
        }
    }
    return sections;
}

std::string option(const Section& section, const std::string& key, const std::string& fallback = {}) {
    auto it = section.values.find(key);
    if (it != section.values.end()) return it->second;
    if (!fallback.empty()) return fallback;
    throw std::runtime_error("missing " + key + " in [" + section.name + "] at line " + std::to_string(section.line));
}

int integer(const Section& section, const std::string& key, const std::string& fallback = {}) {
    auto s = option(section, key, fallback);
    std::size_t used = 0;
    int value = 0;
    try { value = std::stoi(s, &used); }
    catch (const std::exception&) { throw std::runtime_error("invalid integer " + key); }
    if (used != s.size()) throw std::runtime_error("invalid integer " + key);
    return value;
}

float floating(const Section& section, const std::string& key, const std::string& fallback = {}) {
    auto s = option(section, key, fallback);
    std::size_t used = 0;
    float value = 0;
    try { value = std::stof(s, &used); }
    catch (const std::exception&) { throw std::runtime_error("invalid float " + key); }
    if (used != s.size()) throw std::runtime_error("invalid float " + key);
    return value;
}

std::vector<int> list(const Section& section, const std::string& key) {
    std::vector<int> result;
    std::istringstream stream(option(section, key));
    std::string field;
    while (std::getline(stream, field, ',')) {
        field = trim(field);
        std::size_t used = 0;
        int value = 0;
        try { value = std::stoi(field, &used); }
        catch (const std::exception&) { throw std::runtime_error("invalid list " + key); }
        if (used != field.size()) throw std::runtime_error("invalid list " + key);
        result.push_back(value);
    }
    if (result.empty()) throw std::runtime_error("empty list " + key);
    return result;
}

void positive(int x, const char* name) {
    if (x <= 0 || x > 8192) throw std::runtime_error(std::string("invalid ") + name);
}

template <typename T> void read_scalar(std::ifstream& file, T& value) {
    if (!file.read(reinterpret_cast<char*>(&value), sizeof(T))) throw std::runtime_error("truncated Darknet weights header");
}

void read_floats(std::ifstream& file, std::vector<float>& target, std::size_t count, std::size_t layer) {
    target.resize(count);
    if (!file.read(reinterpret_cast<char*>(target.data()), static_cast<std::streamsize>(count * sizeof(float))))
        throw std::runtime_error("truncated Darknet weights at layer " + std::to_string(layer));
}

float sigmoid(float x) {
    if (x >= 0) return 1.0f / (1.0f + std::exp(-x));
    float e = std::exp(x);
    return e / (1.0f + e);
}

float iou(const Detection& a, const Detection& b) {
    float w = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    float h = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    float intersection = w * h;
    float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    float area = area_a + area_b - intersection;
    return area > 0 ? intersection / area : 0.0f;
}

std::vector<Head> cpu_forward(const Network& net, const std::vector<float>& input) {
    const auto& layers = net.layers();
    std::vector<std::vector<float>> outputs(layers.size());
    std::vector<Head> heads;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto& layer = layers[index];
        const auto& source = index ? outputs[index - 1] : input;
        auto& out = outputs[index];
        out.resize(layer.output.elements());
        if (layer.kind == Kind::convolutional) {
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            for (int oc = 0; oc < layer.output.c; ++oc) {
                float scale = layer.batch_normalize ? layer.scales[oc] / std::sqrt(layer.variances[oc] + 1e-5f) : 1.0f;
                float bias = layer.batch_normalize ? layer.biases[oc] - layer.means[oc] * scale : layer.biases[oc];
                for (int oy = 0; oy < layer.output.h; ++oy)
                    for (int ox = 0; ox < layer.output.w; ++ox) {
                        float sum = 0;
                        for (int ic = 0; ic < layer.input.c; ++ic)
                            for (int ky = 0; ky < layer.size; ++ky)
                                for (int kx = 0; kx < layer.size; ++kx) {
                                    int iy = oy * layer.stride + ky - layer.padding;
                                    int ix = ox * layer.stride + kx - layer.padding;
                                    if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                                    std::size_t wi = ((static_cast<std::size_t>(oc) * layer.input.c + ic) * layer.size + ky) * layer.size + kx;
                                    sum += source[static_cast<std::size_t>(ic) * in_plane + iy * layer.input.w + ix] * layer.weights[wi];
                                }
                        float value = sum * scale + bias;
                        out[static_cast<std::size_t>(oc) * out_plane + oy * layer.output.w + ox] = layer.leaky && value < 0 ? value * 0.1f : value;
                    }
            }
        } else if (layer.kind == Kind::maxpool) {
            int offset = (layer.size - 1) / 2;
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            for (int c = 0; c < layer.input.c; ++c)
                for (int oy = 0; oy < layer.output.h; ++oy)
                    for (int ox = 0; ox < layer.output.w; ++ox) {
                        float best = -std::numeric_limits<float>::infinity();
                        for (int ky = 0; ky < layer.size; ++ky)
                            for (int kx = 0; kx < layer.size; ++kx) {
                                int iy = oy * layer.stride + ky - offset;
                                int ix = ox * layer.stride + kx - offset;
                                if (iy < 0 || ix < 0 || iy >= layer.input.h || ix >= layer.input.w) continue;
                                best = std::max(best, source[static_cast<std::size_t>(c) * in_plane + iy * layer.input.w + ix]);
                            }
                        out[static_cast<std::size_t>(c) * out_plane + oy * layer.output.w + ox] = best;
                    }
        } else if (layer.kind == Kind::route) {
            std::size_t offset = 0;
            for (int branch : layer.sources) {
                const auto& values = outputs[static_cast<std::size_t>(branch)];
                std::copy(values.begin(), values.end(), out.begin() + static_cast<std::ptrdiff_t>(offset));
                offset += values.size();
            }
        } else if (layer.kind == Kind::upsample) {
            int in_plane = layer.input.h * layer.input.w;
            int out_plane = layer.output.h * layer.output.w;
            for (int c = 0; c < layer.input.c; ++c)
                for (int y = 0; y < layer.output.h; ++y)
                    for (int x = 0; x < layer.output.w; ++x)
                        out[static_cast<std::size_t>(c) * out_plane + y * layer.output.w + x] =
                            source[static_cast<std::size_t>(c) * in_plane + (y / layer.stride) * layer.input.w + x / layer.stride];
        } else if (layer.kind == Kind::shortcut) {
            const auto& branch = outputs[static_cast<std::size_t>(layer.sources.front())];
            for (std::size_t j = 0; j < out.size(); ++j) {
                float value = source[j] + branch[j];
                out[j] = layer.leaky && value < 0 ? value * 0.1f : value;
            }
        } else if (layer.kind == Kind::connected) {
            for (int oc = 0; oc < layer.output.c; ++oc) {
                float value = layer.biases[static_cast<std::size_t>(oc)];
                for (std::size_t ic = 0; ic < source.size(); ++ic)
                    value += source[ic] * layer.weights[static_cast<std::size_t>(oc) * source.size() + ic];
                out[static_cast<std::size_t>(oc)] = layer.leaky && value < 0 ? value * 0.1f : value;
            }
        } else if (layer.kind == Kind::softmax) {
            float maximum = *std::max_element(source.begin(), source.end());
            float total = 0.0f;
            for (std::size_t j = 0; j < source.size(); ++j) {
                out[j] = std::exp(source[j] - maximum);
                total += out[j];
            }
            for (float& value : out) value /= total;
            heads.push_back({index, out});
        } else {
            out = source;
            heads.push_back({index, out});
        }
    }
    return heads;
}

}  // namespace

std::size_t Shape::elements() const {
    if (c <= 0 || h <= 0 || w <= 0) throw std::runtime_error("invalid tensor shape");
    return static_cast<std::size_t>(c) * h * w;
}

Network Network::from_cfg(const std::string& path) {
    auto sections = sections_from(path);
    if (sections.empty() || sections.front().name != "net") throw std::runtime_error("cfg must start with [net]");
    if (sections.size() > 4097) throw std::runtime_error("too many layers");
    Network net;
    net.input_ = {integer(sections.front(), "channels"), integer(sections.front(), "height"), integer(sections.front(), "width")};
    positive(net.input_.c, "channels"); positive(net.input_.h, "height"); positive(net.input_.w, "width");
    Shape shape = net.input_;
    for (std::size_t i = 1; i < sections.size(); ++i) {
        const auto& section = sections[i];
        Layer layer;
        layer.input = shape;
        if (section.name == "convolutional") {
            layer.kind = Kind::convolutional;
            layer.filters = integer(section, "filters");
            layer.size = integer(section, "size");
            layer.stride = integer(section, "stride", "1");
            positive(layer.filters, "filters"); positive(layer.size, "size"); positive(layer.stride, "stride");
            layer.padding = integer(section, "pad", "0") ? layer.size / 2 : integer(section, "padding", "0");
            if (layer.padding < 0) throw std::runtime_error("negative padding");
            layer.batch_normalize = integer(section, "batch_normalize", "0") != 0;
            auto activation = option(section, "activation", "linear");
            if (activation != "linear" && activation != "leaky") throw std::runtime_error("unsupported activation: " + activation);
            layer.leaky = activation == "leaky";
            int nh = shape.h + 2 * layer.padding - layer.size;
            int nw = shape.w + 2 * layer.padding - layer.size;
            if (nh < 0 || nw < 0) throw std::runtime_error("convolution filter exceeds input");
            layer.output = {layer.filters, nh / layer.stride + 1, nw / layer.stride + 1};
        } else if (section.name == "maxpool") {
            layer.kind = Kind::maxpool;
            layer.size = integer(section, "size");
            layer.stride = integer(section, "stride");
            positive(layer.size, "size"); positive(layer.stride, "stride");
            layer.output = {shape.c, (shape.h - 1) / layer.stride + 1, (shape.w - 1) / layer.stride + 1};
        } else if (section.name == "route") {
            layer.kind = Kind::route;
            int channels = 0;
            for (int raw : list(section, "layers")) {
                int resolved = raw < 0 ? static_cast<int>(net.layers_.size()) + raw : raw;
                if (resolved < 0 || resolved >= static_cast<int>(net.layers_.size())) throw std::runtime_error("route references missing layer");
                Shape branch = net.layers_[static_cast<std::size_t>(resolved)].output;
                if (branch.h != shape.h || branch.w != shape.w) throw std::runtime_error("route spatial shapes differ");
                channels += branch.c;
                layer.sources.push_back(resolved);
            }
            layer.output = {channels, shape.h, shape.w};
        } else if (section.name == "upsample") {
            layer.kind = Kind::upsample;
            layer.stride = integer(section, "stride");
            positive(layer.stride, "upsample stride");
            layer.output = {shape.c, shape.h * layer.stride, shape.w * layer.stride};
        } else if (section.name == "shortcut") {
            layer.kind = Kind::shortcut;
            int from = integer(section, "from");
            int resolved = from < 0 ? static_cast<int>(net.layers_.size()) + from : from;
            if (resolved < 0 || resolved >= static_cast<int>(net.layers_.size())) throw std::runtime_error("shortcut references missing layer");
            Shape branch = net.layers_[static_cast<std::size_t>(resolved)].output;
            if (branch.c != shape.c || branch.h != shape.h || branch.w != shape.w)
                throw std::runtime_error("shortcut requires matching input/branch shape (no channel projection)");
            auto activation = option(section, "activation", "linear");
            if (activation != "linear" && activation != "leaky") throw std::runtime_error("unsupported shortcut activation: " + activation);
            layer.leaky = activation == "leaky";
            layer.sources.push_back(resolved);
            layer.output = shape;
        } else if (section.name == "connected") {
            layer.kind = Kind::connected;
            layer.filters = integer(section, "output");
            positive(layer.filters, "connected output");
            auto activation = option(section, "activation", "linear");
            if (activation != "linear" && activation != "leaky")
                throw std::runtime_error("unsupported connected activation: " + activation);
            layer.leaky = activation == "leaky";
            layer.output = {layer.filters, 1, 1};
        } else if (section.name == "softmax") {
            layer.kind = Kind::softmax;
            if (shape.h != 1 || shape.w != 1 || i + 1 != sections.size())
                throw std::runtime_error("[softmax] requires vector input and must be the final layer");
            layer.output = shape;
        } else if (section.name == "yolo") {
            layer.kind = Kind::yolo;
            layer.classes = integer(section, "classes");
            positive(layer.classes, "classes");
            layer.mask = list(section, "mask");
            auto anchors = list(section, "anchors");
            if (anchors.size() % 2) throw std::runtime_error("anchors require width,height pairs");
            for (std::size_t a = 0; a < anchors.size(); a += 2) {
                positive(anchors[a], "anchor width"); positive(anchors[a + 1], "anchor height");
                layer.anchors.emplace_back(static_cast<float>(anchors[a]), static_cast<float>(anchors[a + 1]));
            }
            for (int mask : layer.mask) if (mask < 0 || mask >= static_cast<int>(layer.anchors.size()))
                throw std::runtime_error("yolo mask references missing anchor");
            if (shape.c != static_cast<int>(layer.mask.size()) * (layer.classes + 5))
                throw std::runtime_error("yolo channels must equal mask count * (classes + 5)");
            layer.ignore_thresh = floating(section, "ignore_thresh", "0.7");
            if (!(layer.ignore_thresh > 0.0f && layer.ignore_thresh <= 1.0f))
                throw std::runtime_error("ignore_thresh must be in (0,1]");
            layer.output = shape;
        } else {
            throw std::runtime_error("unsupported layer [" + section.name + "] at line " + std::to_string(section.line));
        }
        layer.output.elements();
        shape = layer.output;
        net.layers_.push_back(std::move(layer));
    }
    if (net.layers_.empty() ||
        (std::none_of(net.layers_.begin(), net.layers_.end(), [](const Layer& layer) { return layer.kind == Kind::yolo; }) &&
         net.layers_.back().kind != Kind::softmax))
        throw std::runtime_error("cfg needs a [yolo] head or final [softmax]");
    return net;
}

void Network::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open weights: " + path);
    std::int32_t major, minor, revision;
    read_scalar(file, major); read_scalar(file, minor); read_scalar(file, revision);
    if (major < 0 || minor < 0 || major >= 1000 || minor >= 1000) throw std::runtime_error("unsupported weights version");
    if (major * 10 + minor >= 2) { std::uint64_t seen; read_scalar(file, seen); }
    else { std::uint32_t seen; read_scalar(file, seen); }
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        if (layer.kind != Kind::convolutional && layer.kind != Kind::connected) continue;
        auto count = static_cast<std::size_t>(layer.filters);
        read_floats(file, layer.biases, count, i);
        if (layer.kind == Kind::convolutional && layer.batch_normalize) {
            read_floats(file, layer.scales, count, i);
            read_floats(file, layer.means, count, i);
            read_floats(file, layer.variances, count, i);
        }
        std::size_t fan_in = layer.kind == Kind::connected ? layer.input.elements() :
            static_cast<std::size_t>(layer.input.c) * layer.size * layer.size;
        read_floats(file, layer.weights, count * fan_in, i);
    }
    char trailing = 0;
    if (file.read(&trailing, 1)) throw std::runtime_error("weights contain trailing data; cfg may not match weights");
    if (!file.eof()) throw std::runtime_error("error reading weights file");
}

std::vector<Head> Network::forward(const std::vector<float>& input, Backend backend) const {
    if (input.size() != input_.elements()) throw std::runtime_error("input does not match cfg shape");
    for (const auto& layer : layers_) if ((layer.kind == Kind::convolutional || layer.kind == Kind::connected) && layer.biases.empty())
        throw std::runtime_error("load weights before inference");
    if (backend == Backend::cpu) return cpu_forward(*this, input);
#ifdef DETECTOR_HAS_CUDA
    if (backend == Backend::cuda) return forward_cuda(*this, input);
#endif
    throw std::runtime_error("CUDA backend not built; use -DDETECTOR_CUDA=ON");
}

std::vector<float> Network::classify(const std::vector<float>& input, Backend backend) const {
    if (layers_.empty() || layers_.back().kind != Kind::softmax)
        throw std::runtime_error("classification requires final [softmax]");
    auto outputs = forward(input, backend);
    if (outputs.empty() || outputs.back().layer != layers_.size() - 1)
        throw std::runtime_error("missing classification output");
    return outputs.back().values;
}

std::vector<Detection> Network::detect(const std::vector<float>& input, Backend backend,
                                        float threshold, float iou_threshold) const {
    return decode(*this, forward(input, backend), threshold, iou_threshold);
}

std::vector<Detection> non_max_suppression(std::vector<Detection> boxes, float threshold) {
    if (!(threshold >= 0 && threshold <= 1)) throw std::runtime_error("IoU threshold must be in [0,1]");
    std::sort(boxes.begin(), boxes.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::unordered_map<int, std::vector<Detection>> by_class;
    std::vector<Detection> kept;
    for (const auto& box : boxes) {
        auto& same_class = by_class[box.class_id];
        bool discard = false;
        for (const auto& previous : same_class)
            if (iou(box, previous) > threshold) { discard = true; break; }
        if (!discard) {
            same_class.push_back(box);
            kept.push_back(box);
        }
    }
    return kept;
}

std::vector<Detection> decode(const Network& net, const std::vector<Head>& heads,
                              float threshold, float iou_threshold) {
    if (!(threshold >= 0 && threshold <= 1)) throw std::runtime_error("confidence threshold must be in [0,1]");
    std::vector<Detection> boxes;
    Shape input = net.input_shape();
    for (const auto& head : heads) {
        if (head.layer >= net.layers().size()) throw std::runtime_error("invalid YOLO head index");
        const auto& layer = net.layers()[head.layer];
        if (layer.kind != Kind::yolo || head.values.size() != layer.output.elements()) throw std::runtime_error("invalid YOLO head tensor");
        int area = layer.output.h * layer.output.w;
        int attributes = layer.classes + 5;
        auto at = [&](int anchor, int attr, int cell) {
            return head.values[(static_cast<std::size_t>(anchor) * attributes + attr) * area + cell];
        };
        for (int a = 0; a < static_cast<int>(layer.mask.size()); ++a) {
            auto [anchor_w, anchor_h] = layer.anchors[static_cast<std::size_t>(layer.mask[a])];
            for (int y = 0; y < layer.output.h; ++y)
                for (int x = 0; x < layer.output.w; ++x) {
                    int cell = y * layer.output.w + x;
                    float objectness = sigmoid(at(a, 4, cell));
                    if (objectness < threshold) continue;
                    float cx = (sigmoid(at(a, 0, cell)) + x) * input.w / layer.output.w;
                    float cy = (sigmoid(at(a, 1, cell)) + y) * input.h / layer.output.h;
                    float width = std::exp(std::clamp(at(a, 2, cell), -20.0f, 20.0f)) * anchor_w;
                    float height = std::exp(std::clamp(at(a, 3, cell), -20.0f, 20.0f)) * anchor_h;
                    for (int cls = 0; cls < layer.classes; ++cls) {
                        float score = objectness * sigmoid(at(a, 5 + cls, cell));
                        if (score >= threshold) boxes.push_back({cls, score, cx - width / 2, cy - height / 2,
                                                                  cx + width / 2, cy + height / 2});
                    }
                }
        }
    }
    return non_max_suppression(std::move(boxes), iou_threshold);
}

}  // namespace detector

