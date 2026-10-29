"""Export this project's supported Darknet detector graphs to standard ONNX.

Inputs are a matching Darknet .cfg and .weights; no PyTorch or upstream Darknet
source is used. Supports convolutional/maxpool/route/upsample/shortcut/reorg
bodies with one or more [yolo] and/or [region] heads - i.e. YOLOv3-tiny (2
heads), full YOLOv3/Darknet-53 (3 heads, using shortcut for its residual
blocks), and YOLOv2/Darknet19 (1 region head, using reorg for its passthrough
layer). Outputs are raw NCHW YOLO/region head
logits, one per head in layer order. Letterboxing, decode, and NMS
intentionally remain outside the ONNX graph so the same postprocessor can be
used for native CUDA and TensorRT accuracy comparisons. The input cfg may be
an adaptation of a pjreddie/darknet cfg; see
https://github.com/pjreddie/darknet/blob/master/LICENSE before redistribution.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def sections_from(path: Path) -> list[tuple[str, dict[str, str]]]:
    """Parse a Darknet .cfg into ordered (section_name, options) pairs; raises ValueError on malformed cfg."""
    sections: list[tuple[str, dict[str, str]]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            if not line.endswith("]"):
                raise ValueError(f"bad cfg section at line {number}")
            sections.append((line[1:-1].strip(), {}))
        else:
            if not sections or "=" not in line:
                raise ValueError(f"bad cfg option at line {number}")
            key, value = (part.strip() for part in line.split("=", 1))
            if not key or not value:
                raise ValueError(f"empty cfg option at line {number}")
            sections[-1][1][key] = value
    if not sections or sections[0][0] != "net":
        raise ValueError("cfg must start with [net]")
    return sections


def integers(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",")]


def darknet_floats(path: Path) -> tuple[np.ndarray, int]:
    """Read a Darknet .weights file's header and float payload; returns (weights, images_seen); raises ValueError if invalid."""
    with path.open("rb") as handle:
        major, minor, revision = struct.unpack("<iii", handle.read(12))
        if major < 0 or minor < 0 or major >= 1000 or minor >= 1000:
            raise ValueError("unsupported Darknet weights version")
        seen = struct.unpack("<Q" if major * 10 + minor >= 2 else "<I",
                             handle.read(8 if major * 10 + minor >= 2 else 4))[0]
        values = np.fromfile(handle, dtype="<f4")
    if revision < 0 or not np.isfinite(values).all():
        raise ValueError("invalid Darknet weights")
    return values, seen


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(cfg: Path, weights: Path, output: Path) -> dict[str, object]:
    """Convert this project's supported Darknet detector cfg+weights (YOLOv3-tiny/full YOLOv3-Darknet-53/YOLOv2-region) to an ONNX model at `output`; returns a summary dict."""
    sections = sections_from(cfg)
    net = sections[0][1]
    channels, height, width = (int(net[key]) for key in ("channels", "height", "width"))
    if min(channels, height, width) <= 0:
        raise ValueError("invalid network input shape")
    floats, seen = darknet_floats(weights)
    cursor = 0

    def take(count: int) -> np.ndarray:
        nonlocal cursor
        if count < 0 or cursor + count > floats.size:
            raise ValueError(f"truncated Darknet weights at float {cursor}")
        result = floats[cursor:cursor + count].copy()
        cursor += count
        return result

    nodes = []
    initializers = []
    shapes: list[tuple[int, int, int]] = []
    tensors: list[str] = []
    outputs = []
    previous = "images"
    shape = (channels, height, width)
    head_count = 0
    classes = None
    for index, (kind, options) in enumerate(sections[1:]):
        c, h, w = shape
        name = f"layer_{index}"
        if kind == "convolutional":
            filters = int(options["filters"])
            size = int(options["size"])
            stride = int(options.get("stride", "1"))
            padding = size // 2 if int(options.get("pad", "0")) else int(options.get("padding", "0"))
            if min(filters, size, stride) <= 0 or padding < 0:
                raise ValueError(f"invalid convolution at layer {index}")
            normalized = int(options.get("batch_normalize", "0")) != 0
            biases = take(filters)
            if normalized:
                scales, means, variances = take(filters), take(filters), take(filters)
            kernel = take(filters * c * size * size).reshape(filters, c, size, size)
            if normalized:
                scale = scales / np.sqrt(variances + np.float32(1e-5))
                kernel *= scale[:, None, None, None]
                biases -= means * scale
            weight_name, bias_name = f"{name}_weight", f"{name}_bias"
            initializers.extend((numpy_helper.from_array(kernel, weight_name),
                                 numpy_helper.from_array(biases, bias_name)))
            conv_name = f"{name}_conv"
            nodes.append(helper.make_node("Conv", [previous, weight_name, bias_name],
                                          [conv_name], name=conv_name,
                                          pads=[padding] * 4, strides=[stride, stride]))
            activation = options.get("activation", "linear")
            if activation == "leaky":
                nodes.append(helper.make_node("LeakyRelu", [conv_name], [name],
                                              name=f"{name}_leaky", alpha=0.1))
            elif activation == "linear":
                nodes.append(helper.make_node("Identity", [conv_name], [name], name=name))
            else:
                raise ValueError(f"unsupported activation {activation} at layer {index}")
            shape = (filters, (h + 2 * padding - size) // stride + 1,
                     (w + 2 * padding - size) // stride + 1)
            previous = name
        elif kind == "deconvolutional":
            filters = int(options["filters"])
            size = int(options["size"])
            stride = int(options.get("stride", "1"))
            padding = int(options.get("padding", "0"))
            if min(filters, size, stride) <= 0 or padding < 0 or int(options.get("batch_normalize", "0")):
                raise ValueError(f"invalid deconvolution at layer {index}")
            biases = take(filters)
            # Weight layout [in_channel][filter][ky][kx] is exactly ONNX ConvTranspose's (C_in, C_out, kH, kW).
            kernel = take(c * filters * size * size).reshape(c, filters, size, size)
            weight_name, bias_name = f"{name}_weight", f"{name}_bias"
            initializers.extend((numpy_helper.from_array(kernel, weight_name),
                                 numpy_helper.from_array(biases, bias_name)))
            conv_name = f"{name}_deconv"
            nodes.append(helper.make_node("ConvTranspose", [previous, weight_name, bias_name],
                                          [conv_name], name=conv_name,
                                          pads=[padding] * 4, strides=[stride, stride]))
            activation = options.get("activation", "linear")
            if activation == "leaky":
                nodes.append(helper.make_node("LeakyRelu", [conv_name], [name],
                                              name=f"{name}_leaky", alpha=0.1))
            elif activation == "linear":
                nodes.append(helper.make_node("Identity", [conv_name], [name], name=name))
            else:
                raise ValueError(f"unsupported activation {activation} at layer {index}")
            shape = (filters, stride * (h - 1) + size - 2 * padding, stride * (w - 1) + size - 2 * padding)
            previous = name
        elif kind == "maxpool":
            size, stride = int(options["size"]), int(options["stride"])
            if min(size, stride) <= 0:
                raise ValueError(f"invalid maxpool at layer {index}")
            out_h, out_w = (h - 1) // stride + 1, (w - 1) // stride + 1
            top = left = (size - 1) // 2
            bottom = max(0, (out_h - 1) * stride + size - h - top)
            right = max(0, (out_w - 1) * stride + size - w - left)
            nodes.append(helper.make_node("MaxPool", [previous], [name], name=name,
                                          kernel_shape=[size, size], strides=[stride, stride],
                                          pads=[top, left, bottom, right]))
            shape = (c, out_h, out_w)
            previous = name
        elif kind == "route":
            raw = integers(options["layers"])
            sources = [len(tensors) + value if value < 0 else value for value in raw]
            if not sources or any(source < 0 or source >= len(tensors) for source in sources):
                raise ValueError(f"invalid route at layer {index}")
            # Compare every source branch's spatial shape against the FIRST branch's shape, not
            # against `shape` (the immediately-preceding layer's output) - a route layer's output
            # never depends on what came immediately before it, only on its listed sources. Using
            # `shape` here silently worked only when a route's sources happened to match the
            # preceding layer (e.g. YOLOv3-tiny/mini.cfg's patterns) and incorrectly rejected a
            # standalone passthrough route (YOLOv2's `[route] layers=-9`) - the same bug found and
            # fixed in network.cpp's layer_output_shape (req-31/req-37, 2026-10-13).
            if any(shapes[source][1:] != shapes[sources[0]][1:] for source in sources):
                raise ValueError(f"route spatial shapes differ at layer {index}")
            if len(sources) == 1:
                nodes.append(helper.make_node("Identity", [tensors[sources[0]]], [name], name=name))
            else:
                nodes.append(helper.make_node("Concat", [tensors[source] for source in sources],
                                              [name], name=name, axis=1))
            shape = (sum(shapes[source][0] for source in sources), shapes[sources[0]][1], shapes[sources[0]][2])
            previous = name
        elif kind == "upsample":
            stride = int(options["stride"])
            if stride <= 0:
                raise ValueError(f"invalid upsample at layer {index}")
            scales_name = f"{name}_scales"
            initializers.append(numpy_helper.from_array(
                np.array([1, 1, stride, stride], dtype=np.float32), scales_name))
            nodes.append(helper.make_node("Resize", [previous, "", scales_name], [name],
                                          name=name, mode="nearest",
                                          coordinate_transformation_mode="asymmetric",
                                          nearest_mode="floor"))
            shape = (c, h * stride, w * stride)
            previous = name
        elif kind == "reorg":
            # Space-to-depth (YOLOv2's passthrough layer), this project's own independently-derived
            # channel/spatial convention (see network.cpp's reorg_forward): output channel
            # ic*stride^2 + (iy%stride)*stride + (ix%stride) at spatial (iy//stride, ix//stride).
            # Reshape [1,C,H,W] -> [1,C,OH,S,OW,S] (splitting H/W into outer,inner by stride),
            # transpose the two inner (stride) axes before the two outer (OH,OW) axes, then
            # reshape to [1, C*S*S, OH, OW] - a bijection, so any consistent mapping is correct for
            # training, but this one is NOT guaranteed to match original Darknet's own interleaving.
            stride = int(options["stride"])
            if stride <= 0 or h % stride or w % stride:
                raise ValueError(f"invalid reorg at layer {index}")
            out_h, out_w = h // stride, w // stride
            split_name, merge_name = f"{name}_split_shape", f"{name}_merge_shape"
            initializers.append(numpy_helper.from_array(
                np.array([1, c, out_h, stride, out_w, stride], dtype=np.int64), split_name))
            initializers.append(numpy_helper.from_array(
                np.array([1, c * stride * stride, out_h, out_w], dtype=np.int64), merge_name))
            split_tensor, transposed = f"{name}_split", f"{name}_transposed"
            nodes.append(helper.make_node("Reshape", [previous, split_name], [split_tensor], name=f"{name}_reshape_split"))
            nodes.append(helper.make_node("Transpose", [split_tensor], [transposed],
                                          name=f"{name}_transpose", perm=[0, 1, 3, 5, 2, 4]))
            nodes.append(helper.make_node("Reshape", [transposed, merge_name], [name], name=f"{name}_reshape_merge"))
            shape = (c * stride * stride, out_h, out_w)
            previous = name
        elif kind == "shortcut":
            from_raw = int(options["from"])
            source = len(tensors) + from_raw if from_raw < 0 else from_raw
            if source < 0 or source >= len(tensors):
                raise ValueError(f"shortcut references missing layer at {index}")
            if shapes[source] != shape:
                raise ValueError(f"shortcut shape mismatch at layer {index}")
            activation = options.get("activation", "linear")
            if activation not in ("linear", "leaky"):
                raise ValueError(f"unsupported shortcut activation {activation} at layer {index}")
            add_name = f"{name}_add" if activation == "leaky" else name
            nodes.append(helper.make_node("Add", [previous, tensors[source]], [add_name], name=add_name))
            if activation == "leaky":
                nodes.append(helper.make_node("LeakyRelu", [add_name], [name], name=f"{name}_leaky", alpha=0.1))
            previous = name
        elif kind in ("yolo", "region"):
            if kind == "yolo":
                anchor_count = len(integers(options["mask"]))
            else:
                anchor_count = int(options.get("num", "1"))
            count = int(options["classes"])
            if c != anchor_count * (count + 5) or (classes is not None and classes != count):
                raise ValueError(f"{kind} head channels/classes disagree at layer {index}")
            classes = count
            head_name = f"head_{head_count}"
            nodes.append(helper.make_node("Identity", [previous], [head_name],
                                          name=f"{name}_raw_head"))
            outputs.append(helper.make_tensor_value_info(
                head_name, TensorProto.FLOAT, [1, c, h, w]))
            head_count += 1
            previous = head_name
        else:
            raise ValueError(f"unsupported layer [{kind}] at index {index}")
        if min(shape) <= 0:
            raise ValueError(f"invalid output shape at layer {index}")
        tensors.append(previous)
        shapes.append(shape)
    segmentation = head_count == 0 and sections[-1][0] in ("convolutional", "deconvolutional")
    if segmentation:
        # Dense-prediction cfg (semantic logits / binary mask logits): the last layer is the output.
        nodes.append(helper.make_node("Identity", [previous], ["logits"], name="segmentation_logits"))
        outputs.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, *shape]))
        head_count = 1
    if head_count < 1 or cursor != floats.size:
        raise ValueError(f"expected at least one complete YOLO/region head or a convolutional segmentation "
                         f"output; got {head_count}, unused floats={floats.size - cursor}")
    graph = helper.make_graph(nodes, "darknet_detector_raw_heads",
                              [helper.make_tensor_value_info(
                                  "images", TensorProto.FLOAT, [1, channels, height, width])],
                              outputs, initializer=initializers)
    model = helper.make_model(graph, producer_name="native-darknet-onnx-export",
                              opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 9
    metadata = {"darknet_cfg_sha256": sha256(cfg),
                "darknet_weights_sha256": sha256(weights),
                "darknet_images_seen": str(seen),
                "output_semantics": "raw NCHW per-pixel logits (softmax/sigmoid external)" if segmentation else
                    "raw NCHW YOLO/region head logits; decode and NMS external"}
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key, item.value = key, value
    onnx.checker.check_model(model)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite ONNX artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output)
    return {"output": str(output), "input": [1, channels, height, width],
            "heads": [list(item.type.tensor_type.shape.dim[i].dim_value for i in range(4))
                      for item in outputs], "classes": classes, "images_seen": seen,
            "cfg_sha256": metadata["darknet_cfg_sha256"],
            "weights_sha256": metadata["darknet_weights_sha256"],
            "onnx_sha256": sha256(output)}


def main() -> None:
    """CLI entry point: export Darknet cfg+weights to ONNX and print the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(export(args.cfg, args.weights, args.output), indent=2))


if __name__ == "__main__":
    main()
