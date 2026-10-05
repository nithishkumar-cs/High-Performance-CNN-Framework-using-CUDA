"""Optional independent Darknet-format inference check; not the project runtime.

Requires opencv-python and numpy. Uses the same cfg/weights but OpenCV DNN's
own graph importer, kernels, and YOLO decoder. Image interpolation can differ
slightly from the native detector, so compare results with a small tolerance.
"""

import argparse

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg")
    parser.add_argument("weights")
    parser.add_argument("image")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    height, width = image.shape[:2]
    side = 416
    scale = min(side / width, side / height)
    resized_width = max(1, min(side, int(np.floor(width * scale + 0.5))))
    resized_height = max(1, min(side, int(np.floor(height * scale + 0.5))))
    left = (side - resized_width) // 2
    top = (side - resized_height) // 2
    canvas = np.full((side, side, 3), 0.5, dtype=np.float32)
    scaled = cv2.resize(image.astype(np.float32) / 255.0,
                        (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas[top:top + resized_height, left:left + resized_width] = scaled
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0, size=(side, side),
                                 swapRB=True, crop=False)
    net = cv2.dnn.readNetFromDarknet(args.cfg, args.weights)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    for output in outputs:
        for row in output:
            objectness = float(row[4])
            for class_id, probability in enumerate(row[5:]):
                score = objectness * float(probability)
                if score < args.threshold:
                    continue
                cx, cy, box_width, box_height = map(float, row[:4])
                x1 = ((cx - box_width / 2) * side - left) * width / resized_width
                y1 = ((cy - box_height / 2) * side - top) * height / resized_height
                x2 = ((cx + box_width / 2) * side - left) * width / resized_width
                y2 = ((cy + box_height / 2) * side - top) * height / resized_height
                boxes.append((class_id, score, *[max(0.0, min(limit, value))
                                                 for value, limit in zip((x1, y1, x2, y2),
                                                                         (width, height, width, height))]))
    for box in sorted(boxes, key=lambda item: item[1], reverse=True)[:10]:
        print(f"{box[0]} {box[1]:.4f} {box[2]:.4f} {box[3]:.4f} {box[4]:.4f} {box[5]:.4f}")
    print(f"raw_candidates={len(boxes)}")


if __name__ == "__main__":
    main()
