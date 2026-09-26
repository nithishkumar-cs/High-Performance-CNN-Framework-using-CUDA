#!/usr/bin/env python3
'''Run an independently trained raw-head detector on one image.'''

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from scratch_yolo.decode import decode_heads
from scratch_yolo.model import TinyDetector
from scratch_yolo.preprocess import letterbox


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--classes', type=Path, default=Path('data/voc2007/yolo/classes.txt'))
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--nms-iou', type=float, default=.45)
    parser.add_argument('--annotated', type=Path)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    if checkpoint.get('train_limit') or checkpoint.get('val_limit'):
        print('WARNING: smoke-test checkpoint, not a fully trained model')
    side = int(checkpoint['side'])
    count = int(checkpoint['classes'])
    names = args.classes.read_text(encoding='utf-8').splitlines()
    if len(names) != count:
        parser.error('class-name count differs from checkpoint')
    model = TinyDetector(classes=count).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    with Image.open(args.image) as opened:
        source = opened.convert('RGB')
    old_w, old_h = source.size
    tensor, scale, left, top = letterbox(source, side)
    with torch.inference_mode():
        predictions = decode_heads(model(torch.from_numpy(tensor).to(device)),
                                   args.confidence, args.nms_iou)[0].cpu()
    detections = []
    for x1, y1, x2, y2, score, class_id in predictions.tolist():
        box = [max(0, min(old_w, round((x1 * side - left) / scale))),
               max(0, min(old_h, round((y1 * side - top) / scale))),
               max(0, min(old_w, round((x2 * side - left) / scale))),
               max(0, min(old_h, round((y2 * side - top) / scale)))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append({'class_id': int(class_id), 'class_name': names[int(class_id)],
                           'confidence': score, 'box': box})
    if args.annotated:
        draw = ImageDraw.Draw(source)
        for item in detections:
            draw.rectangle(item['box'], outline='red', width=2)
            draw.text((item['box'][0], item['box'][1]),
                      f'{item["class_name"]} {item["confidence"]:.2f}', fill='red')
        args.annotated.parent.mkdir(parents=True, exist_ok=True)
        source.save(args.annotated)
    print(json.dumps({'image': str(args.image), 'detections': detections}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
