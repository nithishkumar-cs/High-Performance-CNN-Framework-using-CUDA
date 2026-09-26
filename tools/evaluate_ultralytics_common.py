#!/usr/bin/env python3
'''Score the YOLOv8n baseline with the scratch project's AP@0.5 evaluator.'''

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

from scratch_yolo.data import VocYoloDataset
from scratch_yolo.metrics import ap50


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path,
                        default=Path('runs/voc2007/yolov8n/weights/best.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/voc2007/yolo'))
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--side', type=int, default=416)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--confidence', type=float, default=.01)
    parser.add_argument('--nms-iou', type=float, default=.6)
    parser.add_argument('--limit-images', type=int, default=0, help='smoke test only')
    parser.add_argument('--output', type=Path,
                        default=Path('runs/voc2007/yolov8n/common_ap50.json'))
    args = parser.parse_args()
    if args.batch < 1 or args.limit_images < 0:
        parser.error('invalid batch or image limit')
    dataset = VocYoloDataset(args.data, args.split, args.side)
    full_size = len(dataset)
    size = min(args.limit_images, full_size) if args.limit_images else full_size
    model = YOLO(str(args.weights))
    device = (0 if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else (
        0 if args.device == 'cuda' else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    predictions, truth = [], []
    for start in range(0, size, args.batch):
        stop = min(start + args.batch, size)
        results = model.predict([str(path) for path in dataset.images[start:stop]],
                                imgsz=args.side, conf=args.confidence,
                                iou=args.nms_iou, max_det=300, batch=args.batch,
                                device=device, verbose=False)
        for index, result in enumerate(results, start=start):
            height, width = result.orig_shape
            scale = min(args.side / width, args.side / height)
            left = (args.side - round(width * scale)) // 2
            top = (args.side - round(height * scale)) // 2
            boxes = result.boxes.xyxy.cpu().float().clone()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * scale + left) / args.side
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * scale + top) / args.side
            boxes = boxes.clamp(0, 1)
            scores = result.boxes.conf.cpu().float().unsqueeze(1)
            labels = result.boxes.cls.cpu().float().unsqueeze(1)
            predictions.append(torch.cat((boxes, scores, labels), dim=1))
            truth.append(dataset[index][1])
        if (stop // args.batch) % 50 == 0:
            print(f'evaluated {stop}/{size} images', flush=True)
    classes = len((args.data / 'classes.txt').read_text(encoding='utf-8').splitlines())
    metrics = ap50(predictions, truth, classes)
    names = (args.data / 'classes.txt').read_text(encoding='utf-8').splitlines()
    summary = {
        'evaluated_utc': datetime.now(timezone.utc).isoformat(),
        'model': 'ultralytics_yolov8n_pretrained_finetuned',
        'weights': str(args.weights.resolve()), 'split': args.split,
        'images': size, 'full_split_images': full_size, 'side': args.side,
        'device': str(device),
        'confidence': args.confidence, 'nms_iou': args.nms_iou,
        'metric': '101-point interpolated AP@0.5 via scratch_yolo.metrics',
        'map50': metrics['map50'],
        'per_class_ap50': {names[key]: value for key, value in metrics['per_class_ap50'].items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
