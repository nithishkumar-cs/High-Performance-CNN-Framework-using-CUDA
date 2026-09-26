#!/usr/bin/env python3
'''Evaluate scratch detector AP@0.5 on the fixed VOC validation/test split.'''

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from scratch_yolo.data import VocYoloDataset, collate_detection
from scratch_yolo.decode import decode_heads
from scratch_yolo.metrics import ap50
from scratch_yolo.model import TinyDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/voc2007/yolo'))
    parser.add_argument('--classes', type=Path, default=Path('data/voc2007/yolo/classes.txt'))
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--confidence', type=float, default=.01)
    parser.add_argument('--nms-iou', type=float, default=.6)
    parser.add_argument('--limit-images', type=int, default=0, help='smoke test only')
    parser.add_argument('--output', type=Path, default=Path('runs/scratch_yolo/evaluation.json'))
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or args.limit_images < 0:
        parser.error('invalid batch, workers, or limit')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    names = args.classes.read_text(encoding='utf-8').splitlines()
    count = int(checkpoint['classes'])
    if len(names) != count:
        parser.error('class-name count differs from checkpoint')
    model = TinyDetector(count).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    dataset = VocYoloDataset(args.data, args.split, int(checkpoint['side']))
    full_size = len(dataset)
    if args.limit_images:
        dataset = Subset(dataset, range(min(args.limit_images, full_size)))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_detection)
    predictions, truth = [], []
    with torch.inference_mode():
        for batch_id, (images, targets) in enumerate(loader):
            raw = model(images.to(device))
            predictions.extend(boxes.cpu() for boxes in
                               decode_heads(raw, args.confidence, args.nms_iou))
            for image_id in range(len(images)):
                truth.append(targets[targets[:, 0] == image_id, 1:].cpu())
            if (batch_id + 1) % 50 == 0:
                print(f'evaluated {len(predictions)}/{len(dataset)} images', flush=True)
    metrics = ap50(predictions, truth, count)
    summary = {
        'evaluated_utc': datetime.now(timezone.utc).isoformat(),
        'weights': str(args.weights.resolve()), 'checkpoint_epoch': checkpoint['epoch'],
        'dataset': str(args.data.resolve()),
        'checkpoint_train_limit': checkpoint.get('train_limit', 0),
        'checkpoint_val_limit': checkpoint.get('val_limit', 0),
        'split': args.split, 'images': len(dataset), 'full_split_images': full_size,
        'side': checkpoint['side'], 'confidence': args.confidence,
        'nms_iou': args.nms_iou, 'metric': '101-point interpolated AP@0.5',
        'map50': metrics['map50'],
        'per_class_ap50': {names[key]: value for key, value in metrics['per_class_ap50'].items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
