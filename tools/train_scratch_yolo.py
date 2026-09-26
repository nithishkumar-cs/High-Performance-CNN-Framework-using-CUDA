#!/usr/bin/env python3
'''Train the independent tiny detector from random weights on the fixed VOC split.'''

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scratch_yolo.data import VocYoloDataset, collate_detection
from scratch_yolo.model import TinyDetector
from scratch_yolo.objective import DetectionLoss


def run_epoch(model, loader, objective, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            loss = objective(model(images), targets)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * len(images)
    return total / len(loader.dataset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/voc2007/yolo'))
    parser.add_argument('--output', type=Path, default=Path('runs/scratch_yolo'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--side', type=int, default=416)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-train', type=int, default=0, help='smoke test only')
    parser.add_argument('--limit-val', type=int, default=0, help='smoke test only')
    parser.add_argument('--resume', action='store_true', help='continue from output/last.pt')
    parser.add_argument('--save-every', type=int, default=5,
                        help='archive a checkpoint every N epochs for AP selection')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.side < 32 or args.side % 32:
        parser.error('epochs/batch must be positive; side must be a multiple of 32')
    if (args.workers < 0 or args.learning_rate <= 0 or args.limit_train < 0
            or args.limit_val < 0 or args.save_every < 1):
        parser.error('invalid workers, learning rate, or limits')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = VocYoloDataset(args.data, 'train', args.side)
    val = VocYoloDataset(args.data, 'val', args.side)
    if args.limit_train:
        train = Subset(train, range(min(args.limit_train, len(train))))
    if args.limit_val:
        val = Subset(val, range(min(args.limit_val, len(val))))
    shuffle_rng = torch.Generator()
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_detection,
                              generator=shuffle_rng)
    val_loader = DataLoader(val, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_detection)
    model = TinyDetector().to(device)
    objective = DetectionLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                                weight_decay=0.0005)
    args.output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    history = []
    best = float('inf')
    first_epoch = 1
    if args.resume:
        path = args.output / 'last.pt'
        if not path.is_file():
            parser.error(f'no checkpoint to resume: {path}')
        saved = torch.load(path, map_location=device, weights_only=True)
        expected = {'side': args.side, 'seed': args.seed, 'batch': args.batch,
                    'learning_rate': args.learning_rate,
                    'train_limit': args.limit_train, 'val_limit': args.limit_val,
                    'dataset': str(args.data.resolve())}
        for key, value in expected.items():
            if saved.get(key) != value:
                parser.error(f'resume mismatch for {key}: checkpoint={saved.get(key)!r}, requested={value!r}')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        first_epoch = int(saved['epoch']) + 1
        started = saved['started_utc']
        history = saved['history']
        best = saved['best_val_loss']
    if first_epoch > args.epochs:
        parser.error(f'checkpoint already reached epoch {first_epoch - 1}')
    for epoch in range(first_epoch, args.epochs + 1):
        shuffle_rng.manual_seed(args.seed + epoch)
        train_loss = run_epoch(model, train_loader, objective, optimizer, device)
        val_loss = run_epoch(model, val_loader, objective, None, device)
        record = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        history.append(record)
        improved = val_loss < best
        best = min(best, val_loss)
        checkpoint = {
            'model': model.state_dict(), 'epoch': epoch, 'classes': 20,
            'side': args.side, 'seed': args.seed, 'optimizer': optimizer.state_dict(),
            'train_limit': args.limit_train, 'val_limit': args.limit_val,
            'batch': args.batch, 'learning_rate': args.learning_rate,
            'dataset': str(args.data.resolve()),
            'started_utc': started, 'history': history, 'best_val_loss': best,
        }
        if improved:
            torch.save(checkpoint, args.output / 'best.pt.tmp')
            (args.output / 'best.pt.tmp').replace(args.output / 'best.pt')
        torch.save(checkpoint, args.output / 'last.pt.tmp')
        (args.output / 'last.pt.tmp').replace(args.output / 'last.pt')
        if epoch % args.save_every == 0 or epoch == args.epochs:
            archive = args.output / 'epochs'
            archive.mkdir(exist_ok=True)
            torch.save(checkpoint, archive / f'epoch_{epoch:03d}.pt.tmp')
            (archive / f'epoch_{epoch:03d}.pt.tmp').replace(
                archive / f'epoch_{epoch:03d}.pt')
        summary = {
            'started_utc': started, 'updated_utc': datetime.now(timezone.utc).isoformat(),
            'model': 'scratch_yolov3_tiny', 'pretrained': False, 'device': str(device),
            'dataset': str(args.data.resolve()), 'epochs_requested': args.epochs,
            'train_images': len(train), 'val_images': len(val), 'side': args.side,
            'batch': args.batch, 'seed': args.seed, 'history': history,
            'best_val_loss': best,
        }
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(record), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
