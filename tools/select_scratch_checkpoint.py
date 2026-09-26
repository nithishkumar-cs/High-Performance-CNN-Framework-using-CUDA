#!/usr/bin/env python3
'''Select a scratch checkpoint using full validation AP, never test AP.'''

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluations', nargs='+', type=Path,
                        help='full-validation JSON files from evaluate_scratch_yolo')
    parser.add_argument('--output', type=Path, default=Path('runs/scratch_yolo/selected.pt'))
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding='utf-8')) for path in args.evaluations]
    common = None
    for record in records:
        if record['split'] != 'val' or record['images'] != record['full_split_images']:
            parser.error('selection requires every evaluation to cover the full validation split')
        if record['checkpoint_train_limit'] or record['checkpoint_val_limit']:
            parser.error('smoke-test checkpoints cannot be selected')
        settings = (record['dataset'], record['side'], record['confidence'], record['nms_iou'])
        if common is None:
            common = settings
        elif settings != common:
            parser.error('all evaluations must use the same data and inference settings')
        checkpoint = torch.load(record['weights'], map_location='cpu', weights_only=True)
        if checkpoint['epoch'] != record['checkpoint_epoch']:
            parser.error(f'checkpoint changed since evaluation: {record["weights"]}')
    winner = max(records, key=lambda record: record['map50'])
    source = Path(winner['weights'])
    if source.resolve() == args.output.resolve():
        parser.error('selected output must differ from the source checkpoint')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    shutil.copy2(source, temporary)
    temporary.replace(args.output)
    summary = {
        'selected_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(args.output.resolve()), 'selected_epoch': winner['checkpoint_epoch'],
        'validation_map50': winner['map50'],
        'evaluation_files': [str(path.resolve()) for path in args.evaluations],
        'candidates': [{'epoch': record['checkpoint_epoch'], 'map50': record['map50']}
                       for record in records],
        'settings': {'dataset': common[0], 'side': common[1],
                     'confidence': common[2], 'nms_iou': common[3]},
    }
    record_path = args.output.with_suffix('.selection.json')
    record_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
