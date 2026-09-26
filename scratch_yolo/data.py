'''VOC-to-tensor loader using the existing, fixed YOLO-label splits.'''

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocess import letterbox


class VocYoloDataset(Dataset):
    def __init__(self, root: Path, split: str, side: int = 416):
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, or test')
        if side < 32 or side % 32:
            raise ValueError('side must be divisible by 32')
        self.root = Path(root)
        self.split = split
        self.side = side
        self.images = sorted((self.root / 'images' / split).glob('*.jpg'))
        if not self.images:
            raise FileNotFoundError(f'no JPEG images in {self.root / "images" / split}')

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.images[index]
        with Image.open(path) as source:
            old_w, old_h = source.size
            tensor, scale, left, top = letterbox(source, self.side)
        pixels = torch.from_numpy(tensor[0])

        label_path = self.root / 'labels' / self.split / f'{path.stem}.txt'
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        rows = []
        for line in label_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            label, cx, cy, width, height = map(float, line.split())
            rows.append((
                label,
                (cx * old_w * scale + left) / self.side,
                (cy * old_h * scale + top) / self.side,
                width * old_w * scale / self.side,
                height * old_h * scale / self.side,
            ))
        labels = torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)
        return pixels, labels


def collate_detection(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([item[0] for item in batch])
    targets = []
    for index, (_, labels) in enumerate(batch):
        if labels.numel():
            column = torch.full((len(labels), 1), index, dtype=torch.float32)
            targets.append(torch.cat((column, labels), dim=1))
    return images, torch.cat(targets) if targets else torch.empty((0, 6), dtype=torch.float32)
