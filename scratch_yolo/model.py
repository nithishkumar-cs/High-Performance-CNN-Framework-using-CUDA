'''Clean-room YOLOv3-tiny-inspired network; no Darknet source or weights.'''

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )


class TinyDetector(nn.Module):
    '''Two raw anchor heads, returned in coarse-then-fine order.

    Each output is [batch, 3, 5 + classes, grid_y, grid_x]. Inputs must be
    square RGB tensors with a side divisible by 32. Outputs are raw logits.
    '''

    def __init__(self, classes: int = 20):
        super().__init__()
        if classes < 1:
            raise ValueError('classes must be positive')
        self.classes = classes
        channels = (3, 16, 32, 64, 128, 256)
        self.stages = nn.ModuleList(
            ConvBlock(channels[i], channels[i + 1], 3) for i in range(5)
        )
        self.deep = nn.Sequential(
            ConvBlock(256, 512, 3),
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 256, 1),
            ConvBlock(256, 512, 3),
        )
        self.coarse_head = nn.Conv2d(512, 3 * (5 + classes), 1)
        self.fine_reduce = ConvBlock(256, 128, 1)
        self.fine_fuse = ConvBlock(384, 256, 3)
        self.fine_head = nn.Conv2d(256, 3 * (5 + classes), 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] != x.shape[3] or x.shape[2] % 32:
            raise ValueError('expected [batch, 3, side, side] with side divisible by 32')
        route = None
        for index, stage in enumerate(self.stages):
            features = stage(x)
            if index == 4:
                route = features
            x = F.max_pool2d(features, 2)
        # Match the asymmetric right/bottom padding of the stride-one pool.
        x = self.deep[0](x)
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=float('-inf')), 2, stride=1)
        x = self.deep[1](x)
        branch = self.deep[2](x)
        coarse = self.coarse_head(self.deep[3](branch))
        up = F.interpolate(self.fine_reduce(branch), scale_factor=2, mode='nearest')
        fine = self.fine_head(self.fine_fuse(torch.cat((up, route), dim=1)))
        batch, _, height, width = coarse.shape
        coarse = coarse.reshape(batch, 3, 5 + self.classes, height, width)
        batch, _, height, width = fine.shape
        fine = fine.reshape(batch, 3, 5 + self.classes, height, width)
        return coarse, fine
