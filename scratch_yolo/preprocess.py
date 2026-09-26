'''Shared RGB letterbox preprocessing for training and deployment.'''

import numpy as np
from PIL import Image


def letterbox(image: Image.Image, side: int) -> tuple[np.ndarray, float, int, int]:
    if side < 32 or side % 32:
        raise ValueError('side must be divisible by 32')
    rgb = image.convert('RGB')
    width, height = rgb.size
    scale = min(side / width, side / height)
    new_w, new_h = round(width * scale), round(height * scale)
    left, top = (side - new_w) // 2, (side - new_h) // 2
    canvas = Image.new('RGB', (side, side), (114, 114, 114))
    canvas.paste(rgb.resize((new_w, new_h), Image.Resampling.BILINEAR), (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    return np.ascontiguousarray(tensor), scale, left, top
