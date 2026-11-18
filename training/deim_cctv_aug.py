"""DEIMv2 transform `CCTVPixelAug`: the shared CCTV pixel augmentations (cctv_aug.py) as a DEIMv2 config op.

Importing this module registers the op with DEIMv2's config system, so it must be imported before the config
is built: `deim_launch.py` does that and then runs the repo's own train.py, which leaves the DEIMv2 clone
unmodified. Use it in a config like any other op:

    - {type: CCTVPixelAug, level: cctv, strength: 1.0, colour_jitter: True}

It acts on the PIL image only; boxes, labels and the dataset handle pass through unchanged.
"""
import numpy as np
import PIL.Image
import torchvision.transforms.v2 as T

from engine.core import register  # DEIMv2's registry; the repo must be on sys.path

import cctv_aug


@register()
class CCTVPixelAug(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, level='cctv', strength=1.0, colour_jitter=True) -> None:
        super().__init__()
        self.level, self.strength, self.colour_jitter = level, strength, colour_jitter
        self._pipe = None   # built lazily inside each dataloader worker (albumentations objects need not pickle)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_pipe'] = None
        return state

    def _get_pipe(self):
        if self._pipe is None:
            import albumentations as A
            ops = cctv_aug.build(self.level, self.strength, self.colour_jitter)
            self._pipe = A.Compose(ops) if ops else False
        return self._pipe

    def _transform(self, inpt, params):
        pipe = self._get_pipe()
        if not pipe:
            return inpt
        mode = inpt.mode
        out = pipe(image=np.asarray(inpt.convert('RGB')))['image']
        img = PIL.Image.fromarray(out)
        return img if mode == 'RGB' else img.convert(mode)

    # torchvision >= 0.21 calls transform(); 0.20 (the DEIMv2 pin) calls _transform()
    def transform(self, inpt, params):
        return self._transform(inpt, params)
