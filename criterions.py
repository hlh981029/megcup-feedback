import megengine as mge
import megengine.module as M
import megengine.functional as F
import numpy as np
from abc import abstractmethod


class BaseCriterion(M.Module):
    def __init__(self, keys) -> None:
        super().__init__()
        self.keys = keys
        self.register_forward_hook(
            lambda _, __, outs: {k: v for k, v in zip(self.keys, outs)}
        )
        self.output_length = len(keys)

    def __len__(self):
        return len(self.keys)

    @abstractmethod
    def forward(self, x, y):
        pass

class NumpyScores(BaseCriterion):
    def __init__(self):
        super().__init__(['score', 'diff'])

    def unpack_raw(self, packed_raw, max_value=65535.0, min_clip=0.0, max_clip=65535.0):
        n, c, h, w = packed_raw.shape
        raw = packed_raw.reshape((n, 1, 2, 2, h, w)).transpose((0, 1, 4, 2, 5, 3)).reshape((n, h * 2, w * 2))
        raw = F.round(F.clip(raw * max_value, min_clip, max_clip))
        return raw

    def forward(self, x, y):
        n, c, h, w = x.shape
        assert c == 4
        # print(x.shape, y.shape, x.min(), y.min(), x.max(), y.max())
        x = self.unpack_raw(x).numpy()
        y = self.unpack_raw(y).numpy()
        # print(x.shape, y.shape, x.min(), y.min(), x.max(), y.max())
        means = y.mean(axis=(1, 2))
        weight = (1 / means) ** 0.5
        diff = np.abs(x - y).mean(axis=(1, 2))
        diff = diff * weight
        diff = diff.mean()
        score = np.log10(100 / diff) * 5
        # print(score, diff)
        return score, diff
