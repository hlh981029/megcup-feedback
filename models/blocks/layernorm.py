import megengine as mge
from megengine import module as M
from megengine import functional as F
import numpy as np


class ChannelLayerNorm(M.Module):
    def __init__(self, channels, has_bias=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.has_bias = has_bias
        self.weight = mge.Parameter(np.ones((1, channels, 1, 1), dtype='float32'))
        if has_bias:
            self.bias = mge.Parameter(np.zeros((1, channels, 1, 1), dtype='float32'))
        else:
            self.bias = 0

    def forward(self, x):
        mu = F.mean(x, axis=1, keepdims=True) if self.has_bias else 0
        sigma = F.var(x, axis=1, keepdims=True)
        return (x - mu) / F.sqrt(sigma + self.eps) * self.weight + self.bias
