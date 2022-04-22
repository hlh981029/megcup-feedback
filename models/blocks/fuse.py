import megengine as mge
from megengine import module as nn
from megengine import functional as F
import numpy as np
from megengine.module import GELU


class PDConvFuse(nn.Module):
    def __init__(self, in_channels, act_layer, feature_num=2, bias=True, **kwargs) -> None:
        super().__init__()
        self.feature_num = feature_num
        self.act_layer = act_layer()

        self.pwconv = nn.Conv2d(feature_num * in_channels, in_channels, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias, groups=in_channels)

    def forward(self, *inp_feats):
        assert len(inp_feats) == self.feature_num
        return self.dwconv(self.act_layer(self.pwconv(F.concat(inp_feats, axis=1))))
