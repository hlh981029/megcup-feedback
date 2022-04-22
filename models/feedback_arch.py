import math
import megengine.module as nn
from megengine.module import GELU
from .blocks.fuse import PDConvFuse
from .blocks.trans_block import BigKernelRestormerBlock
from utils import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class FeedbackModel(nn.Module):
    def __init__(self,
                 f_number=32,
                 num_layers=8,
                 act_layer=GELU,
                 skip_connect=PDConvFuse,
                 trans_block=BigKernelRestormerBlock):
        super().__init__()

        self.num_layers = num_layers

        self.pre_conv = nn.Sequential(
            nn.Conv2d(4, f_number, 5, padding=2),
            nn.Conv2d(f_number, f_number, 5, padding=2, groups=f_number),
            nn.Conv2d(f_number, f_number, 1, padding=0),
        )

        self.trans_blocks = [
            trans_block() for _ in range(self.num_layers)
        ]

        self.skip_connect_blocks = [
            skip_connect(f_number, act_layer, feature_num=2)
            for _ in range(math.ceil(self.num_layers / 2) - 1)
        ]

        self.post_conv = nn.Sequential(
            nn.Conv2d(f_number, f_number, 5, padding=2, groups=f_number),
            nn.Conv2d(f_number, f_number, 1, padding=0),
            nn.Conv2d(f_number, 4, 5, padding=2),
        )

    def forward(self, x):
        shortcut = x

        x = self.pre_conv(x)

        skip_features = []
        idx_skip = 1
        for idx, b in enumerate(self.trans_blocks):
            if idx > math.floor(self.num_layers / 2):
                x = self.skip_connect_blocks[idx_skip - 1](x, skip_features[-idx_skip])
                idx_skip += 1

            x = b(x)

            if idx < (math.ceil(self.num_layers / 2) - 1):
                skip_features.append(x)

        x = self.post_conv(x)

        return x + shortcut
