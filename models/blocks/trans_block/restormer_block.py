import megengine as mge
from megengine import module as nn
from megengine import functional as F
from ..layernorm import ChannelLayerNorm


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, skip=False):
        super(FeedForward, self).__init__()
        self.skip = skip

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        ori_x = x
        x = self.project_in(x)
        x1, x2 = F.split(self.dwconv(x), 2, axis=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        if self.skip:
            x += ori_x
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = mge.Parameter(F.ones((num_heads, 1, 1)))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = F.split(qkv, 3, axis=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, -1)
        k = k.reshape(b, self.num_heads, c // self.num_heads, -1)
        v = v.reshape(b, self.num_heads, c // self.num_heads, -1)

        q = F.normalize(q, axis=-1)
        k = F.normalize(k, axis=-1)

        attn = F.matmul(q, k.transpose(0, 1, 3, 2)) * self.temperature
        attn = self.softmax(attn)
        # save_attn(attn)
        out = F.matmul(attn, v)

        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)
        return out


class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, norm_layer, bias=False, skip=False):
        super(RestormerBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = norm_layer(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, skip)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class BigKernelRestormerBlock(RestormerBlock):
    def __init__(self, dim=32, num_heads=4, ffn_expansion_factor=1.2, norm_layer=ChannelLayerNorm, bias=False,
                 dilated_rate=4, kernel_size=5):
        super().__init__(dim, num_heads, ffn_expansion_factor, norm_layer, bias)
        padding = (kernel_size * dilated_rate - dilated_rate) // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, padding, dilated_rate, dim)

    def forward(self, x):
        return self.conv(super().forward(x))
