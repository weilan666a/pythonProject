import paddle
from paddle import nn
from Model.Conv import BaseConv

## Focus层
class Focus(nn.Layer):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # 分别获得4个2倍下采样结果
        patch_1 = x[...,  ::2,  ::2]
        patch_2 = x[..., 1::2,  ::2]
        patch_3 = x[...,  ::2, 1::2]
        patch_4 = x[..., 1::2, 1::2]
        # 沿通道方向拼接4个下采样结果
        x = paddle.concat((patch_1, patch_2, patch_3, patch_4), axis=1)
        # 拼接结果做卷积
        out = self.conv(x)
        return out

## 测试FOCUS模块
x = paddle.ones([1, 3, 640, 640])
layer = Focus(3, 64)
print(layer(x).shape)