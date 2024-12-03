import paddle
from paddle import nn
from Model.Conv import BaseConv

## SPPBottleneck
class SPPBottleneck(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1      = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.pool_block = nn.Sequential(*[nn.MaxPool2D(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2      = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = paddle.concat([x] + [pool(x) for pool in self.pool_block], axis=1)
        x = self.conv2(x)
        return x

## 测试SPPBottleneck模块
x = paddle.ones([1, 3, 640, 640])
layer = SPPBottleneck(3, 64)
print(layer(x).shape)