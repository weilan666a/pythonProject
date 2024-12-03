# 引入库
import paddle
from paddle import nn

## 构建卷积块
class BaseConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, act='silu'):
        super().__init__()
        padding = (kernel_size-1)//2
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2D(out_channels,momentum=0.03, epsilon=0.001)
        if act == 'silu':
            self.act = nn.Silu()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

## 构建深度可分离卷积
class DWConv(nn.Layer):
    # Some Problem
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act='silu'):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, kernel_size, stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, 1, 1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

## 构建残差结构
class Bottleneck(nn.Layer):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        # 1x1 卷积进行通道数的缩减(缩减率默认50%)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # 3x3 卷积进行通道数的拓张(特征提取)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

## 测试卷积模块
x = paddle.ones([1, 3, 640, 640])
conv1 = BaseConv(3, 64, 3, 1)
conv2 = DWConv(3, 64, 3, 1)
block1 = Bottleneck(3, 64)
print(conv1(x).shape)
print(conv2(x).shape)
print(block1(x).shape)
