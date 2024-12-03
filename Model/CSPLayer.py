import paddle
from paddle import nn
from Model.Conv import BaseConv, Bottleneck


## CSPLayer
class CSPLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # 主干部分的基本卷积块
        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # 残差边部分的基本卷积块
        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # 拼接主干与残差后的基本卷积块
        self.conv3  = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        # 根据循环次数构建多个残差块瓶颈结构
        res_block = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        # 主干部分
        x_main = self.conv1(x)
        x_main = self.res_block(x_main)
        # 残差边部分
        x_res = self.conv2(x)

        # 主干部分和残差边部分进行堆叠
        x = paddle.concat((x_main, x_res), axis=1)

        # 对堆叠的结果进行卷积的处理
        out = self.conv3(x)
        return out

## 测试CSPLayer模块
x = paddle.ones([1, 3, 640, 640])
layer = CSPLayer(3, 64, 5)
print(layer(x).shape)