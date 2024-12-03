import paddle
from paddle import nn
from Model.Conv import BaseConv,DWConv
from Model.Focus import Focus
from Model.CSPLayer import CSPLayer
from Model.SPPBottleneck import SPPBottleneck


## CSPDarknet
class CSPDarknet(nn.Layer):

    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu",):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        # Image Size : [3, 640, 640]
        base_channels   = int(wid_mul * 64)  # 64
        base_depth      = max(round(dep_mul * 3), 1)  # 3

        # 利用focus网络特征提取
        # [-1, 3, 640, 640] -> [-1, 64, 320, 320]
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # Resblock1[dark2]
        # [-1, 64, 320, 320] -> [-1, 128, 160, 160]
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        # Resblock2[dark3]
        # [-1, 128, 160, 160] -> [-1, 256, 80, 80]
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # Resblock3[dark4]
        # [-1, 256, 80, 80] -> [-1, 512, 40, 40]
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # Resblock4[dark5]
        # [-1, 512, 40, 40] -> [-1, 1024, 20, 20]
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        # dark3输出特征层：[256, 80, 80]
        x = self.dark3(x)
        outputs["dark3"] = x
        # dark4输出特征层：[512, 40, 40]
        x = self.dark4(x)
        outputs["dark4"] = x
        # dark5输出特征层：[1024, 20, 20]
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


## 测试CSPDarknet模块
x = paddle.ones([1, 3, 640, 640])
net1 = CSPDarknet(1, 1)
print(net1(x)['dark3'].shape, net1(x)['dark4'].shape, net1(x)['dark5'].shape)