import paddle
import paddle.nn as nn
from Model.Conv import DWConv,BaseConv
from Model.CSPDarknet import CSPDarknet
from Model.CSPLayer import CSPLayer

class YOLOPAFPN(nn.Layer):

    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode='nearest')

        # [-1, 1024, 20, 20] -> [-1, 512, 20, 20]
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)

        # [-1, 1024, 40, 40] -> [-1, 512, 40, 40]
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act
        )

        # [-1, 512, 40, 40] -> [-1, 256, 40, 40]
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        # [-1, 512, 80, 80] -> [-1, 256, 80, 80]
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act
        )

        # Bottom-Up Conv
        # [-1, 256, 80, 80] -> [-1, 256, 40, 40]
        self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        # [-1, 512, 40, 40] -> [-1, 512, 40, 40]
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act
        )

        # [-1, 512, 40, 40] -> [-1, 512, 20, 20]
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        # [-1, 1024, 20, 20] -> [-1, 1024, 20, 20]
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act
        )

    def forward(self, input):
        out_features            = self.backbone(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        # [-1, 1024, 20, 20] -> [-1, 512, 20, 20]
        P5          = self.lateral_conv0(feat3)
        # [-1, 512, 20, 20] -> [-1, 512, 40, 40]
        P5_upsample = self.upsample(P5)
        # [-1, 512, 40, 40] + [-1, 512, 40, 40] -> [-1, 1024, 40, 40]
        P5_upsample = paddle.concat([P5_upsample, feat2], axis=1)
        # [-1, 1024, 40, 40] -> [-1, 512, 40, 40]
        P5_upsample = self.C3_p4(P5_upsample)

        # [-1, 512, 40, 40] -> [-1, 256, 40, 40]
        P4          = self.reduce_conv1(P5_upsample)
        # [-1, 256, 40, 40] -> [-1, 256, 80, 80]
        P4_upsample = self.upsample(P4)
        # [-1, 256, 80, 80] + [-1, 256, 80, 80] -> [-1, 512, 80, 80]
        P4_upsample = paddle.concat([P4_upsample, feat1], axis=1)
        # [-1, 512, 80, 80] -> [-1, 256, 80, 80]
        P3_out      = self.C3_p3(P4_upsample)

        # [-1, 256, 80, 80] -> [-1, 256, 40, 40]
        P3_downsample   = self.bu_conv2(P3_out)
        # [-1, 256, 40, 40] + [-1, 256, 40, 40] -> [-1, 512, 40, 40]
        P3_downsample   = paddle.concat([P3_downsample, P4], axis=1)
        # [-1, 512, 40, 40] -> [-1, 512, 40, 40]
        P4_out          = self.C3_n3(P3_downsample)

        # [-1, 512, 40, 40] -> [-1, 512, 20, 20]
        P4_downsample   = self.bu_conv1(P4_out)
        # [-1, 512, 20, 20] + [-1, 512, 20, 20] -> [-1, 1024, 20, 20]
        P4_downsample   = paddle.concat([P4_downsample, P5], axis=1)
        # [-1, 1024, 20, 20] -> [-1, 1024, 20, 20]
        P5_out          = self.C3_n4(P4_downsample)
        return (P3_out, P4_out, P5_out)

## 测试YOLOPAFPN模块
features = (paddle.ones([1, 256, 80, 80]), paddle.ones([1, 512, 40, 40]),
            paddle.ones([1, 1024, 20, 20]))
net2 = YOLOPAFPN()
