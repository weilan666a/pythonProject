import paddle
import paddle.nn as nn
from Model.Conv import DWConv,BaseConv

class YOLOXHead(nn.Layer):

    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv

        self.cls_convs  = []
        self.reg_convs  = []
        self.cls_preds  = []
        self.reg_preds  = []
        self.obj_preds  = []
        self.stems      = []

        for i in range(len(in_channels)):
            # 预处理卷积: 1个1x1卷积
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), kernel_size = 1, stride = 1, act = act))
            # 分类特征提取: 2个3x3卷积
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), kernel_size= 3, stride = 1, act = act),
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), kernel_size= 3, stride = 1, act = act),
            ]))
            # 分类预测: 1个1x1卷积
            self.cls_preds.append(
                nn.Conv2D(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )

            # 回归特征提取: 2个3x3卷积
            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), kernel_size = 3, stride = 1, act = act),
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), kernel_size = 3, stride = 1, act = act)
            ]))
            # 回归预测(位置): 1个1x1卷积
            self.reg_preds.append(
                nn.Conv2D(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            # 回归预测(是否含有物体): 1个1x1卷积
            self.obj_preds.append(
                nn.Conv2D(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        # 输入[P3_out, P4_out, P5_out]
        # P3_out: [-1, 256, 80, 80]
        # P4_out: [-1, 512, 40, 40]
        # P5_out: [-1, 1024, 20, 20]
        outputs = []
        for k, x in enumerate(inputs):
            # 1x1卷积通道整合
            x           = self.stems[k](x)

            # 2个3x3卷积特征提取
            cls_feat    = self.cls_convs[k](x)
            # 1个1x1卷积预测类别
            # 分别输出: [-1, num_classes, 80, 80], [-1, num_classes, 40, 40], [-1, num_classes, 20, 20]
            cls_output  = self.cls_preds[k](cls_feat)

            # 2个3x3卷积特征提取
            reg_feat    = self.reg_convs[k](x)
            # 1个1x1卷积预测位置
            # 分别输出: [-1, 4, 80, 80], [-1, 4, 40, 40], [-1, 4, 20, 20]
            reg_output  = self.reg_preds[k](reg_feat)
            # 1个1x1卷积预测是否有物体
            # 分别输出: [-1, 1, 80, 80], [-1, 1, 40, 40], [-1, 1, 20, 20]
            obj_output  = self.obj_preds[k](reg_feat)

            # 整合结果
            # 输出: [-1, num_classes+5, 80, 80], [-1, num_classes+5, 40, 40], [-1, num_classes+5, 20, 20]
            output      = paddle.concat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

## 测试YOLOX Head模块
features = paddle.ones([1, 256, 80, 80]), paddle.ones([1, 512, 40, 40]), paddle.ones([1, 1024, 20, 20])
net3 = YOLOXHead(10)
print(net3(features)[0].shape, net3(features)[1].shape, net3(features)[2].shape)