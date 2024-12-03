import paddle
from paddle import nn
from Model.YOLOPAFPN import YOLOPAFPN
from Model.YOLOXHead import YOLOXHead

class YoloBody(nn.Layer):

    def __init__(self, num_classes, kind):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[kind], width_dict[kind]
        depthwise       = True if kind == 'nano' else False

        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs    = self.backbone.forward(x)
        outputs     = self.head.forward(fpn_outs)
        return outputs

## 测试YOLO Body模块
x = paddle.ones([1, 3, 640, 640])
net4 = YoloBody(20, 'x')
print(net4(x)[0].shape, net4(x)[1].shape, net4(x)[2].shape)