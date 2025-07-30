import torch
from torch import nn


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    :param ch: 输入特征矩阵的channel
    :param divisor: 基数
    :param min_ch: 最小通道数
    """
    if min_ch is None:
        min_ch = divisor
    #   将ch调整到距离8最近的整数倍
    #   int(ch + divisor / 2) // divisor 向上取整
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    #   确保向下取整时不会减少超过10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


#   定义 卷积-BN-ReLU6 联合操作
class ConvBNReLU(nn.Sequential):
    #   PyTorch中DW卷积通过调用 nn.Conv2d() 来实现
    #   参数 (groups=1) 为普通卷积，参数 (groups=输入特征矩阵的深度) 为DW卷积
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


#   倒残差结构
class InvertedResidual(nn.Module):
    #   expand_ratio:扩展因子(t)
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        #   定义隐层，对应第一层的输出通道数 (tk)
        hidden_channel = in_channel * expand_ratio
        #   当stride=1且输入特征矩阵与输出特征矩阵shape相同是才有shortcut
        self.use_shotcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            #   1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            #   3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            #   1x1 pointwise conv(linear)  linear:不添加激活函数就等于线性函数
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shotcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    #   alpha:用来控制卷积层中所使用卷积核个数的参数
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        #   初始化倒残差模块
        block =InvertedResidual
        #   通过_make_divisible将卷积核个数调整为8的整数倍
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        #   创建参数列表
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        features.append(ConvBNReLU(3, input_channel, stride=2))
        #   定义一系列block结构
        for t, c, n, s in inverted_residual_setting:
            #   调整输出通道数
            output_channel = _make_divisible(c * alpha, round_nearest)
            #   重复倒残差结构
            #   第一层：stride=n  其它层：stride=1
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        #   定义最后一个卷积层
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        #   特征提取层
        self.features = nn.Sequential(*features)

        #   分类器部分
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        #   初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x
