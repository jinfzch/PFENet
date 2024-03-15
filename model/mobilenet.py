from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

# 将传入的channel调整到最近的8的整数倍。提高硬件计算效率【不管】
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 整体结构 = 卷积 + BN + 激活函数
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int, # 输入特征矩阵的channel
                 out_planes: int, # 输出特征矩阵的channel = 卷积核个数
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, # 卷积之后的bn层
                 activation_layer: Optional[Callable[..., nn.Module]] = None): # 激活函数
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # 默认bn
        if activation_layer is None:
            activation_layer = nn.ReLU6 # 默认relu6

        # 传入需要依次搭建的几个结构
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes, # 卷积层的一系列参数
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False), # 后续会使用到bn层，所以不使用偏置
                                               norm_layer(out_planes), # BN层
                                               activation_layer(inplace=True)) # 激活函数


# 注意力机制模块 = 2 * 全连接层
# 第一个全连接层，节点个数 = 输入特征矩阵channel / 4 ;  第二个全连接层，节点个数 = 输入特征矩阵channel
# 激活函数： relu / hard-sigmoid
class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4): # squeeze_factor, 挤压稀疏，将channel挤压为原先的1/4
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1) # 直接使用卷积作为全连接层，作用相同。1：卷积核大小 1*1
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor: # x 输入特征矩阵
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1)) # 自适应平均池化，将每个channel上的数据，平均池化到1*1的大小
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


# 这个config文件，对应的是MobileNetv3中每个bneck结构的参数配置（每一层的参数）
class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,
                 kernel: int, # deepwishConv对应的卷积核的大小
                 expanded_c: int, # 对应第一个1*1卷积层使用的卷积核的个数
                 out_c: int, # 最后一个1*1卷积层输出得到的特征矩阵的channel
                 use_se: bool, # 是否使用se模块
                 activation: str, # RE（relu） / HS (hard-swish)
                 stride: int, # deepwishConv对应的步距
                 width_multi: float): # alpha参数：调节每个卷积层使用channel的倍率因子
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float): # alpha倍率因子
        return _make_divisible(channels * width_multi, 8)


# 倒残差结构 = 1*1卷积层 + deepwishConv + SE + 1*1卷积层
# bneck，瓶颈结构
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]: # stride只能为1/2
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c) # 是否使用short_cut连接（resnet）

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand:第一个1*1卷积层（升维卷积层），用于将输入特征矩阵升维
        if cnf.expanded_c != cnf.input_c: #（第一个bneck的输入与输出channel相同，所有没有升维）
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwiseConv
        layers.append(ConvBNActivation(cnf.expanded_c, # 上一层输出的特征矩阵的channel = expand channel
                                       cnf.expanded_c, # 输入与输出channel相同
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c, # group数与channel保持一致
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project：最后一个1*1卷积层（降维卷积层）
        layers.append(ConvBNActivation(cnf.expanded_c, # 输入 = 上一层输出
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                        activation_layer=nn.Identity)) # 线性激活 y=x，无需操作

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        # self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x) # 代码2
        if self.use_res_connect:
            result += x

        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig], # 一系列层结构的参数
                 last_channel: int, # 倒数第二个卷积层 / 全连接层 的输出节点个数（1280）
                 num_classes: int = 1000, # 分类类别
                 block: Optional[Callable[..., nn.Module]] = None, # bneck结构，默认为空
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        # 数据检查
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01) # 设置bn层的默认参数

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3, # RGB
                                       firstconv_output_c, # 第一个bneck结构的input——channel
                                       kernel_size=3, # 卷积核大小 3*3
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks【遍历bneck结构】
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        # 最后一个bneck结构
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # mobileNetV3网络用于提取特征的主干部分
        self.features = nn.Sequential(*layers)
        # --------------------------------------------------------------------------------------------------------------
        # 自适应全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平处理，舍弃高宽维度
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    预训练权重 The MobileNetV3-large
    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0 # alpha参数
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi) # partial   给方法传入一个默认参数
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1 # 官方超参数，对于最后三个bneck卷积参数的调整。默认为false不使用

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1 2
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2 4
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3 7
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1), # C5
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)

def mobilenet_v3_large_use(pretrained=True):
    model = mobilenet_v3_large()

    # mobilenet在coco上的预训练权重

    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_path = './initmodel/mobilenet_v3_large-8738ca79.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    return model