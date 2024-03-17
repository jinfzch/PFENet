# ppm
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPM(nn.Module):
    def __init__(self, in_dim=256, reduction_dim, bins): # bins 池化的区域大小 列表
        super(PPM, self).__init__()
        self.features = [] # 存储池化之后的特征
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin), # 自适应平均池化，将输入大小调整为池化区域大小
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False), # 1x1卷积，降维
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.down_merge = nn.Conv2d(in_dim*2, in_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x_size = x.size()
        out = [x] # 初始化输出张量列表，并添加输入张量
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)) # 升维
        return self.down_merge(torch.cat(out, 1)) # 按通道维度拼接