# SE-block
import torch
from torch import nn
 
 
class SE(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE, self).__init__()

        # Squeeze操作：使用全局平均池化压缩空间维
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation操作：使用两个全连接层来建模通道间的依赖性  
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的 【有问题】
        return x * y.expand_as(x) # 注意力作用每一个通道上