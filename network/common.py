import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, \
        bn=False, bias=True,  act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(ConvBlock, self).__init__(*m)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, padding=1, relu=True, bn=False):
        super(DenseBlock, self).__init__()
        if bn:
            self.bn = nn.BatchNorm2d(in_channels)
        if relu:
            self.relu = nn.ReLU(inplace=False)

        self.conv = nn.Conv2d(in_channels, out_channels, k_size, padding=padding, bias=True)

    def forward(self, x):
        if hasattr(self, 'bn'):
            x = self.bn(x)
        if hasattr(self, 'relu'):
            x = self.relu(x)
        x = self.conv(x)
        return x



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, padding=1, relu=True, bn=False):
        super(ResBlock, self).__init__()
        if bn:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, k_size, padding=padding, bias=False),
                                     nn.BatchNorm2d(out_channels))
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, k_size, padding=padding, bias=True)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        self.conv_1x1 = nn.Conv2d(2 * out_channels, out_channels, 1, bias=True)


    def forward(self, x, before_x):
        y = self.conv(x)
        if hasattr(self, 'relu'):
            y = self.relu(y)
        r = 0
        for x in before_x:
            y = y + x
        #r = r / (1 + len(before_x))
        #y = self.conv_1x1(torch.cat([y, r], 1))
        return y
