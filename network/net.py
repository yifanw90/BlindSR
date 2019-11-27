import torch
from torch import nn
import torch.nn.functional as F
from core.config import config
import common


class ClsBranch(nn.Module):
    def __init__(self, bn=True, n_colors=1, conv=common.ConvBlock):
        super(ClsBranch, self).__init__()

        act = nn.LeakyReLU(0.1, True)
        n_feat = 32 
        kernel_size = 3
        ker_num = 29
        n_conv = 3
        # define head module
        m_head = [conv(n_colors, n_feat, kernel_size, stride=2, bn=bn, bias=not bn, act=act)]
        m_body = [conv(n_feat*(2**i), 2*n_feat*(2**i), kernel_size, stride=2, bn=bn, bias=not bn, act=act) 
                  for i in range(n_conv)]

        m_tail = [nn.AdaptiveAvgPool2d((1, 1)),
                  conv(n_feat*(2**n_conv), 4, kernel_size=1, bias=True, bn=False, act=None),
                 ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.norm = nn.Softmax(1) 
        
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        out_w = self.tail(x)
        out_w = self.norm(out_w)
        return out_w


class SRBranch(nn.Module):
    def __init__(self, factor=4, num_blocks=8, out_channels=64, relu=True, bn=False):
        super(SRBranch, self).__init__()
        self.factor = factor
        self.dense_block = nn.ModuleList()
        in_channels = 1
        self.head = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.dense_block.append(nn.ReLU())
        
        for i in range(num_blocks):
            conv = nn.Sequential(nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True))
            self.dense_block.append(conv)
        self.dense_block.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True))
        self.upsample = nn.ConvTranspose2d(out_channels, 8, factor+2, stride=factor, padding=1, bias=True)
        self.conv3 = nn.Conv2d(8, 1, 3, padding=1, bias=True)
    def forward(self, x):
        x_in = self.head(x)
        x = x_in
        for block in self.dense_block:
            x_out = block(x)
            x = x_out + x
        x = self.upsample(x+x_in)
        x = self.conv3(x)
        return x


class Net(nn.Module):
    def __init__(self, factor=4, n_colors=1, input_range=255.0):
        super(Net, self).__init__()
        self.input_range = input_range
        self.sr_branch1 = SRBranch(factor, 7)
        self.sr_branch2 = SRBranch(factor, 7)
        self.sr_branch3 = SRBranch(factor, 7)
        self.sr_branch4 = SRBranch(factor, 7)
        self.cls_branch = ClsBranch(True, n_colors)
        self.factor = factor
        self.initialize()

    def forward(self, x, x_bi):
        x_input = x 
        sr1 = self.sr_branch1(x_input)
        sr2 = self.sr_branch2(x_input)
        sr3 = self.sr_branch3(x_input)
        sr4 = self.sr_branch4(x_input)
        cls_w = self.cls_branch(x_input)
        sr = torch.cat([sr1, sr2, sr3, sr4], 1)

        out = torch.sum(cls_w*sr, 1, keepdim=True) + x_bi
        return out
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()
