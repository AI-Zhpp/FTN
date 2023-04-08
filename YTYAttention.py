import numpy as np
import torch
from torch import nn
from torch.nn import init
import math

import math


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_)  # , (1, k), (1, s))
        self.cv2 = Conv(c_, c2)  # , (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ppattention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.conv = CrossConv(2 * in_planes, in_planes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出最后两维1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # b,h,w,c   ---  n*c   #n,,c,h,w   ---- n*c
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.SiLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.bnnorm = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.conv(x)  # 2-->1,CROSS CONV
        res = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.sigmoid(out)
        result = x * attn + res
        return result


class ppattention_wan(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.conv = CrossConv(2 * in_planes, in_planes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出最后两维1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # b,h,w,c   ---  n*c   #n,,c,h,w   ---- n*c
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.SiLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.fc2 = nn.Conv2d(1, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bnnorm = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.conv(x)  # 2-->1,CROSS CONV
        res = x
        avg_out = self.fc(self.avg_pool(x))
        # max_out = self.fc(self.max_pool(x))
        # out = avg_out + max_out
        attn = self.sigmoid(avg_out)
        x_channel_summation = torch.sum(x, dim=1, keepdim=True)
        attn_channel_summation = self.sigmoid(self.fc2(x_channel_summation))
        result = x * attn + res + attn_channel_summation * res
        return result


class DFE(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.BatchNorm2d(in_planes // 2),
                                nn.SiLU())

    def forward(self, x):
        result = self.fc(x)
        return result


class YTYAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, im_channel=49):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(im_channel, 1, bias=False)
        self.fc2 = nn.Linear(im_channel, 1, bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, im1, im2):
        # b, c, _, _ = x.size()
        img = torch.cat([im1, im2], dim=2)
        origin = img
        im1 = im1.transpose(1, 2)
        im2 = im2.transpose(1, 2)
        im1 = self.fc1(im1)  # 1,512,1
        im2 = self.fc2(im2)  # 1,512,1

        im = torch.cat([im1, im2], dim=2)  # 1,512,2
        im = torch.transpose(im, 1, 2)  # 1,2,512
        im = self.fc(im)  # 1,2,512
        im1 = im[:, 0, :].unsqueeze(1)
        im2 = im[:, 1, :].unsqueeze(1)
        im = torch.cat([im1, im2], dim=2)
        im = im.transpose(1, 2)
        im = self.sigmoid(im)
        img = img.transpose(1, 2)
        res = img * im.expand_as(img)
        res = res.transpose(1, 2)
        return res


class TYAttention(nn.Module):
    def __init__(self, in_channel=1024, im_channel=49, gamma=2, b=1):
        super().__init__()
        self.fc1 = nn.Linear(im_channel, 1, bias=False)
        self.fc2 = nn.Linear(im_channel, 1, bias=False)
        self.SiLU = nn.SiLU()
        self.bn = nn.BatchNorm1d(2)
        self.gamma = gamma
        self.b = b
        self.sigmoid = nn.Sigmoid()
        self.in_channel = in_channel
        t = int(abs((math.log(self.in_channel, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(2, 2, kernel_size=k, padding=int(k / 2), bias=False)
        self.conv2 = nn.Conv1d(2, 2, kernel_size=k, padding=int(k / 2), bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, im1, im2):
        img = torch.cat([im1, im1], dim=2)  # B,49,2048
        img = img.transpose(1, 2)
        origin = img
        im1 = im1.transpose(-1, -2)  # B,1024,49
        im2 = im2.transpose(-1, -2)

        im1 = self.fc1(im1)  # B,1024,1
        im2 = self.fc2(im2)
        im = torch.cat([im1, im2], dim=2)  # B,1024,2
        im = self.conv(im.transpose(-1, -2))  # B,2,1024
        im = self.SiLU(im)
        # im = self.bn(im)
        im = self.conv2(im)
        im1 = im[:, 0, :].unsqueeze(1)
        im2 = im[:, 1, :].unsqueeze(1)
        im = torch.cat([im1, im2], dim=2)  # B,1,2048
        im = im.transpose(1, 2)
        im = self.sigmoid(im)
        res = img * im.expand_as(img) + origin
        return res.transpose(1, 2)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv2d(3 * in_planes, in_planes, 1, 1, 0)
        self.bn = nn.BatchNorm2d(in_planes)
        self.SiLU = nn.SiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出最后两维1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.SiLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.bnnorm = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.conv(x)  # 2-1
        x = self.bn(x)
        x = self.SiLU(x)
        res = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.sigmoid(out)
        out = x * attn + res
        return out


class ChannelAttention_1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_1, self).__init__()
        self.conv = nn.Conv2d(2 * in_planes, in_planes, 3, 1, 1)
        self.bn = nn.BatchNorm2d(in_planes)
        self.SiLU = nn.SiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出最后两维1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.SiLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.SiLU(x)
        res = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        result = x * self.sigmoid(out) + res
        return result


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention_1(in_planes)
        self.spatialattention = SpatialAttention()

    def transpose(self, x):
        B, HW, C = x.size()
        H = int(math.sqrt(HW))
        x = x.transpose(1, 2)
        x = x.view(B, C, H, H)
        return x

    def transpose_verse(self, x):
        B, C, H, W = x.size()
        HW = H * W
        x = x.view(B, C, HW)
        x = x.transpose(1, 2)
        return x

    def forward(self, x1, x2):
        x1 = self.transpose(x1)
        x2 = self.transpose(x2)
        x = torch.cat([x1, x2], dim=1)

        channel_attn = self.channelattention(x)
        x1 = x1 * channel_attn
        x2 = x2 * channel_attn
        x1 = self.transpose_verse(x1)
        x2 = self.transpose_verse(x2)
        return x1, x2


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3, 5, 7), s=1, equal_ch=True):
        super().__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))
