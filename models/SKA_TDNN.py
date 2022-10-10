#! /usr/bin/python
# -*- encoding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from collections import OrderedDict
from utils import PreEmphasis
from .specaugment import SpecAugment

class SEModule(nn.Module):
    
    def __init__(self, channels, bottleneck = 128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, kernel_sizes=[5,7], dilation=None, scale=8, group=1):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(width*scale)
        self.nums = scale - 1
        self.skconvs = nn.ModuleList([])
        for i in range(self.nums):
            convs = nn.ModuleList([])
            for k in kernel_sizes:
                convs += [
                    nn.Sequential(
                        OrderedDict(
                            [
                                  ('conv', nn.Conv1d(width, width, kernel_size=k, dilation=dilation, padding=k//2*dilation, groups=group)),
                                  ('relu', nn.ReLU()),
                                  ('bn', nn.BatchNorm1d(width)),
                            ]
                        )
                    )
                ]
            self.skconvs += [convs]
        self.skse = SKAttentionModule(channel=width, reduction=4, num_kernels=len(kernel_sizes))
        self.conv3 = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.se = SEModule(channels=planes)
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.skse(sp, self.skconvs[i])
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.skfwse = fwSKAttention(freq=40, kernels=[5,7], receptive=[5,7], dilations=[1,1], reduction=reduction, groups=1)
        self.skcwse = cwSKAttention(channel=128, kernels=[5,7], receptive=[5,7], dilations=[1,1], reduction=reduction, groups=1)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.skfwse(out)
        out = self.skcwse(out)
        out += residual
        out = self.relu(out)
        return out


class SKAttentionModule(nn.Module):

    def __init__(self, channel=128, reduction=4, L=16, num_kernels=2):
        super(SKAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.D = max(L, channel//reduction)
        self.fc = nn.Linear(channel, self.D)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([])
        for i in range(num_kernels):
            self.fcs += [nn.Linear(self.D, channel)]
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, convs):
        '''
        Input: [B, C, T]
        Split: [K, B, C, T]
        Fues: [B, C, T]
        Attention weight: [B, C, 1]
        Output: [B, C, T]
        '''
        bs, c, t = x.size()
        conv_outs = []
        for conv in convs:
            conv_outs += [conv(x)]
        feats = torch.stack(conv_outs,0)
        U = sum(conv_outs)
        S = self.avg_pool(U).view(bs, c)
        Z = self.fc(S)
        Z = self.relu(Z)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights += [(weight.view(bs, c, 1))]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights*feats).sum(0)
        return V


class fwSKAttention(nn.Module):
    
    def __init__(self, freq=40, channel=128, kernels=[3,5], receptive=[3,5], dilations=[1,1], reduction=8, groups=1, L=16):
        super(fwSKAttention, self).__init__()
        self.convs = nn.ModuleList([])
        for k, d, r in zip(kernels, dilations, receptive):
            self.convs += [
                nn.Sequential(
                    OrderedDict(
                        [
                            ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=r//2, dilation=d, groups=groups)),
                            ('relu', nn.ReLU()),
                            ('bn', nn.BatchNorm2d(channel)),
                        ]
                    )
                )
            ]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.D = max(L, freq//reduction)
        self.fc = nn.Linear(freq, self.D)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs += [nn.Linear(self.D, freq)]
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        '''
        Input: [B, C, F, T]
        Split: [K, B, C, F, T]
        Fues: [B, C, F, T]
        Attention weight: [K, B, 1, F, 1]
        Output: [B, C, F, T]
        '''
        bs, c, f, t = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs += [conv(x)]
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs).permute(0, 2, 3, 1)
        S = self.avg_pool(U).view(bs, f)
        Z = self.fc(S)
        Z = self.relu(Z)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights += [(weight.view(bs, 1, f, 1))]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights*feats).sum(0)
        return V


class cwSKAttention(nn.Module):
    
    def __init__(self, freq=40, channel=128, kernels=[3,5], receptive=[3,5], dilations=[1,1], reduction=8, groups=1, L=16):
        super(cwSKAttention, self).__init__()
        self.convs = nn.ModuleList([])
        for k, d, r in zip(kernels, dilations, receptive):
            self.convs += [
                nn.Sequential(
                    OrderedDict(
                        [
                            ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=r//2, dilation=d, groups=groups)),
                            ('relu', nn.ReLU()),
                            ('bn', nn.BatchNorm2d(channel)),
                        ]
                    )
                )
            ]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.D = max(L, channel//reduction)
        self.fc = nn.Linear(channel, self.D)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs += [nn.Linear(self.D, channel)]
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        '''
        Input: [B, C, F, T]
        Split: [K, B, C, F, T]
        Fuse: [B, C, F, T]
        Attention weight: [K, B, C, 1, 1]
        Output: [B, C, F, T]
        '''
        bs, c, f, t = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs += [conv(x)]
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = self.avg_pool(U).view(bs, c)
        Z = self.fc(S)
        Z = self.relu(Z)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights += [(weight.view(bs, c, 1, 1))]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights*feats).sum(0)
        return V


class SKA_TDNN(nn.Module):

    def __init__(self, block, C, model_scale, log_input=True, num_mels=80, num_out=192, resblock=ResBlock, context=True, **kwargs):
        super(SKA_TDNN, self).__init__()
        self.inplanes = 128
        self.log_input = log_input
        self.scale = model_scale
        self.context = context
        self.pooling_type = kwargs["pooling_type"]
        self.frt_conv1  = nn.Conv2d(1, 128, kernel_size=(3,3), stride=(2,1), padding=1)
        self.frt_bn1    = nn.BatchNorm2d(128)
        self.frt_block1 = resblock(128, 128, stride=(1,1))
        self.frt_block2 = resblock(128, 128, stride=(1,1))
        self.frt_conv2  = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(2,2), padding=1)
        self.frt_bn2    = nn.BatchNorm2d(128)
        self.conv1  = nn.Conv1d(128*20, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = block(C, C, kernel_size=3, dilation=2, scale=self.scale)
        self.layer2 = block(C, C, kernel_size=3, dilation=3, scale=self.scale)
        self.layer3 = block(C, C, kernel_size=3, dilation=4, scale=self.scale)
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        if self.context:
            attn_input = 1536 * 3
        else:
            attn_input = 1536
        print("self.pooling_type", self.pooling_type)
        # Channel- and Context-Dependent Statistics Pooling (CCSP)
        if self.pooling_type == "CCSP":
            attn_output = 1536
        # Attentive Statistics Pooling (ASP)
        elif self.pooling_type == "ASP":
            attn_output = 1
        else:
            raise ValueError("Undefined encoder")
        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
        )                
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=num_mels),
            )
        self.specaug = SpecAugment()
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, num_out)
        self.bn6 = nn.BatchNorm1d(num_out)

    def forward(self, x, aug):
        '''
        Input: [B, # samples of wavform (hop_length*max_frames)] -> log MelSpectrogram: [B, num_mels, max_frames]
        Output: [B, num_out]
        '''
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x)+1e-6
                if self.log_input:
                    x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if aug == True:
                    x = self.specaug(x)
                    
        x = x.unsqueeze(1)
        x = self.frt_conv1(x)
        x = self.relu(x)
        x = self.frt_bn1(x)        
        x = self.frt_block1(x)
        x = self.frt_block2(x)
        x = self.frt_conv2(x)
        x = self.relu(x)
        x = self.frt_bn2(x)
        
        x = x.reshape((x.size()[0], -1, x.size()[-1]))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)
        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)
        
        global_x = torch.cat((x,torch.mean(x,dim=2, keepdim=True).repeat(1, 1, x.size()[-1]), torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, x.size()[-1])), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2)*w, dim=2)-mu**2).clamp(min=1e-4))
        x = torch.cat((mu,sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        return x


def MainModel(eca_c=1024, eca_s=8, log_input=True, num_mels=80, num_out=192, **kwargs):
    model = SKA_TDNN(block=Bottle2neck, C=1024, model_scale=eca_s, log_input=log_input, num_mels=num_mels, num_out=num_out, resblock=ResBlock, context=True, **kwargs)
    return model