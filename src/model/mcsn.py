import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np


def make_model(args, parent=False):
    return MCSN(args)

        
def conv(in_channels, out_channels, kernel_size=3,stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size//2, groups=groups, bias=False)


def conv1x1(in_channels, out_channels, groups=1):
   
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=1, 
        groups=groups,stride=1)

class HFFB(nn.Module):
    """ Hierarchical Feature Fusion Block """
    def __init__(self, 
                 n_feats, 
                 group=1):
        super(HFFB, self).__init__()

        self.b1 = BNC(n_feats)
        self.b2 = BNC(n_feats)
        self.b3 = BNC(n_feats)
        self.c1 = conv1x1(2*n_feats, n_feats, 1) 
        self.c2 = conv1x1(3*n_feats, n_feats, 1) 
        self.c3 = conv1x1(4*n_feats, n_feats, 1) 

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3

class BNC(nn.Module):
    """ Basic Nested Cell"""

    def __init__(self, n_feats):#96,128,16

        super().__init__()
        '''  Multi-scale Features Fusion Block '''
        act = nn.ReLU(True)
        m = []  
        n = []               
        m.append(conv(n_feats, n_feats, kernel_size=5))
        m.append(act)
        n.append(conv(n_feats, n_feats, kernel_size=3)) 
        n.append(act)    
        
        self.compress = conv1x1(2*n_feats, n_feats, 1)         
        self.mid_compress = conv1x1(n_feats, n_feats, 1)       
        self.conv_5x5 = nn.Sequential(*m) 
        self.conv_3x3 = nn.Sequential(*n)
        self.csam = CSAM(n_feats,48)                 


    def forward(self, x):

        # middle opertion
        mid_feature = self.mid_compress(x)
        
        #  MSFFB opertion
        out_1 = self.conv_5x5(mid_feature)  # 3x3
        out_2 = self.conv_3x3(mid_feature)
        feature_1 = torch.cat((out_1, out_2), 1)
        out_1_2 = self.compress(feature_1)

        out_3 = self.conv_5x5(out_1_2)  # 3x3
        out_4 = self.conv_3x3(out_1_2)
        feature_2 = torch.cat((out_3, out_4), 1)
        out_3_4 = self.compress(feature_2)
        
        out_5 = self.conv_5x5(out_3_4)  # 3x3
        out_6 = self.conv_3x3(out_3_4)
        feature_3 = torch.cat((out_5, out_6), 1)
        out_5_6 = self.compress(feature_3)
        #  CSAM opertion
        csam_out = self.csam(mid_feature)
        out = out_5_6+csam_out  
        out += x              
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class CSAM(nn.Module):
    """Channel Shuffle Attention Mechanism """
    def __init__(self,channel, reduction=48):
        super(CSAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.soft = nn.Softmax(dim=1)
        self.channel2reduction = conv1x1(channel, reduction)
        self.reduction2channel = conv1x1(reduction, channel)
        self.GC = conv1x1(reduction, reduction, groups=3)

    def forward(self, x):
       
        c2r = self.channel2reduction(x)
        GC = self.GC(c2r)
        CF = channel_shuffle(GC, 3)
        r2c = self.reduction2channel(GC)
        avg = self.avg_pool(r2c)
        y = self.soft(avg)
        return x*y


class NC(nn.Module):
    """ Nested Cell"""
    def __init__(
        self, n_feats, kernel_size, act=nn.ReLU(True)):
        super(NB, self).__init__()        
        self.hffb = HFFB(n_feats)  
        
    def forward(self, x): 
           
        x_0 = self.hffb(x)
        x_1 = self.hffb(x_0)        
        out = self.hffb(x_1)          
        return out


class MCSN(nn.Module):
    def __init__(self, args):
        super(EDSR, self).__init__()
        # hyper-params
        self.args = args
        scale = args.scale[0]       
        n_feats = args.n_feats
        n_resblocks = args.n_resblocks
        kernel_size = 3        
        act = nn.ReLU(True)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])

        # define head module       
        head = []
        head.append(
            nn.Conv2d(args.n_colors, n_feats, 3, padding=3//2))        
        
        # define body module
        self.body = nn.ModuleList([NC(n_feats, kernel_size, act=act)for _ in range(n_resblocks)])
        
        # define tail module        
        m_tail = []        
        out_feats = scale*scale*args.n_colors
        m_tail.append(
            nn.Conv2d(n_feats, out_feats, 3, padding=3//2))
        m_tail.append(nn.PixelShuffle(scale))        

        # make object members
        self.compress = conv1x1(2*n_feats, n_feats, 1)
        self.conv1x1 = conv(n_feats, n_feats, kernel_size=1) 
        self.head = nn.Sequential(*head)
        self.tail = nn.Sequential(*m_tail)
        

    def forward(self, x):
        x = (x - self.rgb_mean.cuda()*255)/127.5
        inter_res = nn.functional.interpolate(x, scale_factor=3, mode='bilinear', align_corners=False)
        input = self.head(x) 
        
        res = self.conv1x1(input) 
        feature = input
        for i,l in enumerate(self.body):
          feature = self.compress(torch.cat((l(feature), res), 1))
        
        output = self.tail(feature)
        output += inter_res
        output = output*127.5 + self.rgb_mean.cuda()*255
        return output


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
