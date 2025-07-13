import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from thop import profile

import math
from functools import partial
from typing import Optional, Callable

import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as selective_scan_fn_v1
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as selective_scan_ref_v1
except:
    pass

# import sys
# sys.path.append('/home/star/liwei/PGIUN/')
from mamba_customer import mamba
from training.filtered_networks import LReLu, LReLu_regular, LReLu_torch

class Convolution(nn.Module):
    def __init__(self,
                 channels,
                 size,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu'
                 ):
        super(Convolution, self).__init__()

        self.channels = channels
        self.size  = size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm

        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.cutoff  = self.size / cutoff_den        
        self.halfwidth =  half_width_mult*self.size - self.size / cutoff_den
        
        #-----------------------------------------
        
        pad = (self.conv_kernel-1)//2
        self.convolution1 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels, 
                                           kernel_size=self.conv_kernel, stride = 1, 
                                           padding = pad)
        self.convolution2 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels, 
                                           kernel_size=self.conv_kernel, stride = 1, 
                                           padding = pad)
        
        if self.batch_norm:
            self.batch_norm1  = nn.BatchNorm2d(self.channels)
            self.batch_norm2  = nn.BatchNorm2d(self.channels)
        
        if activation == "cno_lrelu":

            self.activation  = LReLu(in_channels           = self.channels, #In _channels is not used in these settings
                                     out_channels          = self.channels,                   
                                     in_size               = self.size,                       
                                     out_size              = self.size,                       
                                     in_sampling_rate      = self.size,               
                                     out_sampling_rate     = self.size,             
                                     in_cutoff             = self.cutoff,                     
                                     out_cutoff            = self.cutoff,                  
                                     in_half_width         = self.halfwidth,                
                                     out_half_width        = self.halfwidth,              
                                     filter_size           = filter_size,       
                                     lrelu_upsampling      = lrelu_upsampling,
                                     is_critically_sampled = self.citically_sampled,
                                     use_radial_filters    = False)
        
        elif activation == "cno_lrelu_torch":
            self.activation = LReLu_torch(in_channels           = self.channels, #In _channels is not used in these settings
                                            out_channels          = self.channels,                   
                                            in_size               = self.size,                       
                                            out_size              = self.size,                       
                                            in_sampling_rate      = self.size,               
                                            out_sampling_rate     = self.size)
        elif activation == "lrelu":

            self.activation = LReLu_regular(in_channels           = self.channels, #In _channels is not used in these settings
                                            out_channels          = self.channels,                   
                                            in_size               = self.size,                       
                                            out_size              = self.size,                       
                                            in_sampling_rate      = self.size,               
                                            out_sampling_rate     = self.size)
        else:
            raise ValueError("Please specify different activation function")
        
    def forward(self, x):
        out = self.convolution1(x)
        if self.batch_norm:
            out = self.batch_norm1(out)
        out = self.activation(out)
        out = self.convolution2(out)
        if self.batch_norm:
            out = self.batch_norm2(out)
        return x + out

class Scanning(nn.Module):
    def __init__(self, dim, conv_mode="deepwise", resdiual=False, act="silu"):
        super(Scanning, self).__init__()

        self.fusionencoder = mamba(dim, bimamba_type="v2", conv_mode=conv_mode, act=act)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.resdiual = resdiual

    def forward(self, x_embed_conv):
        x1 = x_embed_conv
        x2 = x_embed_conv
        b, c, h, w = x_embed_conv.shape
        id1 = x1
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        queryed_x1 = self.fusionencoder(x1, x2)
        x1 = rearrange(queryed_x1, 'b (h w) c -> b c h w', h=h)
        if self.resdiual:
            x1 = x1 + self.skip_scale*id1
        return x1

class DC_layer_I(nn.Module):
    def __init__(self):
        super(DC_layer_I, self).__init__()

    def forward(self, image, kspace_data, mask):
        k_temp= torch.fft.fftshift(torch.fft.fft2(image))
        matrixones = torch.ones_like(mask.data)
        k_rec_dc = (matrixones - mask) * k_temp + kspace_data
        out = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k_rec_dc, dim=(-2, -1))))
        return out
    
class BasicBlock(nn.Module):
    def __init__(self, num_filter=64):
        super(BasicBlock, self).__init__()
        # self.norm = norm
        # num_filter = 64

        self.dc = DC_layer_I()
        self.conv_integration = Convolution(channels = num_filter, size = 256)
        self.Scanning_Integration = Scanning(dim = num_filter, conv_mode="orignal_dinner", resdiual=False, act="silu")

    def forward(self, x_embed, kspace_data, mask):
        x_embed = self.dc(x_embed, kspace_data, mask)

        x_embed_conv = self.conv_integration(x_embed)
        x_embed_conv_mamba = self.Scanning_Integration(x_embed_conv)
        x_embed = x_embed_conv_mamba + x_embed

        x_embed = self.dc(x_embed, kspace_data, mask)

        return x_embed

class LMO(nn.Module):
    def __init__(self, depth=6, embed_dim=64):
        super().__init__()
        dims = 1
        self.depth = depth

        self.Lift = nn.Conv2d(dims, embed_dim, kernel_size=3, padding=1)

        rec_blocks = []
        for _ in range(self.depth):
            rec_blocks.append(BasicBlock(num_filter = embed_dim))
        self.rec_blocks = nn.ModuleList(rec_blocks)

        self.Project = nn.Conv2d(embed_dim, dims, kernel_size=3, padding=1)

        self.dc = DC_layer_I()

    def forward(self, image, kspace_data, mask):

        x_embed = self.Lift(image)
        for i in range(self.depth):
            x_embed = self.rec_blocks[i](x_embed, kspace_data, mask)
        output = self.Project(x_embed)

        return self.dc(output+image, kspace_data, mask)
    

def make_model(args):
    model = LMO(args.depth, args.embed_dim)
    return model
